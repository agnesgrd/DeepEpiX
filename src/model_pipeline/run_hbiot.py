#!/usr/bin/env python3
"""
Prediction function for MEG spike detection.
Can be used as a standalone script or imported as a module.
"""

import os
import logging
from typing import Optional, Any, Dict, Optional, Union
import yaml
import traceback
from pathlib import Path

import pandas as pd
import torch
import lightning as L

from utils_biot.data import *
from utils_biot.models import *


logger = logging.getLogger(__name__)


## --- UTILITY FUNCTIONS --- ##
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file with optional validation.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If validation fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing failed for {config_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise yaml.YAMLError(f"Failed to parse YAML file {config_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to load configuration: {e}")

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    def expand_env_vars(obj: Any, max_recursion_depth: int = 20) -> Any:
        """Recursively expand environment variables in strings within the config."""
        if max_recursion_depth <= 0:
            raise ValueError(
                "Maximum recursion depth reached while expanding environment variables. Default is 20."
            )

        # if this is a dict, recurse into values
        if isinstance(obj, dict):
            return {
                k: expand_env_vars(v, max_recursion_depth - 1) for k, v in obj.items()
            }
        # if this is a list, recurse into items
        elif isinstance(obj, list):
            return [expand_env_vars(i, max_recursion_depth - 1) for i in obj]
        # if this is a string, expand env vars and user (~)
        elif isinstance(obj, str):
            expanded = os.path.expandvars(os.path.expanduser(obj))
            return expanded
        else:
            return obj

    config = expand_env_vars(config)
    return config


class MEGSpikeDetector(L.LightningModule):
    """Lightning module for spike detection in MEG data.

    This module handles training, validation, and testing of MEG spike detection models.
    All metrics computation and reporting is handled by the MetricsEvaluationCallback.

    Attributes:
        config: Configuration dictionary containing all component settings
        model: The neural network model for spike detection
        loss_fn: The loss function for training
        threshold: Classification threshold for binary predictions (updated by callback)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        input_shape: Tuple[int, int, int],
        log_dir: str,
        **_kwargs,
    ) -> None:
        """Initialize the Lightning module with configuration.

        Args:
            config: Configuration dictionary containing model, loss, optimizer settings
            input_shape: Shape of the input data (channels, time_points)
            log_dir: Directory for logging
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If required configuration keys are missing
            TypeError: If input_shape is not a tuple
        """
        # Input validation
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dictionary, got {type(config)}")

        required_keys = ["model", "loss", "optimizer", "data", "evaluation"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be a tuple of length 3, got {input_shape}"
            )
        super().__init__()
        logger.info("Initializing ConfigurableLightningModule")
        self.config = config
        self.log_dir = log_dir
        self.input_shape = input_shape
        config["model"][config["model"]["name"]]["input_shape"] = list(input_shape)
        config["model"][config["model"]["name"]]["log_dir"] = log_dir
        self.save_hyperparameters(config)

        # Create model and processing flags
        self.contextual = config["model"][config["model"]["name"]].get(
            "contextual", False
        )
        self.sequential_processing = config["model"][config["model"]["name"]].get(
            "sequential_processing", False
        )
        if config["model"]["name"] == "BIOT":
            self.model = BIOTClassifier(**config["model"]["BIOT"])
        elif config["model"]["name"] == "BIOTHierarchical":
            self.model = BIOTHierarchicalClassifier(
                **config["model"]["BIOTHierarchical"]
            )
        else:
            raise ValueError(f"Unsupported model name: {config['model']['name']}")

        # Store optimizer and scheduler configs for later use
        self.optimizer_config = config["optimizer"]
        self.scheduler_config = config.get("scheduler", None)
        logger.info(f"Optimizer config: {self.optimizer_config}")
        if self.scheduler_config:
            logger.info(f"Scheduler config: {self.scheduler_config}")

        # Temperature scaling configuration and validation
        self.temperature_scaling_enabled = config["evaluation"].get(
            "temperature_scaling", False
        )
        # Classification threshold (can be updated by MetricsEvaluationCallback if threshold_optimization=True)
        self.threshold = config["evaluation"].get("default_threshold", 0.5)

        # Temperature scaling for calibrated predictions (1.0 = no scaling)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.temperature.requires_grad = (
            False  # Only optimized during temperature scaling phase
        )

        # Log the graph to TensorBoard
        logger.info("ConfigurableLightningModule initialized successfully")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore threshold and temperature from checkpoint if available."""
        super().on_load_checkpoint(checkpoint)
        print(checkpoint["hyper_parameters"])
        if "hyper_parameters" in checkpoint:
            if "threshold" in checkpoint["hyper_parameters"]:
                self.threshold = checkpoint["hyper_parameters"]["threshold"]
                logger.info(f"Restored threshold from checkpoint: {self.threshold:.4f}")
            if "temperature" in checkpoint["hyper_parameters"]:
                temp_value = checkpoint["hyper_parameters"]["temperature"]
                if isinstance(temp_value, torch.Tensor):
                    self.temperature.data = temp_value.to(self.temperature.device)
                else:
                    self.temperature.data = torch.tensor(
                        [temp_value], device=self.temperature.device
                    )
                logger.info(
                    f"Restored temperature from checkpoint: {self.temperature.item():.4f}"
                )

    def apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits for calibrated predictions.

        Temperature scaling divides logits by a learned temperature parameter T:
        - T > 1: Makes predictions less confident (smoother probabilities)
        - T = 1: No scaling (default)
        - T < 1: Makes predictions more confident (sharper probabilities)

        Args:
            logits: Raw model logits [batch_size, n_windows, n_classes] or [batch_size, n_windows]

        Returns:
            Temperature-scaled logits of the same shape
        """
        return logits / self.temperature

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor],
        window_mask: Optional[torch.Tensor] = None,
        force_sequential: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the model with contextual and sequential processing support.

        Handles different processing modes:
        - Contextual: Pass full sequence [batch_size, n_windows, n_channels, n_timepoints] to model
        - Non-contextual + batch mode: Reshape to [BxN_window, n_channels, n_timepoints]
        - Non-contextual + sequential: Loop through windows individually

        Args:
            x: Input tensor of shape [batch_size, n_windows, n_channels, n_timepoints]
            channel_mask: Optional channel mask tensor (B, C) where True=valid, False=masked.
            window_mask: Optional window mask tensor (B, N) where True=valid, False=masked.
            force_sequential: Whether to force sequential processing mode.
            *args: Additional positional arguments to pass to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            torch.Tensor: Output logits of shape [batch_size, n_windows, n_classes]
        """
        if self.contextual:
            # Contextual models process the full sequence with temporal context
            return self.model(x, channel_mask, window_mask, *args, **kwargs)

        # Non-contextual processing for window-level models
        batch_size, n_windows, n_channels, n_timepoints = x.shape
        if self.sequential_processing or force_sequential:
            # Sequential mode: Process each window individually in a loop
            window_outputs = []
            for seg_idx in range(n_windows):
                window = x[:, seg_idx, :, :]  # [batch_size, n_channels, n_timepoints]
                window_output = self.model(
                    window, channel_mask, *args, **kwargs
                )  # [batch_size, n_classes]
                window_outputs.append(
                    window_output.unsqueeze(1)
                )  # [batch_size, 1, n_classes]

            return torch.cat(
                window_outputs, dim=1
            )  # [batch_size, n_windows, n_classes]
        else:
            # Batch mode: Reshape to process all windows simultaneously
            x = x.view(
                batch_size * n_windows, n_channels, n_timepoints
            )  # [B×N_window, n_channels, n_timepoints]
            channel_mask = (
                channel_mask.repeat_interleave(n_windows, dim=0)
                if channel_mask is not None
                else None
            )  # [B×N_window, n_channels]
            if "unknown_mask" in kwargs and kwargs["unknown_mask"] is not None:
                unknown_mask = kwargs["unknown_mask"].repeat_interleave(
                    n_windows, dim=0
                )
                kwargs["unknown_mask"] = unknown_mask  # [B×N_window, n_channels]
            result = self.model(
                x, channel_mask, *args, **kwargs
            )  # [B×N_window, n_classes]
            return result.view(
                batch_size, n_windows, -1
            )  # [batch_size, n_windows, n_classes]

    def predict_step(self, batch, batch_idx):
        """Perform a single prediction step.

        Args:
            batch: Batch data (X, window_mask, channel_mask, metadata) where:
                - X: Input MEG data [batch_size, n_windows, n_channels, n_timepoints]
                - window_mask: Valid window mask [batch_size, n_windows] - 1=valid, 0=padded
                - channel_mask: Valid channel mask [batch_size, n_channels] - 1=valid, 0=masked
                - metadata: Sample metadata for result export
            batch_idx: Index of the batch

        Returns:
            Dictionary containing predictions, probabilities, and metadata
        """
        X, window_mask, channel_mask, metadata = batch

        unknown_mask = None
        if not metadata[0]["USE_REFERENCE_CHANNELS"] and channel_mask is not None:
            # Channel mask is actually true everywhere but for padded channels
            # We actually don't know if good channels are really good at inference time, we just know that this is real data
            # So we use an unknown mask that is all True where channel_mask is given
            unknown_mask = torch.ones_like(channel_mask, dtype=torch.bool)

        # Forward pass with batch-aware channel mask
        logits = self.forward(
            X,
            channel_mask=channel_mask,
            window_mask=window_mask,
            unknown_mask=unknown_mask,
        )

        # Apply temperature scaling and compute calibrated probabilities
        scaled_logits = self.apply_temperature_scaling(logits)
        probs = torch.sigmoid(scaled_logits).cpu().detach()

        # Apply threshold for binary predictions
        predictions = (probs >= self.threshold).float()

        # Prepare outputs
        outputs = {
            "logits": logits.cpu().detach(),
            "probs": probs,
            "predictions": predictions,
            "batch_size": X.shape[0],
            "n_windows": X.shape[1] if len(X.shape) > 2 else 1,
            "batch_idx": batch_idx,
            "metadata": metadata if metadata else {},
            "channel_mask": (
                channel_mask.cpu().detach().float().numpy()
                if channel_mask is not None
                else None
            ),
            "window_mask": (
                window_mask.cpu().detach().float().numpy()
                if window_mask is not None
                else None
            ),
        }
        return outputs

    def _collect_batch_outputs(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict],
        logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """Collect outputs for a single batch.

        Args:
            batch: Input batch (X, y, window_mask, channel_mask, metadata)
            logits: Model output logits for the batch

        Returns:
            Dictionary with batch outputs including per-window losses
        """
        X, y, window_mask, _channel_mask, metadata = batch

        # Apply temperature scaling for calibrated probabilities
        scaled_logits = self.apply_temperature_scaling(logits)

        # Compute per-window BCE loss without reduction for analysis (using scaled logits)
        # Note: Both scaled_logits and y are [B, N] for binary classification
        per_window_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scaled_logits, y, reduction="none"
        )
        probs = torch.sigmoid(scaled_logits)

        return {
            "logits": logits.cpu().detach().float().numpy(),
            "probs": probs.cpu().detach().float().numpy(),
            "predictions": (probs >= self.threshold).float().cpu().detach().numpy(),
            "gt": y.cpu().detach().float().numpy(),
            "mask": window_mask.cpu().detach().float().numpy(),
            "losses": per_window_loss.cpu()
            .detach()
            .float()
            .numpy(),  # Per-window losses
            "metadata": metadata if metadata else {},
            "batch_size": X.shape[0],
            "n_windows": X.shape[1],
        }


def test_model(
    model_name,
    model_type,
    subject,
    output_path,
    threshold,
    adjust_onset=True,
    channel_groups=None,
) -> pd.DataFrame:
    """Predict spikes in a MEG file.

    Args:
        file_path: Path to the MEG file (.fif or .ds)
        config_path: Path to the configuration file
        checkpoint_path: Path to model checkpoint
        reference_channels_path: Path to good channels file (pickle format)
        compute_gfp_peaks: Whether to compute GFP peaks for onset adjustment, else use the center of windows
        output_csv: Path to save CSV results (optional)

    Returns:
        DataFrame with columns: onset, duration, probas

    Example:
        df = predict_spikes(
            file_path="data/patient_001.fif",
            config_path="configs/default_config.yaml"
        )
        print(df.head())
           onset  duration    probas
        0  12.34       0.0  0.823456
        1  15.67       0.0  0.234567
        ...
    """
    # Params
    if os.path.basename(model_name) == "transformer.ckpt":
        config_path = "./utils_biot/config/hparams.yaml"
        checkpoint_path = "./utils_biot/config/epoch=16-val_pr_auc=0.42.ckpt"
        reference_channels_path = "./utils_biot/config/reference_channels.pkl"

    elif os.path.basename(model_name) == "hbiot.ckpt":
        config_path = "./utils_biot/config_hbiot/hparams.yaml"
        checkpoint_path = "./utils_biot/config_hbiot/epoch=23-val_pr_auc=0.48.ckpt"
        reference_channels_path = "./utils_biot/config_hbiot/reference_channels.pkl"
    file_path = subject
    compute_gfp_peaks = adjust_onset
    output_csv = os.path.join(
        output_path, f"{os.path.basename(model_name)}_predictions.csv"
    )

    # Setup logging
    logging.basicConfig(level=logging.WARNING)

    # Resolve paths relative to this script's location if they are relative
    script_dir = Path(__file__).parent.resolve()

    def resolve_path(path_str: str) -> Path:
        """Convert relative paths to absolute paths relative to script location."""
        path = Path(path_str)
        if not path.is_absolute():
            path = script_dir / path
        return path.resolve()

    # Resolve all paths
    file_path = str(resolve_path(file_path))
    config_path = str(resolve_path(config_path))
    checkpoint_path = str(resolve_path(checkpoint_path))
    if reference_channels_path is not None:
        reference_channels_path = str(resolve_path(reference_channels_path))
    if output_csv is not None:
        output_csv = str(resolve_path(output_csv))

    config = load_config(config_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if reference_channels_path is not None and not os.path.exists(
        reference_channels_path
    ):
        raise FileNotFoundError(
            f"Reference channels file not found: {reference_channels_path}"
        )

    logger.info(f"Using checkpoint: {checkpoint_path}")
    input_shape = tuple(config["model"][config["model"]["name"]]["input_shape"])

    # Load model from checkpoint
    model = MEGSpikeDetector.load_from_checkpoint(
        checkpoint_path, config=config, input_shape=input_shape, log_dir=None
    )

    print(f"Best threshold from training: {model.threshold:.4f}")

    # Create trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # Create prediction datamodule
    prediction_config = {
        "file_path": file_path,
        "reference_channels_path": reference_channels_path,
        "dataset_config": config["data"][config["data"]["name"]]["dataset_config"],
        "dataloader_config": config["data"][config["data"]["name"]][
            "dataloader_config"
        ],
    }

    datamodule = PredictionDataModule(**prediction_config)
    datamodule.setup(stage="predict")

    # Run prediction
    predictions = trainer.predict(model, datamodule=datamodule)

    # Process predictions
    results = []
    assert predictions is not None, "No predictions returned from the model."
    for batch_predictions in predictions:
        if not isinstance(batch_predictions, dict):
            continue
        batch_metadata = batch_predictions.get("metadata", [])
        probs = batch_predictions["probs"]
        mask = batch_predictions.get("mask", None)

        # Handle different tensor shapes
        if probs.dim() == 1:
            # Single sample case: reshape to [1, n_windows]
            probs = probs.unsqueeze(0)
        elif probs.dim() == 3:
            # Remove last dimension if it's singleton (n_classes=1)
            probs = probs.squeeze(-1)

        # Process each sample in the batch
        batch_size = probs.shape[0]
        for i in range(batch_size):
            if i >= len(batch_metadata):
                continue

            metadata = batch_metadata[i]

            if "window_times" in metadata:
                # Chunked prediction using unified metadata naming
                window_times = metadata["window_times"]
                n_windows = metadata["n_windows"]  # Unified naming convention

                # Get probabilities for this sample
                # Assume binary classification: squeeze to 1D array
                sample_probs = probs[i].squeeze()
                sample_mask = mask[i] if mask is not None and mask.ndim == 2 else None

                for j in range(n_windows):
                    # Skip odd indexes due to 50% overlap
                    if j % 2 == 1:
                        continue

                    if j >= len(window_times):
                        continue

                    # Skip masked windows
                    if sample_mask is not None and sample_mask[j] == 0:
                        continue

                    seg_time = window_times[j]
                    prob = (
                        float(sample_probs[j].item())
                        if sample_probs.ndim == 1
                        else float(sample_probs[j, 0].item())
                    )  # Assuming binary classification

                    # Calculate onset (use GFP peak for precise timing if requested)
                    onset = (
                        seg_time["center_time"]
                        if not compute_gfp_peaks
                        else seg_time["peak_time"]
                    )
                    results.append(
                        {
                            "onset": onset,
                            "duration": 0,  # Duration is 0 for point events
                            "probas": prob,
                        }
                    )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV if requested
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to: {output_csv}")

    return df


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Predict spikes in MEG data")
#     parser.add_argument(
#         "--file_path",
#         type=str,
#         required=True,
#         help="Path to MEG file (.fif, .ds, of 4D format)",
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="./utils_biot/config/hparams.yaml",
#         help="Path to configuration file",
#     )
#     parser.add_argument(
#         "--checkpoint",
#         type=str,
#         default="./utils_biot/config/epoch=16-val_pr_auc=0.42.ckpt",
#         help="Path to model checkpoint",
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         default=None,
#         help="Output CSV file path (auto-generated if not provided)",
#     )
#     parser.add_argument(
#         "--no-gfp",
#         action="store_true",
#         help="Skip GFP peak calculation for onset adjustment",
#     )

#     args = parser.parse_args()

#     # Run prediction
#     df = predict_spikes(
#         file_path=args.file_path,
#         config_path=args.config,
#         checkpoint_path=args.checkpoint,
#         compute_gfp_peaks=not args.no_gfp,
#         output_csv=args.output,
#     )

#     # Print summary
#     print(f"\nPrediction Summary:")
#     print(f"Input file: {args.file_path}")
#     print(f"Total windows: {len(df)}")
#     print(f"Mean probability: {df['probas'].mean():.4f}")
#     print(f"Output file: {args.output}")
