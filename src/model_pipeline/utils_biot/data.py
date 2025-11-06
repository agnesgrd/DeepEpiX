import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import mne
from mne.io.base import BaseRaw
import os
import pathlib
from scipy.ndimage import median_filter
import torch
from scipy import stats
import logging
import lightning as L
import pickle


logger = logging.getLogger(__name__)


## --- Dataset and Datamodules ---
def load_and_process_meg_data(
    file_path: str,
    config: Dict[str, Any],
    good_channels: Optional[List[str]] = None,
    n_channels: int = 275,
    close_raw: bool = True
) -> Tuple[BaseRaw, np.ndarray, Dict[str, Any]]:
    """Load and process MEG data for prediction.
    
    Args:
        file_path: Path to the MEG data file.
        config: Configuration dictionary with preprocessing parameters.
        good_channels: List of channels that should be present. If None, use all available channels (useful for inference on new systems).
        n_channels: Number of MEG channels to use (default: 275) to enforce consistent input size
        close_raw: Whether to close the MNE Raw object after processing to free memory.
            
    Returns:
        Tuple containing:
            - raw: MNE Raw object after processing.
            - data: Processed MEG data array (n_channels, n_timepoints).
            - channel_info: loc information and channel mask.
    """
    USE_REFERENCE_CHANNELS = False
    try:
        if ".ds" in file_path:
            raw = mne.io.read_raw_ctf(file_path, preload=False).pick(picks=['meg'], exclude='bads').load_data()
            USE_REFERENCE_CHANNELS = True
        elif ".fif" in file_path:
            raw = mne.io.read_raw_fif(file_path, preload=False).pick(picks=['meg'], exclude='bads').load_data()
        elif os.path.isdir(file_path):
            subject_path = pathlib.Path(file_path)
            files = list(subject_path.glob("*"))
            raw_fname = next((f for f in files if "rfDC" in f.name and f.suffix == ""), None)
            config_fname = next((f for f in files if "config" in f.name.lower()), None)
            hs_fname = next((f for f in files if "hs" in f.name.lower()), None)

            if not all([raw_fname, config_fname, hs_fname]):
                raise ValueError("Missing BTi raw/config/hs files.")

            raw = mne.io.read_raw_bti(
                pdf_fname=str(raw_fname),
                config_fname=str(config_fname),
                head_shape_fname=str(hs_fname),
                preload=False,
                verbose=False,
            ).pick(picks=['meg'], exclude='bads').load_data()
        else:
            raise ValueError("Unsupported file type for subject path.")

        # Special case handling for specific file patterns
        for pattern, channels in config.get('special_case_handling', {}).items():
            if pattern in file_path:
                # Drop problematic channels before selecting
                channels_to_drop = [ch for ch in channels if ch in raw.ch_names]
                if channels_to_drop:
                    raw.drop_channels(channels_to_drop)
                    logger.info(f"Dropped {len(channels_to_drop)} special case channels for pattern '{pattern}'")
        
        if good_channels is None or not USE_REFERENCE_CHANNELS:
            good_channels = list(raw.ch_names)  # Use all available channels if no reference provided
            # sample n_channels from good_channels if more than n_channels are available
            if len(good_channels) > n_channels:
                good_channels = good_channels[:n_channels]
            logger.info(f"No good_channels provided, using all {len(good_channels)} available channels")
        
        # Select channels based on good channels and location information
        raw, channel_info = select_channels(raw, good_channels)
        
        # Resample and filter
        if raw.info['sfreq'] != config['sampling_rate']:
            raw.resample(sfreq=config['sampling_rate'])
        raw.filter(l_freq=config.get('l_freq', 0.5), h_freq=config.get('h_freq', 95.0))
        
        if config.get('notch_freq', 50.0) > 0:
            freqs = np.arange(config['notch_freq'], config['sampling_rate']/2, config['notch_freq']).tolist()
            raw.notch_filter(freqs=freqs)
        
        # Get raw data from MNE (in order of selected_channels)
        raw_data = np.array(raw.get_data())  # Shape: (n_selected_channels, n_timepoints)
        n_timepoints = raw_data.shape[1]


        # Now normalize and filter
        raw_data = normalize_data(raw_data, config.get('normalization', {'method': 'robust_zscore', 'axis': None}))

        if config.get('median_filter_temporal_window_ms', 0) > 0:
            raw_data = apply_median_filter(raw_data, config['sampling_rate'], config['median_filter_temporal_window_ms'])

        if close_raw:
            raw.close()
        
        # Reorder data to match good_channels exactly
        # This ensures all samples in batch have data at same positions
        # Position i in data array ALWAYS represents good_channels[i]
        num_channels = max(n_channels, len(good_channels))
        data = np.zeros((num_channels, n_timepoints), dtype=raw_data.dtype)
        channel_mask = torch.zeros(num_channels, dtype=torch.bool)

        # Create index mapping for efficiency
        good_channels_index = {ch: i for i, ch in enumerate(good_channels)}

        # Place each channel's data at its correct position
        for ch_idx, ch_name in enumerate(channel_info['selected_channels']):
            if ch_name in good_channels_index:
                target_idx = good_channels_index[ch_name]
                data[target_idx, :] = raw_data[ch_idx, :]
                channel_mask[target_idx] = True
            else:
                logger.warning(f"Channel {ch_name} not in good_channels reference - skipping")

        n_valid = channel_mask.sum().item()
        logger.debug(f"Channel masking (file: {os.path.basename(file_path)}): {n_valid}/{len(good_channels)} valid channels")

        # Store channel mask for batch collation
        channel_info['channel_mask'] = channel_mask
        channel_info['USE_REFERENCE_CHANNELS'] = USE_REFERENCE_CHANNELS and (good_channels is not None)

        return raw, data, channel_info

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise


def select_channels(raw: BaseRaw, good_channels: List[str]) -> Tuple[BaseRaw, Dict[str, Any]]:
    """Select channels ensuring consistent ordering across all samples for batch compatibility.

    Args:
        raw: MNE Raw object containing MEG data
        good_channels: ORDERED list of reference channel names (defines canonical ordering)

    Returns:
        Tuple of (processed_raw, channel_info) where channel_info contains:
            - 'loc': Dictionary mapping selected channel names to coordinates (legacy)
            - 'selected_channels': ORDERED list of channel names matching good_channels order
            - 'n_selected': Number of channels actually present in raw data
            - 'n_with_coordinates': Number of channels with coordinate info (legacy)
    """
    logger.debug(f"Raw channels available: {len(raw.ch_names)} channels")
    logger.debug(f"Good channels reference: {len(good_channels)} channels")
    
    # Get available MEG channels from the raw data
    available_channels = set(raw.ch_names)  # Use set for O(1) lookup
    logger.debug(f"Available MEG channels: {len(available_channels)}")

    # Ensure batch consistency - all samples have same channel ordering
    selected_channels = [ch for ch in good_channels if ch in available_channels]

    if len(selected_channels) == 0:
        raise ValueError(f"No channels from good_channels found in raw data! "
                        f"Raw has: {list(raw.ch_names)[:10]}..., "
                        f"Expected: {good_channels[:10]}...")

    logger.debug(f"Selected {len(selected_channels)}/{len(good_channels)} channels from reference list")

    # Pick only the selected channels in the raw object
    raw = raw.pick_channels(selected_channels)
    channel_info = {
        'ch_info': raw.info['chs'],  # Full channel info from MNE},
        'selected_channels': selected_channels,
    }
    return raw, channel_info


def normalize_data(data: np.ndarray, norm_config: Dict, eps: Optional[float] = None) -> np.ndarray:
    """Normalize data using specified method."""
    if eps is None:
        eps = norm_config.get('epsilon', 1e-20)
    
    method = norm_config.get('method', 'robust_zscore')
    axis = norm_config.get('axis', None)
    
    if method == 'percentile':
        percentile = norm_config.get('percentile', 95)
        if not (0 < percentile < 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        q = np.percentile(np.abs(data), percentile, axis=axis, keepdims=True)
        return data / (q + eps)
    
    elif method == 'robust_normalize':
        median = np.median(data, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        return (data - median) / (iqr + eps)
    
    elif method == 'robust_zscore':
        median = np.median(data, axis=axis, keepdims=True)
        mad = stats.median_abs_deviation(data, axis=axis)  # type: ignore
        if axis is not None:
            mad = np.expand_dims(mad, axis=axis)
        return (data - median) / (mad + eps)  # type: ignore
    
    elif method == 'zscore':
        return (data - np.mean(data, axis=axis, keepdims=True)) / (np.std(data, axis=axis, keepdims=True) + eps)
    
    elif method == 'minmax':
        min_v = np.min(data, axis=axis, keepdims=True)
        max_v = np.max(data, axis=axis, keepdims=True)
        return (data - min_v) / (max_v - min_v + eps)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. Supported methods: percentile, zscore, minmax, robust_normalize, robust_zscore")


def apply_median_filter(data: np.ndarray, sfreq: float, temporal_window_ms: float) -> np.ndarray:
    """Apply median filter with adaptive kernel size based on sampling frequency.
    
    Args:
        data: MEG data array of shape (n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        temporal_window_ms: Temporal smoothing window in milliseconds
        
    Returns:
        Filtered data with same shape as input
    """
    if temporal_window_ms <= 0:
        return data
    
    # Calculate kernel size based on sampling frequency and temporal window
    kernel_samples = int(temporal_window_ms * sfreq / 1000)
    # Ensure odd kernel size for symmetric filtering
    kernel_size = kernel_samples if kernel_samples % 2 == 1 else kernel_samples + 1
    
    # Apply median filter along time axis (axis=1) for each channel
    return median_filter(data, size=(1, kernel_size))


def create_windows(
    meg_data: np.ndarray,
    sampling_rate: float,
    window_duration_s: float,
    window_overlap: float,
) -> np.ndarray:
    """Create windows from MEG data.
    
    Args:
        meg_data: MEG data array (n_channels, n_timepoints)
        sampling_rate: Sampling rate in Hz
        window_duration_s: Duration of each window in seconds
        window_overlap: Overlap between windows (0.0 to 1.0)
        
    Returns:
        Array of windows with shape (n_windows, n_channels, n_samples_per_window)
    """
    window_duration_samples = int(window_duration_s * sampling_rate)
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))
    
    windows = []
    seg_start = 0
    
    while seg_start + window_duration_samples <= meg_data.shape[1]:
        seg_end = seg_start + window_duration_samples
        windows.append(meg_data[:, seg_start:seg_end])
        seg_start += window_step
    
    return np.array(windows)


def compute_gfp(meg_data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute Global Field Power (GFP) from MEG data.
    
    Args:
        meg_data: MEG data array, shape (n_channels, n_timepoints) or (n_timepoints, n_channels)
        axis: Axis along which channels are located (0 for first dim, 1 for second dim)
        
    Returns:
        GFP values, shape (n_timepoints,)
    """
    # Compute GFP as standard deviation across channels
    gfp = np.std(meg_data, axis=axis)
    return gfp


def find_gfp_peak_in_window(
    meg_data: np.ndarray,
    window_start: int,
    window_end: int,
    sampling_rate: float
) -> Tuple[int, float]:
    """Find the peak GFP within a window.
    
    Args:
        meg_data: MEG data array, shape (n_channels, n_timepoints)
        window_start: Start sample of the window
        window_end: End sample of the window
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (peak_sample, peak_time_in_seconds)
    """
    # Extract window
    window_data = meg_data[:, window_start:window_end]
    
    # Compute GFP
    gfp = compute_gfp(window_data, axis=0)
    
    # Find peak
    peak_idx = np.argmax(gfp)
    peak_sample = window_start + peak_idx
    peak_time = peak_sample / sampling_rate
    
    return int(peak_sample), float(peak_time)


class PredictDataset(torch.utils.data.Dataset):
    """Dataset for prediction using sequential chunk extraction.

    Returns:
        Tuple of (chunk_data, metadata) with unified metadata convention including
        chunk_onset_sample, chunk_idx, window_times, etc.
    """

    def __init__(
        self,
        file_path: str,
        dataset_config: Dict[str, Any],
        n_channels: int = 275,
        reference_channels_path: Optional[str] = None,
    ):
        """Initialize prediction dataset with sequential chunk extraction.

        Args:
            file_path: Path to the MEG file (.fif or .ds).
            dataset_config: Configuration for data processing.
            n_channels: Number of MEG channels (default: 275) for consistent input size.
        """
        self.file_path = file_path
        self.dataset_config = dataset_config
        self.n_channels = n_channels
        if reference_channels_path is not None:
            with open(reference_channels_path, 'rb') as f:
                self.reference_channels = pickle.load(f)
        else:
            self.reference_channels = None

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing PredictDataset for {file_path}")

        # Load and preprocess the recording once
        self.meg_data = None
        self.channel_info = None
        self.sampling_rate = None
        self.n_chunks = 0

        self._load_recording()

    def _load_recording(self):
        """Load and preprocess the MEG recording once."""
        try:
            raw, self.meg_data, self.channel_info = load_and_process_meg_data(
                self.file_path,
                self.dataset_config,
                good_channels=self.reference_channels,
                n_channels=self.n_channels,
            )

            self.sampling_rate = raw.info['sfreq']
            raw.close()

            self.all_windows = create_windows(
                self.meg_data,
                self.sampling_rate,
                self.dataset_config['window_duration_s'],
                self.dataset_config.get('window_overlap', 0.0),
            )

            num_context_windows = self.dataset_config['n_windows']
            total_windows = len(self.all_windows)
            self.n_chunks = (total_windows + num_context_windows - 1) // num_context_windows

            self.logger.info(f"Loaded recording: {self.meg_data.shape[1]} samples, "
                           f"{total_windows} windows, {self.n_chunks} chunks")

        except Exception as e:
            self.logger.error(f"Error loading file {self.file_path}: {e}")
            raise

    def __len__(self) -> int:
        """Return number of chunks."""
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extract a chunk sequentially for prediction.

        Args:
            idx: Chunk index (0-based).

        Returns:
            Tuple of (chunk_data, metadata) with chunk_data as tensor of shape
            (n_windows, n_channels, window_samples) and metadata dictionary.
        """
        num_context_windows = self.dataset_config['n_windows']
        window_duration_samples = int(self.dataset_config['window_duration_s'] * self.sampling_rate)
        window_overlap = self.dataset_config.get('window_overlap', 0.0)
        window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

        start_window_idx = idx * num_context_windows
        end_window_idx = min(start_window_idx + num_context_windows, len(self.all_windows))

        windows = self.all_windows[start_window_idx:end_window_idx]

        chunk_onset_sample = start_window_idx * window_step

        window_times = []
        for local_idx, global_idx in enumerate(range(start_window_idx, end_window_idx)):
            window_start = global_idx * window_step
            window_end = window_start + window_duration_samples
            window_center = window_start + window_duration_samples // 2

            peak_sample, peak_time = find_gfp_peak_in_window(
                self.meg_data, window_start, window_end, self.sampling_rate
            )

            window_times.append({
                'start_sample': int(window_start),
                'end_sample': int(window_end),
                'center_sample': int(window_center),
                'peak_sample': int(peak_sample),
                'start_time': float(window_start / self.sampling_rate),
                'end_time': float(window_end / self.sampling_rate),
                'center_time': float(window_center / self.sampling_rate),
                'peak_time': float(peak_time),
                'window_idx_in_chunk': local_idx,
                'global_window_idx': global_idx,
            })

        metadata = {
            'chunk_onset_sample': chunk_onset_sample,
            'chunk_offset_sample': chunk_onset_sample + len(windows) * window_step + (window_duration_samples - window_step),
            'chunk_duration_samples': len(windows) * window_step + (window_duration_samples - window_step),
            'chunk_idx': idx,
            'start_window_idx': start_window_idx,
            'end_window_idx': end_window_idx,
            'n_windows': len(windows),
            'window_times': window_times,
            'window_duration_s': self.dataset_config['window_duration_s'],
            'window_duration_samples': window_duration_samples,
            'file_name': self.file_path,
            'patient_id': self.file_path.split('/')[-2] if '/' in self.file_path else 'unknown',
            'channel_mask': self.channel_info.get('channel_mask', None) if self.channel_info else None,
            'selected_channels': self.channel_info.get('selected_channels', []) if self.channel_info else [],
            'n_selected_channels': len(self.channel_info.get('selected_channels', [])) if self.channel_info else 0,
            'USE_REFERENCE_CHANNELS': self.channel_info.get('USE_REFERENCE_CHANNELS', False) if self.channel_info else False,
            'sampling_rate': self.sampling_rate,
            'is_test_set': False,
            'extraction_mode': 'sequential',
        }

        return torch.tensor(windows, dtype=torch.float32), metadata


def predict_collate_fn(batch):
    """Collate function for prediction batches with padding and masking.

    Handles batches with (data, metadata) tuples from PredictDataset.
    Pads variable-length sequences and extracts channel masks from metadata.

    Args:
        batch: List of (data, metadata) tuples from dataset

    Returns:
        Tuple of (batch_data, batch_window_mask, batch_channel_mask, metadata_list)
    """
    # For chunked prediction: (data, metadata) - use padded collate for consistency with training
    data_list = [item[0] for item in batch]
    metadata_list = [item[1] for item in batch]

    # Pad data to same length as training (handles variable chunk sizes)
    seg_counts = [d.shape[0] for d in data_list]
    max_segs = max(seg_counts)

    padded_data, window_mask_list = [], []
    channel_mask_list = []

    for i, data in enumerate(data_list):
        n = data.shape[0]
        pad = max_segs - n
        padded_data.append(torch.cat([data, torch.zeros(pad, *data.shape[1:])]))
        window_mask_list.append(torch.cat([torch.ones(n), torch.zeros(pad)]))

        # Extract channel mask from metadata
        if metadata_list and i < len(metadata_list):
            ch_mask = metadata_list[i].get('channel_mask', None)
            if ch_mask is not None:
                if isinstance(ch_mask, list):
                    ch_mask = torch.tensor(ch_mask, dtype=torch.bool)
                elif not isinstance(ch_mask, torch.Tensor):
                    ch_mask = torch.tensor(ch_mask, dtype=torch.bool)
                channel_mask_list.append(ch_mask)
            else:
                n_channels = data.shape[1] if len(data.shape) > 1 else 1
                channel_mask_list.append(torch.ones(n_channels, dtype=torch.bool))
        else:
            n_channels = data.shape[1] if len(data.shape) > 1 else 1
            channel_mask_list.append(torch.ones(n_channels, dtype=torch.bool))

    batch_data = torch.stack(padded_data, dim=0)
    batch_window_mask = torch.stack(window_mask_list, dim=0)  # 1=real, 0=padded
    batch_channel_mask = torch.stack(channel_mask_list, dim=0) if channel_mask_list else None

    return batch_data, batch_window_mask, batch_channel_mask, metadata_list


class PredictionDataModule(L.LightningDataModule):
    """Lightning DataModule for prediction on single MEG files.

    Note: At inference time, channel selection is handled automatically by PredictDataset
    based on the available channels in the MEG file. No reference channels are needed.
    """

    def __init__(
        self,
        file_path: str,
        dataset_config: Dict[str, Any],
        dataloader_config: Dict[str, Any],
        reference_channels_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize prediction data module.

        Args:
            file_path: Path to the MEG file (.fif or .ds)
            dataset_config: Configuration for data processing
            dataloader_config: Configuration for data loaders
            **kwargs: Additional parameters for compatibility (unused)
        """
        super().__init__()
        self.file_path = file_path
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.reference_channels_path = reference_channels_path

        self.predict_dataset: Optional[PredictDataset] = None
        self.input_shape: Optional[torch.Size] = None
        self.output_shape: Optional[torch.Size] = None
        
    def prepare_data(self):
        """Prepare data - verify file exists."""
        import os
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"MEG file not found: {self.file_path}")
            
        # Verify it's a supported file type
        if not (self.file_path.endswith('.fif') or self.file_path.endswith('.ds') or 
                self.file_path.endswith('.meg')):
            logger.warning(f"File type might not be supported: {self.file_path}")
            
    def setup(self, stage: Optional[str] = None):
        """Set up the prediction dataset."""
        if stage == 'predict' or stage is None:
            self.predict_dataset = PredictDataset(
                file_path=self.file_path,
                dataset_config=self.dataset_config,
                reference_channels_path=self.reference_channels_path,
            )
           
            # Set shapes
            if len(self.predict_dataset) > 0:
                sample = self.predict_dataset[0]
                data = sample[0]  # chunk data
                self.input_shape = data.shape
                self.output_shape = torch.Size([data.shape[0]])  # n_windows
                    
                logger.info(f"Prediction dataset setup: {len(self.predict_dataset)} samples")
                logger.info(f"Input shape: {self.input_shape}")

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Create the prediction dataloader."""
        if self.predict_dataset is None:
            raise RuntimeError("Call setup() before getting prediction dataloader")
            
        predict_config = self.dataloader_config.get('predict', self.dataloader_config.get('test', {}))
        # Check that shuffle is False for prediction
        if predict_config.get('shuffle', True):
            logger.warning("Shuffle should be False for prediction dataloader, setting to False")
            predict_config['shuffle'] = False

        # Use module-level predict_collate_fn
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            **predict_config,
            collate_fn=predict_collate_fn,
        )
    
    def get_input_shape(self) -> torch.Size:
        """Get the input shape for model initialization."""
        if self.input_shape is None:
            raise RuntimeError("Call setup() before getting input shape")
        return self.input_shape
    
    def get_output_shape(self) -> torch.Size:
        """Get the output shape for model initialization."""
        if self.output_shape is None:
            raise RuntimeError("Call setup() before getting output shape")
        return self.output_shape
