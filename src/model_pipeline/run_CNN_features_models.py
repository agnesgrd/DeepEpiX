"""
Author: Agnès GUINARD
Created: 2025-09-30
Description: Testing Pipeline for model_CNN.keras & model_features_only.keras (trained by Pauline Mouchès)
"""

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import model_pipeline.params as params
from model_pipeline.features_utils import get_win_data_feat
from model_pipeline.sliding_windows_utils import (
    save_data_matrices,
    create_windows,
    generate_database,
    get_win_data_signal,
)
from model_pipeline.utils import (
    load_obj,
    compute_gfp,
    find_peak_gfp,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# === Helper functions specific to models ===


def prepare_data(subject, output_path, channel_groups):
    """Prepare data matrices, create windows, and generate test IDs."""
    save_data_matrices(subject, output_path, channel_groups)
    window_size = params.window_size_ms
    total_nb_windows = create_windows(output_path, window_size)
    return generate_database(total_nb_windows)


def load_model(model_name):
    """Load and compile a Keras model."""
    model = keras.models.load_model(model_name, compile=False)
    model.compile()
    return model


def predict_windows(model, X_test_ids, model_name, output_path):
    """Predict probabilities for all windows using the given model."""
    file_path = os.path.join(output_path, "data_raw_windows_bi")
    with open(file_path, "rb") as f:
        y_pred_probas = []

        device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        with tf.device(device):
            for cur_win in X_test_ids:
                sample = get_win_data_signal(f, cur_win, params.dim)

                if "features" in model_name:
                    sample = get_win_data_feat(sample)

                y_pred_probas.append(model(sample).numpy()[0][0])
                del sample

    return y_pred_probas


def get_adjusted_onsets(X_test_ids, output_path):
    """Adjust onset times based on GFP peaks."""
    y_timing_data = load_obj("data_raw_timing.pkl", output_path)
    onsets = []

    for win in X_test_ids:
        cur_win = X_test_ids[win]
        window = get_win_data_signal(
            open(
                os.path.join(
                    output_path,
                    "data_raw_windows_bi",
                ),
                "rb",
            ),
            cur_win,
            params.dim,
        ).squeeze()

        gfp = compute_gfp(window.T)
        times = np.linspace(0, window.shape[0] / params.sfreq, window.shape[0])
        peak_time = find_peak_gfp(gfp, times)

        onset = ((y_timing_data[win] - window.shape[0] / 2) / params.sfreq) + peak_time
        onsets.append(round(onset, 3))

    return onsets


def get_onsets(output_path):
    """Get raw timing data."""
    y_timing_data = load_obj("data_raw_timing.pkl", output_path)
    onsets = (y_timing_data / params.sfreq).round(3).tolist()
    return onsets


def save_predictions(output_path, model_name, onsets, y_pred_probas):
    """Save predictions into a CSV file compatible with MNE annotations."""
    df = pd.DataFrame(
        {
            "onset": onsets,
            "duration": 0,
            "probas": y_pred_probas,
        }
    )
    output_file = os.path.join(
        output_path, f"{os.path.basename(model_name)}_predictions.csv"
    )
    df.to_csv(output_file, index=False)
    return output_file


# === Main function ===
def test_model(
    model_name,
    model_type,
    subject,
    output_path,
    threshold=0.5,
    adjust_onset=True,
    channel_groups=None,
):
    """Run the full pipeline: prepare data, predict, adjust onsets, and save results."""
    # 1. Data preparation
    X_test_ids = prepare_data(subject, output_path, channel_groups)

    # 2. Load model
    model = load_model(model_name)

    # 3. Predictions
    y_pred_probas = predict_windows(model, X_test_ids, model_name, output_path)

    # 4. Cleanup model & GPU memory
    del model
    gc.collect()
    keras.backend.clear_session()

    # 5. Adjust onset times
    if adjust_onset:
        onsets = get_adjusted_onsets(X_test_ids, output_path)
    else:
        onsets = get_onsets(output_path)

    # 6. Save predictions
    return save_predictions(output_path, model_name, onsets, y_pred_probas)
