from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import model_pipeline.params as params
from model_pipeline.utils import (
    read_raw,
    interpolate_missing_channels,
    fill_missing_channels,
    save_obj,
    load_obj,
    standardize,
)


def save_data_matrices(
    subject_path: str, output_dir: str, channel_groups: dict
) -> None:
    """
    Apply preprocessing, extract MEG data, and save it in a pickle file.

    Args:
        subject_path: Path to raw MEG data file (.ds, .fif, or directory).
        output_dir: Directory where processed data will be stored.
        channel_groups: Dict of channel groups (must include "bad" if applicable).
    """
    subject_path = Path(subject_path)
    output_dir = Path(output_dir)

    raw = read_raw(
        subject_path,
        preload=True,
        verbose=False,
        bad_channels=channel_groups.get("bad", []),
    )

    with open("good_channels_dict.pkl", "rb") as f:
        good_channels = pickle.load(f)

    # Filtering & resampling
    raw.filter(0.5, 50, n_jobs=8)
    raw.resample(params.sfreq).pick(["mag"])

    if subject_path.suffix == ".ds":
        raw = interpolate_missing_channels(raw, good_channels)
        data = {"meg": [raw.get_data()], "file": [str(subject_path)]}

    elif subject_path.suffix == ".fif" or subject_path.is_dir():
        # Flatten nested channel groups, excluding "bad"
        channels_order = [
            ch
            for group, chans in channel_groups.items()
            if group != "bad"
            for ch in chans
        ]
        raw.reorder_channels(channels_order)

        meg_data = fill_missing_channels(raw, len(good_channels))
        data = {"meg": [meg_data], "file": [str(subject_path)]}

    else:
        raise ValueError(f"Unsupported file type: {subject_path}")

    save_obj(data, "data_raw", output_dir)


def create_windows(output_dir: str, window_size_ms: int, stand: bool = False) -> int:
    """
    Crop windows from the pickle file and save them in a binary file.

    Args:
        output_dir: Directory where processed data is stored.
        window_size_ms: Window size in milliseconds.
        stand: If True, standardize the data.

    Returns:
        Total number of windows created.
    """
    output_dir = Path(output_dir)

    # Window size in samples (ms × sampling frequency)
    window_size = int(window_size_ms * params.sfreq)
    # Spacing between window centers (samples)
    window_spacing = int(
        (window_size_ms - 2 * params.spike_spacing_from_border_ms) * params.sfreq
    )

    # Load preprocessed data
    data = load_obj("data_raw.pkl", output_dir)

    all_windows = []
    window_centers_all = []
    block_indices_all = []

    for block_idx, block_data in enumerate(data["meg"]):
        # Compute window centers (in samples)
        window_centers = np.arange(window_size / 2, block_data.shape[1], window_spacing)

        block_windows = []
        for center in tqdm(window_centers, desc=f"Block {block_idx}"):
            if window_size / 2 <= center <= block_data.shape[1] - window_size / 2:
                low = int(center - window_size / 2)
                high = int(center + window_size / 2 + 0.1)  # Handle odd sizes
                block_windows.append(block_data[:, low:high])

                window_centers_all.append(center)
                block_indices_all.append(block_idx)

        if block_windows:
            all_windows.extend(block_windows)

    if not all_windows:
        raise RuntimeError("No valid windows were created. Check your parameters.")

    # Stack and convert to float32 for binary saving
    X_all = np.stack(all_windows).astype("float32")

    if stand:
        X_all = standardize(X_all)

    # Save binary MEG windows
    (output_dir / "data_raw_windows_bi").write_bytes(X_all.tobytes())

    # Save metadata
    save_obj(np.array(window_centers_all), "data_raw_timing", output_dir)
    save_obj(np.array(block_indices_all), "data_raw_blocks", output_dir)

    return len(X_all)


def generate_database(total_nb_windows: int) -> np.ndarray:
    """
    Generate a database of test window IDs.

    Args:
        total_nb_windows: Total number of windows.

    Returns:
        Array of shape (N, 1), for window index
    """
    X_test_ids = np.arange(total_nb_windows, dtype=int)
    print(X_test_ids.shape)
    return X_test_ids


def get_win_data_signal(f, win, dim):

    # Store sample
    f.seek(dim[0] * dim[1] * win * 4)  # 4 because its float32 and dtype.itemsize = 4
    sample = np.fromfile(f, dtype="float32", count=dim[0] * dim[1])
    sample = sample.reshape(dim[1], dim[0])
    sample = np.swapaxes(sample, 0, 1)
    sample = np.expand_dims(sample, axis=-1)
    sample = np.expand_dims(sample, axis=0)

    mean = np.mean(sample)
    std = np.std(sample)
    sample_norm = (sample - mean) / std

    return sample_norm
