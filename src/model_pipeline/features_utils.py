import numpy as np


def compute_window_ppa(window):
    return np.max(window, axis=0) - np.min(window, axis=0)


def compute_window_upslope(window):
    return np.max(np.diff(window, axis=0), axis=0)


def compute_window_std(window):
    return np.std(window, axis=0)


def compute_window_average_slope(window):
    abs_slopes = np.abs(np.diff(window, axis=0))
    return np.max((abs_slopes[:-1] + abs_slopes[1:]) / 2, axis=0)


def compute_window_downslope(window):
    return np.max(np.diff(window, axis=0), axis=0)


def compute_window_amplitude_ratio(window):
    ampl = np.max(window, axis=0) - np.min(window, axis=0)
    mean = np.mean(window, axis=0)
    mean[mean == 0] = 1
    return ampl / mean


def compute_window_sharpness(window):
    slopes = np.diff(window, axis=0)
    return np.max(np.abs(slopes[1:] - slopes[:-1]), axis=0)


def get_win_data_feat(
    sample_norm,
    compute_features={
        "ppa": compute_window_ppa,
        "std": compute_window_std,
        "upslope": compute_window_upslope,
        "downslope": compute_window_downslope,
        "average_slope": compute_window_average_slope,
        "sharpness": compute_window_sharpness,
    },
):

    sample_norm = np.squeeze(sample_norm)
    features_sample = np.empty((1, len(list(compute_features)), sample_norm.shape[1]))
    for feat, func in compute_features.items():
        features_sample[0, list(compute_features).index(feat)] = func(sample_norm)

    return features_sample
