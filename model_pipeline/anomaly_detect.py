import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import os.path as op
import params
import pandas as pd
import csv
from utils import save_obj, load_obj, standardize
from model_pipeline.utils import compute_gfp, find_peak_gfp
from torchsummary import summary
import pickle
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import utils

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2),
			nn.MaxPool2d(kernel_size=(1, 2)),
			nn.Conv2d(32, 64, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			nn.MaxPool2d(kernel_size=(1, 2)),
			nn.Conv2d(64, 128, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			nn.Conv2d(128, 256, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.Conv2d(256, 1, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(1),
		)

	def forward(self, x):
		# x = nn.functional.pad(x, (2, 2))  # Zero padding only on the right side for PyTorch
		encoded = self.encoder(x)
		return encoded


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()

		self.decoder = nn.Sequential(
			nn.Conv2d(1, 256, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.Upsample(scale_factor=(1, 2), mode="nearest"),
			nn.Conv2d(256, 128, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			nn.Upsample(scale_factor=(1, 2), mode="nearest"),
			nn.Conv2d(128, 64, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 32, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2),
			nn.Conv2d(32, 1, kernel_size=(1, 5), padding="same"),
			nn.BatchNorm2d(1),
		)

	def forward(self, x):
		decoded = self.decoder(x)
		# decoded = torch.narrow(decoded, -1, 1, 60)
		return decoded


class AE(nn.Module):
	def __init__(self, first_direction, sfreq, window_size):
		super(AE, self).__init__()

		if first_direction == "channel":
			input_shape = (1, 274, int(sfreq * window_size))
		elif first_direction == "time":
			input_shape = (1, int(sfreq * window_size), 274)

		self.encoder = Encoder()
		self.decoder = Decoder()  # Get output channels of the last conv layer

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

	
def resume(model, filename, device):
	checkpoint = torch.load(filename, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])

# def postprocess(mse_win):
#     thresh = np.quantile(mse_win,0.75)
#     mse_win[mse_win<thresh] = np.min(mse_win[mse_win>thresh])/2
#     mse_win[mse_win>thresh] = 0.25* ((mse_win[mse_win>thresh] -  np.min(mse_win[mse_win>thresh]))/(np.max(mse_win[mse_win>thresh])-np.min(mse_win[mse_win>thresh]))) + 0.75
#     mse_win = signal.wiener(mse_win, (7,7))
#     return mse_win

def postprocess(mse_win):
	# Compute threshold at the 75th percentile
	thresh = np.quantile(mse_win, 0.75)

	# Instead of setting low values to an arbitrary min/2, use log-scaling for better distribution
	mse_win = np.log1p(mse_win)  # log(1 + x) avoids issues with zero values

	# Normalize high MSE values using robust scaling (less sensitive to outliers)
	high_mse = mse_win[mse_win > thresh]
	if high_mse.size > 0:
		mse_win[mse_win > thresh] = 0.25 * ((high_mse - np.median(high_mse)) / (np.percentile(high_mse, 90) - np.median(high_mse))) + 0.75

	# # Apply Wiener filter with a smaller window (reduces blurring of sharp spikes)
	mse_win = signal.wiener(mse_win, (7,7))

	# # Apply Gaussian filter for smoothing
	# mse_win = gaussian_filter(mse_win, sigma=1)

	return mse_win

def get_win_data_signal(f,win,sub,dim):

	# Store sample 
	f.seek(dim[0]*dim[1]*win*4) #4 because its float32 and dtype.itemsize = 4
	sample = np.fromfile(f, dtype='float32', count=dim[0]*dim[1])
	sample = sample.reshape(dim[1],dim[0])

	# sample = (sample - np.mean(sample, axis=1, keepdims=True)) / np.std(sample, axis=1, keepdims=True)

	sample = np.expand_dims(sample,axis=0)
	sample = np.expand_dims(sample,axis=0)

	# mean = np.mean(sample)
	# std = np.std(sample)
	# sample = (sample - mean)/std
	
	return sample

# def detect_events(mse_win, signal, fs, duration_ranges={'heartbeat': (0.02, 0.05), 
#                                                          'spike': (0.08, 0.12), 
#                                                          'bad_segment': (0.15, np.inf)}):
#     """
#     Detects heartbeats, spikes, and bad segments based on preprocessed MSE patterns, 
#     then refines onsets using GFP computed on the original signal.

#     Parameters:
#     - mse_win: np.array (time x channels) - Preprocessed MSE values
#     - signal: np.array (time x channels) - Original input signal
#     - fs: Sampling frequency in Hz
#     - duration_ranges: Dict with min/max duration (in seconds) for each event type
	
#     Returns:
#     - Dictionary with onsets (in seconds) for heartbeats, spikes, and bad segments.
#     """
	
#     # Thresholding: Identify high-MSE regions
#     thresh = np.quantile(mse_win, 0.75)  # Use 75th percentile as threshold
#     binary_mask = mse_win > thresh
#     event_mask = np.any(binary_mask, axis=1)  # Collapse across channels

#     # Step 1: Detect Events Based on Duration
#     event_onsets = {'heartbeat': [], 'spike': [], 'bad_segment': []}
	
#     for event_type, (min_dur, max_dur) in duration_ranges.items():
#         min_samples = int(min_dur * fs)
#         max_samples = int(max_dur * fs) if np.isfinite(max_dur) else None  # Handle np.inf
		
#         # Find event boundaries
#         diff_mask = np.diff(np.concatenate(([0], event_mask.astype(int), [0])))
#         event_starts = np.where(diff_mask == 1)[0]
#         event_ends = np.where(diff_mask == -1)[0]
		
#         for start, end in zip(event_starts, event_ends):
#             event_duration = end - start
			
#             # Apply duration filtering (ignore max_samples if it's None)
#             if min_samples <= event_duration and (max_samples is None or event_duration <= max_samples):
#                 # Step 2: Compute GFP on the original signal segment
#                 signal_segment = signal[start:end, :]
#                 gfp_segment = np.std(signal_segment, axis=1)  # GFP over time
				
#                 # Find peak GFP to adjust onset
#                 peak_idx = np.argmax(gfp_segment)  # Find highest variance point
#                 onset_time = (start + peak_idx) / fs  # Convert to seconds
#                 event_onsets[event_type].append(onset_time)
	
#     return event_onsets

def detect_events_by_duration(mse_win, fs, heartbeat_dur=(0.02, 0.05), spike_dur=(0.05, 0.15), bad_seg_dur=(0.3, 1), mse_thresh=0.75):
	"""Detect heartbeats, spikes, and bad segments based on their duration and MSE threshold."""
	event_onsets = []
	
	# Loop through each channel and detect events
	for ch_idx in range(mse_win.shape[1]):
		mse_channel = mse_win[:, ch_idx]
		
		# Apply thresholding to MSE (adjustable thresholding)
		thresh = np.quantile(mse_channel, mse_thresh)  # MSE threshold at 75th percentile
		mse_channel[mse_channel < thresh] = 0  # Set values below threshold to 0
		mse_channel[mse_channel >= thresh] = 1  # Mark regions above the threshold as 1 (high MSE)
		
		# Apply Gaussian filter for smoothing
		mse_channel = gaussian_filter(mse_channel, sigma=1)
		
		# Detect event onsets based on thresholded and smoothed MSE signal
		heartbeat_onsets = find_event_onsets_by_duration(mse_channel, fs, heartbeat_dur)
		spike_onsets = find_event_onsets_by_duration(mse_channel, fs, spike_dur)
		bad_seg_onsets = find_event_onsets_by_duration(mse_channel, fs, bad_seg_dur)

		# Add the detected onsets to the global list
		event_onsets.extend(heartbeat_onsets)
		event_onsets.extend(spike_onsets)
		event_onsets.extend(bad_seg_onsets)
	
	return event_onsets

def find_event_onsets_by_duration(mse_channel, fs, event_dur):
	"""Find event onsets based on MSE with duration filtering."""
	min_samples = int(event_dur[0] * fs)
	max_samples = int(event_dur[1] * fs)
	event_onsets = []
	event_start = None

	for i in range(len(mse_channel)):
		if mse_channel[i] > 0:  # High MSE indicating a potential event
			if event_start is None:
				event_start = i  # Start of a potential event
		else:
			if event_start is not None:
				event_end = i  # End of a potential event
				event_duration = (event_end - event_start) / fs  # Duration in seconds
				if min_samples <= (event_end - event_start) <= max_samples:
					event_onsets.append((event_start / fs, event_end / fs))  # Convert to seconds
				event_start = None
	
	return event_onsets

def refine_onset_with_gfp(window, onset, fs):
	gfp = compute_gfp(window.T)  # Compute GFP
	times = np.linspace(0, window.shape[0] / fs, window.shape[0])  # Time vector
	
	peak_time = find_peak_gfp(gfp, times)  # Find max GFP time
	adjusted_onset = onset - window.shape[0]/2/ fs + peak_time  # Align event to GFP peak
	return adjusted_onset

def detect_events_by_duration_with_gap_filling(std_per_time, fs, heartbeat_quantile=0.95, spike_quantile=0.75, bad_segment_quantile=0.5, 
											  heartbeat_min_duration=0.015, spike_min_duration=0.1, bad_segment_min_duration=0.5,
											  heartbeat_max_duration=0.5, spike_max_duration=0.4, bad_segment_max_duration=2.0,
											  heartbeat_max_gap_duration=0.01, spike_max_gap_duration=0.02, bad_segment_max_gap_duration=0.1,
											  conflict_gap_duration=0.5):
	"""
	Detect heartbeats, spikes, and bad segments based on standard deviation per time and their durations,
	filling any gaps in events based on a maximum gap duration, with maximum event durations.

	Parameters:
	- std_per_time (np.array): 1D array of standard deviations per time point (shape: (time,))
	- fs (int): Sampling frequency (Hz)
	- heartbeat_thresh (float): Threshold for detecting heartbeats based on std
	- spike_thresh (float): Threshold for detecting spikes based on std
	- bad_segment_thresh (float): Threshold for detecting bad segments based on std
	- heartbeat_min_duration (float): Minimum duration (in seconds) for heartbeats
	- spike_min_duration (float): Minimum duration (in seconds) for spikes
	- bad_segment_min_duration (float): Minimum duration (in seconds) for bad segments
	- heartbeat_max_duration (float): Maximum duration (in seconds) for heartbeats
	- spike_max_duration (float): Maximum duration (in seconds) for spikes
	- bad_segment_max_duration (float): Maximum duration (in seconds) for bad segments
	- max_gap_duration (float): Maximum gap duration (in seconds) allowed to merge two events

	Returns:
	- heartbeats_onset (list): List of onset times (in seconds) of detected heartbeats
	- spikes_onset (list): List of onset times (in seconds) of detected spikes
	- bad_segments_onset (list): List of onset times (in seconds) of detected bad segments
	"""
	
	# Convert minimum and maximum durations and max gap duration from seconds to samples
	heartbeat_min_samples = int(heartbeat_min_duration * fs)
	spike_min_samples = int(spike_min_duration * fs)
	bad_segment_min_samples = int(bad_segment_min_duration * fs)
	heartbeat_max_samples = int(heartbeat_max_duration * fs)
	spike_max_samples = int(spike_max_duration * fs)
	bad_segment_max_samples = int(bad_segment_max_duration * fs)
	heartbeat_max_gap_samples = int(heartbeat_max_gap_duration * fs)
	spike_max_gap_samples = int(spike_max_gap_duration * fs)
	bad_segment_max_gap_samples = int(bad_segment_max_gap_duration * fs)
	conflict_gap_samples = int(conflict_gap_duration * fs)
	
	# Identify time points where std exceeds the threshold
	heartbeat_thresh = np.quantile(std_per_time, heartbeat_quantile)
	spike_thresh = np.quantile(std_per_time, spike_quantile)
	bad_segment_thresh = np.quantile(std_per_time, bad_segment_quantile)

	# Identify time points where std exceeds the quantile threshold
	heartbeat_mask = std_per_time > heartbeat_thresh
	spike_mask = std_per_time > spike_thresh
	bad_segment_mask = std_per_time > bad_segment_thresh
	
	# Function to detect events based on a mask and a minimum duration, with gap filling
	def detect_event_onsets_with_gap_filling(mask, min_samples, max_samples, max_gap_samples):
		event_onsets = []
		event_start = 0
		event_end = 0
		last_event_end = 0
		
		for i in range(1, len(mask)):
			if mask[i] and not mask[i-1]:  # Event starts
				event_start = i
			elif not mask[i] and mask[i-1]:  # Event ends
				event_end = i
				event_duration = event_end - event_start
				if event_duration >= min_samples and event_duration <= max_samples:  # Valid event duration
					if last_event_end is not None and event_start - last_event_end <= max_gap_samples:
						# Merge with the last event if gap is small enough
						event_onsets[-1] = event_start / fs  # Update previous event start time
					else:
						event_onsets.append(event_start / fs)
					last_event_end = event_end
		# If the event ends at the last sample
		if event_end is not None and event_end - event_start >= min_samples and event_end - event_start <= max_samples:
			event_onsets.append(event_start / fs)
		
		return event_onsets

	# Detect onsets for heartbeats, spikes, and bad segments
	heartbeats_onset = detect_event_onsets_with_gap_filling(heartbeat_mask, heartbeat_min_samples, heartbeat_max_samples, heartbeat_max_gap_samples)
	spikes_onset = detect_event_onsets_with_gap_filling(spike_mask, spike_min_samples, spike_max_samples, spike_max_gap_samples)
	bad_segments_onset = detect_event_onsets_with_gap_filling(bad_segment_mask, bad_segment_min_samples, bad_segment_max_samples, bad_segment_max_gap_samples)
	
	# Resolve conflicts between detected events
	heartbeats_onset, spikes_onset, bad_segments_onset = resolve_event_conflicts(heartbeats_onset, spikes_onset, bad_segments_onset, fs)
	
	return heartbeats_onset, spikes_onset, bad_segments_onset

def resolve_event_conflicts(heartbeats, spikes, bad_segments, fs, heartbeat_gap=0.4, spike_gap=0.06, bad_segment_exclusion_win=1):
	"""
	Resolve conflicts among detected events based on priority and time constraints.
	
	Rules:
	- Heartbeats must be at least `heartbeat_gap` seconds apart.
	- Spikes must be at least `spike_gap` seconds apart.
	- Heartbeats and spikes should not be detected within `bad_segment_exclusion_win` seconds around bad segments.
	- Priority: Bad segment > Spike > Heartbeat.

	Parameters:
	- heartbeats (list): List of heartbeat onset times (seconds)
	- spikes (list): List of spike onset times (seconds)
	- bad_segments (list): List of bad segment onset times (seconds)
	- fs (int): Sampling frequency (Hz)

	Returns:
	- final_heartbeats (list): Filtered heartbeat events
	- final_spikes (list): Filtered spike events
	- final_bad_segments (list): Filtered bad segment events
	"""
	
	# Convert time constraints to samples
	heartbeat_min_gap = heartbeat_gap * fs  
	spike_min_gap = spike_gap * fs  
	bad_segment_exclusion_window = bad_segment_exclusion_win * fs  

	# Sort all events by time with their types
	all_events = []
	for t in bad_segments:
		all_events.append((t, 0))  # 0 = bad segment
	for t in spikes:
		all_events.append((t, 1))  # 1 = spike
	for t in heartbeats:
		all_events.append((t, 2))  # 2 = heartbeat

	all_events.sort()  # Sort events by time

	# Final lists
	final_bad_segments = []
	final_spikes = []
	final_heartbeats = []

	last_heartbeat = -float("inf")  # Track last valid heartbeat time
	last_spike = -float("inf")  # Track last valid spike time

	i = 0
	while i < len(all_events):
		current_event, current_type = all_events[i]
		has_conflict = False

		# Check if it's within bad segment exclusion window
		if current_type in [1, 2]:  # Spike or heartbeat
			for bad_segment in final_bad_segments:
				if abs(current_event - bad_segment) * fs < bad_segment_exclusion_window:
					has_conflict = True
					break  # Skip this event if it's too close to a bad segment

		if not has_conflict:
			if current_type == 0:  # Bad segment
				final_bad_segments.append(current_event)

			elif current_type == 1:  # Spike
				if (current_event - last_spike) * fs >= spike_min_gap:
					final_spikes.append(current_event)
					last_spike = current_event

			else:  # Heartbeat
				if (current_event - last_heartbeat) * fs >= heartbeat_min_gap:
					final_heartbeats.append(current_event)
					last_heartbeat = current_event

		# Move to the next event
		i += 1

	return final_heartbeats, final_spikes, final_bad_segments


def create_event_dataframe_with_description(heartbeats_onset, spikes_onset, bad_segments_onset):
	"""
	Create a DataFrame with onsets, durations, probability scores, and event descriptions.
	
	Parameters:
	- heartbeats_onset (list): List of onset times (in seconds) of detected heartbeats
	- spikes_onset (list): List of onset times (in seconds) of detected spikes
	- bad_segments_onset (list): List of onset times (in seconds) of detected bad segments
	- fs (int): Sampling frequency (Hz)
	- y_pred_probas (np.array): Raw predicted probabilities corresponding to the events
	- max_gap_duration (float): Maximum gap duration (in seconds) to merge events
	
	Returns:
	- df (pd.DataFrame): DataFrame with onset, duration, probability scores, and description
	"""
	
	# Combine all onsets and label each onset with its event type
	all_onsets = []
	event_types = []  # List to store the event type for each onset
	
	# Add heartbeats
	all_onsets.extend(heartbeats_onset)
	event_types.extend(['heartbeat'] * len(heartbeats_onset))
	
	# Add spikes
	all_onsets.extend(spikes_onset)
	event_types.extend(['spike'] * len(spikes_onset))
	
	# Add bad segments
	all_onsets.extend(bad_segments_onset)
	event_types.extend(['bad segment'] * len(bad_segments_onset))
	
	# Sort onsets and corresponding event types
	sorted_indices = np.argsort(all_onsets)
	all_onsets = np.array(all_onsets)[sorted_indices]
	event_types = np.array(event_types)[sorted_indices]
		
	event_durations = []
	
	# For each event onset, determine the duration and probability score
	for onset, event_type in zip(all_onsets, event_types):
				
		# For simplicity, let's assume a fixed duration for each event type (e.g., heartbeats = 0.02s, spikes = 0.1s, bad segments = 0.5s)
		# You can modify this based on your event type (e.g., use a dynamic duration depending on the event type)
		if event_type == 'heartbeat':
			event_duration = 0  # Duration in seconds for heartbeat
		elif event_type == 'spike':
			event_duration = 0  # Duration in seconds for spike
		else:
			event_duration = 0.5  # Duration in seconds for bad segment
		
		# Append to lists
		event_durations.append(event_duration)
	
	# Create the DataFrame
	df = pd.DataFrame({
		"onset": all_onsets,
		"duration": event_durations,  # Use calculated durations
		"description": event_types  # Event type as description
	})
	
	return df

def extract_features_windows(signal, event_onsets, fs, window_duration=0.1):
	"""
	Extracts 0.4s signal windows around event onsets.

	Parameters:
	- signal (numpy array): The full signal data.
	- event_onsets (list): List of event onset times (seconds).
	- fs (int): Sampling frequency.
	- window_duration (float): Window duration in seconds.

	Returns:
	- windows (numpy array): Extracted windows of shape (num_events, window_size).
	"""
	window_size = int(window_duration * fs)
	half_window = window_size // 2
	windows = []
	compute_features = {"ppa" : utils.compute_window_ppa, "std": utils.compute_window_std, "upslope" : utils.compute_window_upslope, "downslope": utils.compute_window_downslope, "average_slope": utils.compute_window_average_slope, "sharpness" : utils.compute_window_sharpness}
	features = []
					 
	for onset in event_onsets:
		
		center_idx = int(onset * fs)  # Convert time to sample index
		start_idx = max(0, center_idx - half_window)
		end_idx = min(len(signal), center_idx + half_window)

		# Extract window and pad if necessary
		window = signal[start_idx:end_idx]
		
		# new_onset = refine_onset_with_gfp(window, onset, fs)
		# print(onset)
		# print(new_onset)

		# center_idx = int(new_onset * fs)  # Convert time to sample index
		# start_idx = max(0, center_idx - half_window)
		# end_idx = min(len(signal), center_idx + half_window)

		
		for feat, func in compute_features.items():
			feature = func(window)

		# if len(window) < window_size:
		# 	window = np.pad(window, (0, window_size - len(window)), 'constant')

		windows.append(window)
		features.append(feature)
	
	print(window.shape)

	return np.array(features)

def compute_matches(model_onsets, gt_onsets, delta):
	true_positive = 0
	false_positive = 0
	false_negative = 0
	
	matched_gt = set()  # To track which ground truth values have been matched

	# Count true positives: each model prediction must match one unique ground truth
	for m in model_onsets:
		# Check if model prediction m matches any ground truth g within the delta
		matched = False
		for g in gt_onsets:
			if abs(m - g) <= delta and g not in matched_gt:
				true_positive += 1
				matched_gt.add(g)  # Mark this ground truth as matched
				matched = True
				break
		if not matched:
			false_positive += 1  # If no match, it's a false positive

	# Count false negatives: ground truth not matched by any model prediction
	for g in gt_onsets:
		if g not in matched_gt:
			false_negative += 1  # This ground truth has no matching model prediction

	return true_positive, false_positive, false_negative

def compute_performance(model_prediction, ground_truth, tolerance):

	delta = tolerance

	# model_onsets = pd.DataFrame(model_prediction)
	# gt_onsets = pd.DataFrame(ground_truth)

	true_positive, false_positive, false_negative = compute_matches(model_prediction, ground_truth, delta)

	# Compute Precision, Recall, F1-score
	precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
	recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
	f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

	# Confusion Matrix Data
	conf_matrix_data = {
		"Metric": ["True Positive", "False Positive", "False Negative", "True Negative"],
		"Count": [true_positive, false_positive, false_negative, "nan"]
	}
	conf_matrix_data = [
		{"": "Actual Negative", "Predicted Negative": "nan", "Predicted Positive": false_positive},
		{"": "Actual Positive", "Predicted Negative": false_negative, "Predicted Positive": true_positive}
	]

	# Performance Metrics Data
	perf_metrics_data = {
		"Metric": ["Precision", "Recall", "F1 Score"],
		"Value": [f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]
	}

	# Display results
	results = f"""
	**Performance Metrics:**
	- **Precision:** {precision:.3f}
	- **Recall:** {recall:.3f}
	- **F1 Score:** {f1:.3f}
	"""

	# Create Dash tables
	confusion_matrix_df = pd.DataFrame(conf_matrix_data)
	performance_metrics_df = pd.DataFrame(perf_metrics_data)

	return confusion_matrix_df, performance_metrics_df

def test_model_dash(model_name, X_test_ids, output_path, threshold, adjust_onset):

	f = open(f'{output_path}/data_raw_1_windows_bi')
	blocks_file = load_obj('data_raw_1_blocks.pkl', output_path)
	data_file = load_obj('data_raw_1.pkl', output_path)

	total_nb_windows = len(blocks_file)
	total_nb_points = data_file['meg'][0].shape[1]

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Initialize the model
	model = AE("time", sfreq = params.sfreq_ae, window_size = params.window_size_ms_ae * params.sfreq_ae)

	# Move model to the selected device first
	model.to(device)

	# Load the pre-trained model weights
	resume(model, model_name, device)

	# Use summary to print model details on the selected device
	print(summary(model, (1, 274, 60)))

	model.eval() 

	f = open(op.join(output_path, "data_raw_"+str(params.subject_number)+'_windows_bi'))

	MSE = nn.MSELoss(reduction = 'none')
	# -- instantiate arrays to store the full signal portion between start_win and stop_win and the corresponding gradient values 
	full_mse = np.zeros((total_nb_points, 274))
	full_signal = np.zeros((total_nb_points, 274))

	with torch.no_grad():
		for w in range(0,X_test_ids.shape[0]):

			cur_sub = X_test_ids[w,1]
			cur_win = X_test_ids[w,0]

			sample = get_win_data_signal(f,cur_win,cur_sub,params.dim_ae)

			mean = np.mean(sample)
			std = np.std(sample)
			input = (sample - mean)/std

			input = torch.tensor(input,dtype = torch.float32).to(device)
			output = model(input)

			input = input.squeeze()
			output = output.squeeze()
			mse = np.swapaxes(MSE(output, input).cpu().numpy(), 0, 1)
			
			full_signal[(w*(params.dim_ae[0]-9)):(w*(params.dim_ae[0]-9))+params.dim_ae[0],:] = np.swapaxes(sample[0][0], 0, 1)
			full_mse[(w*(params.dim_ae[0]-9)):(w*(params.dim_ae[0]-9))+params.dim_ae[0],:] = postprocess(mse)

			del sample
			del input
			del output

	del model

	std_per_time = np.std(full_mse, axis=1)  # This gives an array of shape (time,)
	full_mse = signal.wiener(full_mse, 3)
	# # Now, duplicate the standard deviation across the channels to match the shape (time, channels)
	full_mse = np.tile(std_per_time[:, np.newaxis], (1, full_mse.shape[1]))  # Shape (time, channels)

	grad_path = f"{output_path}/{os.path.basename(model_name)}_anomDetect.pkl"
	with open(grad_path, 'wb') as f:
		pickle.dump(full_mse, f)

	# Call the function to detect events
	heartbeats_onset, spikes_onset, bad_segments_onset = detect_events_by_duration_with_gap_filling(
		std_per_time, fs=params.sfreq_ae
	)

	# Create the DataFrame with descriptions
	df_pred = create_event_dataframe_with_description(heartbeats_onset, spikes_onset, bad_segments_onset)
	
	# Save DataFrame as CSV
	df_pred.to_csv(f'{output_path}/{os.path.basename(model_name)}_predictions.csv', index=False)


	# PERFORMANCE
	# Load ground truth
	output_csv_path = "/home/admin_mel/Code/DeepEpiX/data/testData/patient_annotations.csv"
	df_gt = pd.read_csv(output_csv_path)
	subject = 'Conti'
	ds = 'conti_Epi-001_20090709_07.ds'
	model_descriptions = [["spike"], ["heartbeat"]]
	target_descriptions = [["jj_add", "JJ_add", "jj_valid", "JJ_valid"], ["ECG Event"]]
	tolerance_by_event = [0.3, 0.1]

	for i in range(2):

		# Select only spike events in both DataFrames
		df_gt_spike = df_gt.loc[
			(df_gt["description"].isin(target_descriptions[i])) & (df_gt["ds_id"] == ds),
			"onset"
		]

		df_pred_spike = df_pred.loc[(df_pred["description"].isin(model_descriptions[i])), "onset"]

		cf_matrix, perf = compute_performance(df_pred_spike, df_gt_spike, tolerance= tolerance_by_event[i])

		print(model_descriptions[i], cf_matrix, perf)



	# event_onsets = heartbeats_onset + spikes_onset + bad_segments_onset
	# # Extract signal windows
	# features = extract_features_windows(full_signal, event_onsets, fs=params.sfreq_ae)
	# print(features.shape)

	# # Normalize and reduce dimensions

	# scaler = StandardScaler()
	# features = scaler.fit_transform(features)

	# # pca = PCA(n_components=10)  # Reduce to 5 main features
	# # event_windows_reduced = pca.fit_transform(event_windows_scaled)

	# # Apply K-Means clustering
	# num_clusters = 2
	# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
	# labels = kmeans.fit_predict(features)

	# # Create the DataFrame
	# df = pd.DataFrame({
	# 	"onset": event_onsets,
	# 	"duration": 0,  # Use calculated durations
	# 	"description": [f" cluster {l}" for l in labels]  # Event type as description
	# })

	# # Save DataFrame as CSV
	# df.to_csv(f'{output_path}/{os.path.basename(model_name)}_predictions.csv', index=False)

	# # Print cluster assignments
	# for i, (onset, cluster) in enumerate(zip(event_onsets, labels)):
	# 	print(f"Event at {onset}s assigned to cluster {cluster}")