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
	
	return sample

def refine_onset_with_gfp(window, onset, fs):
	gfp = compute_gfp(window.T)  # Compute GFP
	times = np.linspace(0, window.shape[0] / fs, window.shape[0])  # Time vector
	
	peak_time = find_peak_gfp(gfp, times)  # Find max GFP time
	adjusted_onset = onset - window.shape[0]/2/ fs + peak_time  # Align event to GFP peak
	return adjusted_onset

def detect_events_by_duration_with_gap_filling(std_per_time, mean_per_time, fs, heartbeat_quantile=0.95, spike_quantile=0.75, bad_segment_quantile=0.5, 
											  heartbeat_min_duration=0.015, spike_min_duration=0.1, bad_segment_min_duration=0.5,
											  heartbeat_max_duration=0.5, spike_max_duration=0.4, bad_segment_max_duration=2.0,
											  heartbeat_max_gap_duration=0.01, spike_max_gap_duration=0.02, bad_segment_max_gap_duration=0.1,
											  heartbeat_gap=0.4, spike_gap=0.06, conflict_gap_duration=0.1, bad_segment_exclusion_win=1):
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
	
	# Identify time points where std exceeds the threshold
	heartbeat_thresh = np.quantile(mean_per_time, heartbeat_quantile)
	spike_thresh = np.quantile(std_per_time, spike_quantile)
	bad_segment_thresh = np.quantile(std_per_time, bad_segment_quantile)

	# Identify time points where std exceeds the quantile threshold
	heartbeat_mask = mean_per_time > heartbeat_thresh
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
						if event_onsets:
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
	heartbeats_onset, spikes_onset, bad_segments_onset = resolve_event_conflicts(heartbeats_onset, spikes_onset, bad_segments_onset, fs, heartbeat_gap, spike_gap, bad_segment_exclusion_win, conflict_gap_duration)
	
	return heartbeats_onset, spikes_onset, bad_segments_onset

def resolve_event_conflicts(heartbeats, spikes, bad_segments, fs,
                            heartbeat_gap=0.4, spike_gap=0.06, 
                            bad_segment_exclusion_win=1, conflit_gap_duration=0.1):
    """
    Resolve conflicts among detected events based on priority and time constraints.

    Rules:
    - Heartbeats must be at least `heartbeat_gap` seconds apart.
    - Spikes must be at least `spike_gap` seconds apart.
    - Heartbeats and spikes should not be detected within `bad_segment_exclusion_win` seconds around bad segments.
    - Spikes should be excluded if a heartbeat is detected within `conflit_gap_samples` samples.
    - Priority: Bad segment > Spike > Heartbeat.

    Parameters:
    - heartbeats (list): List of heartbeat onset times (seconds)
    - spikes (list): List of spike onset times (seconds)
    - bad_segments (list): List of bad segment onset times (seconds)
    - fs (int): Sampling frequency (Hz)
    - conflit_gap_samples (int): Exclusion window in samples for spikes near heartbeats.

    Returns:
    - final_heartbeats (list): Filtered heartbeat events
    - final_spikes (list): Filtered spike events
    - final_bad_segments (list): Filtered bad segment events
    """

    # Convert time constraints to seconds
    heartbeat_min_gap = heartbeat_gap * fs  
    spike_min_gap = spike_gap * fs  
    bad_segment_exclusion_window = bad_segment_exclusion_win * fs  
    conflit_gap = conflit_gap_duration / fs  # Convert samples to seconds

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
                if ((current_event - last_spike) * fs >= spike_min_gap and
                    (current_event - last_heartbeat) * fs >= conflit_gap):  # Check conflict gap
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
		
		for feat, func in compute_features.items():
			feature = func(window)

		windows.append(window)
		features.append(feature)

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
	# print(confusion_matrix_df)
	performance_metrics_df = pd.DataFrame(perf_metrics_data)
	# print(performance_metrics_df)

	return true_positive, false_positive, false_negative, precision, recall, f1

def append_to_csv(read_csv_path, new_data):
    columns = [
        "heartbeat_quantile", "spike_quantile", "bad_segment_quantile", 
		"heartbeat_min_duration", "spike_min_duration", "bad_segment_min_duration", 
		"heartbeat_max_duration", "spike_max_duration", "bad_segment_max_duration",
		"heartbeat_max_gap_duration", "spike_max_gap_duration", "bad_segment_max_gap_duration",
		"conflict_gap_duration",
		"heartbeat_gap", "spike_gap", "bad_segment_exclusion_win",
		"heartbeat_tolerance", "spike_tolerance",
		"ds",
		"description",
        "TP", "FN", "FP", 
		"precision", "recall", "f1-score"
    ]
    
    # Vérifier si le fichier existe
    if not os.path.exists(read_csv_path):
        # Créer un DataFrame vide avec les bonnes colonnes
        df = pd.DataFrame(columns=columns)
        df.to_csv(read_csv_path, index=False)
    
    # Lire le fichier existant
    df = pd.read_csv(read_csv_path)
    
    # Ajouter la nouvelle ligne
    new_row = pd.DataFrame([new_data], columns=columns)
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Sauvegarder dans le fichier
    df.to_csv(read_csv_path, index=False)

def test_model_dash(model_name, X_test_ids, output_path, threshold, adjust_onset, subject):

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

	# # Use summary to print model details on the selected device
	# print(summary(model, (1, 274, 60)))

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
	mean_per_time = np.mean(full_mse, axis=1)  # This gives an array of shape (time,)

	# # Now, duplicate the standard deviation across the channels to match the shape (time, channels)
	# full_mse = np.tile(std_per_time[:, np.newaxis], (1, full_mse.shape[1]))  # Shape (time, channels)

	grad_path = f"{output_path}/{os.path.basename(model_name)}_anomDetect.pkl"
	with open(grad_path, 'wb') as f:
		pickle.dump(full_mse, f)

	param_values = {
		"hq": 0.95,  # Heartbeat quantile
		"sq": 0.75,  # Spike quantile
		"bsq": 0.5,  # Bad segment quantile

		"hmd": 0.015,  # Min duration heartbeats
		"smd": 0.1,  # Min duration spikes
		"bsmd": 0.5,  # Min duration bad segments

		"hmxd": 0.5,  # Max duration heartbeats
		"smxd": 0.4,  # Max duration spikes
		"bsmxd": 2,  # Max duration bad segments

		"hmgd": 0.01,  # Heartbeat gap
		"smgd": 0.02,  # Spike gap
		"bsmgd": 0.1,  # Bad segment gap

		"cgd": 0.1,  # Conflict gap for spikes near heartbeats

		"hg": 0.4,  # Heartbeat exclusion if too close
		"sg": 0.04,  # Spike exclusion if too close
		"bsew": 1,  # Bad segment exclusion window
	}

	hq, sq, bsq, hmd, smd, bsmd, hmxd, smxd, bsmxd, hmgd, smgd, bsmgd, cgd, hg, sg, bsew = 	param_values["hq"], param_values["sq"], param_values["bsq"], param_values["hmd"], param_values["smd"], param_values["bsmd"], param_values["hmxd"], param_values["smxd"], param_values["bsmxd"], param_values["hmgd"], param_values["smgd"], param_values["bsmgd"], param_values["hg"], param_values["sg"], param_values["cgd"], param_values["bsew"] # 

	# Call the function to detect events
	heartbeats_onset, spikes_onset, bad_segments_onset = detect_events_by_duration_with_gap_filling(
		std_per_time, mean_per_time, params.sfreq_ae,
		hq, sq, bsq, hmd, smd, bsmd, hmxd, smxd, bsmxd, hmgd, smgd, bsmgd, hg, sg, cgd, bsew
		)

	# Create the DataFrame with descriptions
	df_pred = create_event_dataframe_with_description(heartbeats_onset, spikes_onset, bad_segments_onset)
	
	# Save DataFrame as CSV
	df_pred.to_csv(f'{output_path}/{os.path.basename(model_name)}_predictions.csv', index=False)

	# PERFORMANCE
	# Load ground truth
	output_csv_path = "/home/admin_mel/Code/DeepEpiX/data/testData/patient_annotations.csv"
	result_csv_path = "/home/admin_mel/Code/DeepEpiX/data/testData/DeepEpiX_results.csv"

	df_gt = pd.read_csv(output_csv_path)
	model_descriptions = [["heartbeat"], ["spike"]]
	target_descriptions = [["ECG Event"], ["jj_add", "JJ_add", "jj_valid", "JJ_valid"]]
	tolerance_by_event = [0.1, 0.4]

	for i in range(2):

		# Select only spike events in both DataFrames
		df_gt_spike = df_gt.loc[
			(df_gt["description"].isin(target_descriptions[i])) & (df_gt["ds_id"] == str(os.path.basename(subject))),
			"onset"
		]

		df_pred_spike = df_pred.loc[(df_pred["description"].isin(model_descriptions[i])), "onset"]

		tp, fp, fn, p, r, f1 = compute_performance(df_pred_spike, df_gt_spike, tolerance= tolerance_by_event[i])
	
		new_result = {"heartbeat_quantile" : hq, "spike_quantile" : sq, "bad_segment_quantile": bsq, 
			"heartbeat_min_duration" : hmd, "spike_min_duration": smd, "bad_segment_min_duration": bsmd, 
			"heartbeat_max_duration": hmxd, "spike_max_duration": smxd, "bad_segment_max_duration": bsmxd,
			"heartbeat_max_gap_duration": hmgd, "spike_max_gap_duration": smgd, "bad_segment_max_gap_duration" : bsmgd,
			"conflict_gap_duration": cgd,
			"heartbeat_gap": hg, "spike_gap": sg, "bad_segment_exclusion_win": bsew,
			"heartbeat_tolerance": tolerance_by_event[0], "spike_tolerance": tolerance_by_event[1],
			"ds": str(os.path.basename(subject)),
			"description": model_descriptions[i],
			"TP" : tp, "FP": fp, "FN": fn, 
			"precision": p, "recall": r, "f1-score" : f1}
		
		append_to_csv(result_csv_path, new_result)