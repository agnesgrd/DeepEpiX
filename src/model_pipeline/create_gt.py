import pandas as pd
import os
from pathlib import Path
import mne

output_path = "src/cache-directory"
gt_names = ['jj_valid', 'jj_add']
data_dir = Path("data/testData")

def read_raw(folder_path, preload, verbose, bad_channels=None):
	folder_path = Path(folder_path)
	print(folder_path)

	if folder_path.suffix == ".ds":
		raw = mne.io.read_raw_ctf(str(folder_path), preload=preload, verbose=verbose)

	elif folder_path.suffix == ".fif":
		raw = mne.io.read_raw_fif(str(folder_path), preload=preload, verbose=verbose)

	elif folder_path.is_dir():
		# Assume BTi/4D format: folder must contain 3 specific files
		files = list(folder_path.glob("*"))
		# Try to identify the correct files by names
		raw_fname = next((f for f in files if "rfDC" in f.name and f.suffix==""), None)
		config_fname = next((f for f in files if "config" in f.name.lower()), None)
		hs_fname = next((f for f in files if "hs" in f.name.lower()), None)

		if not all([raw_fname, config_fname, hs_fname]):
			raise ValueError("Could not identify raw, config, or hs file in BTi folder.")

		raw = mne.io.read_raw_bti(
			pdf_fname=str(raw_fname),
			config_fname=str(config_fname),
			head_shape_fname=str(hs_fname),
			preload=preload,
			verbose=verbose,
		)
	
	else:
		raise ValueError("Unrecognized file or folder type for MEG data.")

	if bad_channels:
		raw.drop_channels(bad_channels)
	
	return raw

def load_gt(subject_path, gt_names):
	raw = read_raw(subject_path, preload=False, verbose=False, bad_channels=None)
	
	# Convert annotations to DataFrame
	annotations_df = raw.annotations.to_data_frame()
	
	# Convert 'onset' column to UTC-aware timestamps
	annotations_df['onset'] = pd.to_datetime(annotations_df['onset']).dt.tz_localize('UTC')
	
	# Calculate seconds relative to raw.annotations.orig_time
	origin_time = pd.Timestamp(raw.annotations.orig_time)
	annotations_df['onset'] = (annotations_df['onset'] - origin_time).dt.total_seconds()

	# Filter annotations with description in ['spike', 'spikes']
	spike_onsets = annotations_df[
		annotations_df['description'].str.lower().isin(gt_names)
	]['onset'].tolist()
	
	return spike_onsets

for subject in data_dir.rglob("*.ds"):
	print(subject)
	if os.path.basename(subject) != "hz.ds":
		# Load ground truth spike times
		gt_spike_times = load_gt(subject, gt_names) 

		gt_df = pd.DataFrame({
			"Patient": [os.path.basename(subject)] * len(gt_spike_times),
			"SpikeTime_s": gt_spike_times
		})

		gt_df.to_csv(os.path.join(output_path, "ground_truth.csv"), mode='a', index=False, header=not os.path.exists(os.path.join(output_path, "ground_truth.csv")))
