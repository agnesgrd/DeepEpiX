import pandas as pd
import os
from pathlib import Path
import mne
import config

output_path = "src/results"
gt_names = ['MEG']
data_dir = config.DATA_DIR

def read_raw(folder_path, preload, verbose, bad_channels=None):
	folder_path = Path(folder_path)

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

def extract_meg_timepoints_from_mrk(file_path):
    """
    Extracts timepoints labeled 'MEG', 'meg left', or 'meg right' from a .mrk file (AnyWave marker file),
    accounting for cases where -1 indicates the actual timepoint follows.
    
    Args:
        file_path (str): Path to the .mrk file.
    
    Returns:
        List[float]: List of MEG timepoints.
    """
    meg_timepoints = []
    meg_labels = {"meg", "meg left", "meg right"}

    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("//") or not stripped:
                continue

            parts = stripped.split()
            label = parts[0].lower()

            if label in meg_labels and len(parts) >= 4:
                try:
                    time_val = float(parts[2])
                    if time_val == -1 and len(parts) > 3:
                        # Try the next part after -1
                        time_val = float(parts[3])
                    if time_val != -1:
                        meg_timepoints.append(time_val)
                except ValueError:
                    continue

    return meg_timepoints

def get_mrk_annotations_dataframe(folder_path):
    mrk_file_path = None
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(".mrk"):
                mrk_file_path = os.path.join(folder_path, file)
                break

    # --- If .mrk file found, extract MEG timepoints and add to annotations ---
    if mrk_file_path and os.path.exists(mrk_file_path):
        meg_timepoints = extract_meg_timepoints_from_mrk(mrk_file_path)

    return meg_timepoints


for subject in data_dir.rglob("*.ds"):
	print(subject)
	if os.path.basename(subject) != "hz.ds":
		# Load ground truth spike times
		gt_spike_times = get_mrk_annotations_dataframe(subject)

		gt_df = pd.DataFrame({
			"Patient": [os.path.basename(subject)] * len(gt_spike_times),
			"SpikeTime_s": gt_spike_times
		})

		gt_df.to_csv(os.path.join(output_path, "ground_truth.csv"), mode='a', index=False, header=not os.path.exists(os.path.join(output_path, "ground_truth.csv")))
