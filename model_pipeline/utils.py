import pickle
import numpy as np
import os.path as op
import mne
from tqdm import tqdm
import pandas as pd
import model_pipeline.params as params
from scipy.ndimage import gaussian_filter1d

#####################################################################Preparing the data

#To read and write pickle files
def save_obj(obj, name, path):
	with open(op.join(path, name + '.pkl'), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path):
	with open(op.join(path, name), 'rb') as f:
		return pickle.load(f)

# center scale each window using all window mean and std
def standardize(X,mean=False,std=False):
	if not mean:
		mean = np.mean(X,axis=(1,2))
	if not std:
		std = np.std(X,axis=(1,2))
	X_stand=np.zeros(X.shape)
	nb_data=X.shape[0]
	for i in range(0,nb_data,1):
		X_stand[i,:,:] = (X[i,:,:] - mean[i]) / std[i]
	return X_stand

# tentative function to interpolate missing channels using mne 
def interpolate_missing_channels(raw, good_channels, loc_meg_channels):
	print("interpolate")
	existing_channels = raw.info['ch_names'] # returns the list of chanel names that are present in the data
	missing_channels = list(set(good_channels) - set(existing_channels)) # gets the list of missing channels by comparing the existing channel names with the list of good channels
	new_raw = raw.copy() 
	# missing_channels.append('MRO23-2805')
	# new_raw.drop_channels('MRO23-2805')

	# creates fake channels and set them to "bad channels", rename them with the name of the missing channels, 
	#then mne is supposed to be able to reconstruct bad channels with "interpolate_bads" 
	for miss in missing_channels:
		to_copy = raw.info['ch_names'][71] #pick a random channel
		new_channel = raw.copy().pick([to_copy])
		new_channel.rename_channels({to_copy: miss})
		new_raw.add_channels([new_channel], force_update_info=True)

		#specifies the location of the missing channel
		for i in range(len(new_raw.info['chs'])):
			if new_raw.info['chs'][i]['ch_name'] == miss:
				new_raw.info['chs'][i]['loc'] = loc_meg_channels[miss]
			
	new_raw.reorder_channels(good_channels)
	new_raw.info['bads'] = missing_channels

	new_raw.interpolate_bads(origin=(0, 0, 0.04),reset_bads=True) 

	return new_raw

#Applies preprocessing, extracts and saves the data in pickle
def save_data_matrices(good_channels_file, subject, path_output):

	# open a file containing the good 274 channels
	with open(good_channels_file, 'rb') as fp:
		good_channels = pickle.load(fp)

	with open(params.loc_meg_channels_path, 'rb') as fp: #path to the file.pkl containing for each channel name its location
		loc_meg_channels = pickle.load(fp)

	raw_names = list([subject])
	data = dict()
	all_raws = list()
	all_files = list()
	#FOR EACH .DS
	for raw_file in (raw_names):
		# try:
		raw_file=raw_file
		raw = mne.io.read_raw_ctf(raw_file, preload=True, verbose=False)
		if "Liogier_AllDataset1200Hz" in raw_file:
			raw.drop_channels('MRO23-2805')

		#Resample the data
		raw.resample(params.sfreq).pick(['mag'])
		raw=interpolate_missing_channels(raw, good_channels, loc_meg_channels)
		
		raw.filter(0.5,50, n_jobs=8)

		raw = raw.get_data()
		all_raws.append(raw)
		all_files.append(raw_file)
			
		data['meg'] = all_raws
		data['file'] = all_files

	#Saves a pickle file -> one pickle file for each patient
	#Access MEG data from first .ds: data['meg'][0]
	save_obj(data, 'data_raw_%s' % params.subject_number, path_output)

#Crops the windows from the pickle file and saves it in a binary file
def create_windows(path_output, window_size_ms, stand=False):

	total_nb_windows = 0
	#Window size in time points (based on window duration)
	window_size = window_size_ms * params.sfreq
	#Spacing between two window centers (made such that windows slightly overlap)
	window_spacing = (window_size_ms - 2*params.spike_spacing_from_border_ms) * params.sfreq

	data = load_obj('data_raw_%s' % params.subject_number+'.pkl', path_output)

	X_all = np.empty((0,274,int(window_size))) # Will store MEG windows
	window_center_time = list() # Will store MEG window center timing
	nb_block = list()  # Will store MEG window block to be able to go back to original .ds files 


	for block in range(len(data['meg'])):
		X_block = list()  # Will store MEG windows for the current .ds
		# get data
		block_data = data['meg'][block]

		# Slice in short windows (seconds)
		# get the center of each windows in seconds
		window_centers = np.arange(window_size/2, block_data.shape[1], window_spacing)
		print(window_centers)
		# Start getting the data for each window
		for window_center in tqdm(window_centers):
			# get the data only if time duration is big enough before and after the
			# center of the window
			if (window_size/2. <= window_center <= block_data.shape[1] - window_size/2.):
				# get low and high limit of the window in samples
				low_limit = round((window_center - window_size/2.))
				high_limit = round((window_center + window_size/2. +0.1))# Because of odd window size
				X_block.append(block_data[:, low_limit:high_limit])
				window_center_time.append(window_center)
				nb_block.append(block)
				total_nb_windows = total_nb_windows+1

		X_block = np.stack(X_block,axis=0)       
		X_all = np.append(X_all,X_block,axis=0)

	if stand:
		X_all = standardize(X_all)      

	#SAVES INFOS FOR A GIVEN SUBJECT
	# cast MEG data to float32. VERY IMPORTANT as we then save them in a binary file and this info allows us to know how many bytes to read in the binary
	X_all = X_all.astype('float32')
	##saves windows to binary file (for faster reading later)
	X_all.tofile(f"{path_output}/data_raw_{params.subject_number}_windows_bi")
	#saves window center timings
	save_obj(np.array(window_center_time), "data_raw_"+str(params.subject_number)+'_timing', path_output)
	#saves window blocks
	save_obj(np.array(nb_block), "data_raw_"+str(params.subject_number)+'_blocks', path_output)

	return total_nb_windows

def generate_database(total_nb_windows):  

	X_test_ids = np.zeros((total_nb_windows,3),dtype=int)
	X_test_ids[:,0] = np.linspace(0,total_nb_windows-1,num=total_nb_windows,dtype=int)
	X_test_ids[:,1] =  np.ones((total_nb_windows),dtype=int)*params.subject_number

	return X_test_ids

########################################################################### Compute features functions ###########################################################################
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
	ampl = (np.max(window, axis=0)-np.min(window, axis=0))
	mean = np.mean(window, axis=0)
	mean[mean==0]=1
	return ampl/mean

def compute_window_sharpness(window):
	slopes = np.diff(window, axis=0)
	return np.max(np.abs(slopes[1:]-slopes[:-1]), axis=0)

# def compute_window_main_frequency(window):
# 	n = len(window)
# 	fft_result = fft(window)
# 	frequencies = fftfreq(n)
# 	amplitudes = np.abs(fft_result)
# 	peak_frequency_index = np.argmax(amplitudes)
# 	main_frequency = frequencies[peak_frequency_index]
# 	return main_frequency

# def compute_window_phase_congruency(window):
# 	fft_window = fft(window)
# 	phases = np.angle(fft_window)
# 	phase_diff = np.diff(phases)
# 	phase_congruencies = 1 - (np.abs(phase_diff)/np.pi)
	# return np.max(phase_congruencies)

def compute_gfp(window):
    """Compute Global Field Power (GFP) as standard deviation across channels."""
    return np.std(window, axis=0)

def find_peak_gfp(gfp, times, smoothing_sigma=2, percentile=90):
    """Find the peak GFP time within a window after smoothing."""
    gfp_smooth = gaussian_filter1d(gfp, sigma=smoothing_sigma)
    
    # Thresholding: Find first peak above percentile threshold
    threshold = np.percentile(gfp_smooth, percentile)
    peak_indices = np.where(gfp_smooth >= threshold)[0]
    
    if len(peak_indices) > 0:
        return times[peak_indices[0]]  # First peak above threshold
    else:
        return times[np.argmax(gfp_smooth)]  # Default to max if no peak found

#####################################################################Saving Predictions As MarkerFile

def write_mrk_file(filepath,raw_name,onset_list_detected_spikes, marker_name):

	with open(raw_name[:-3]+'.mrk', 'w') as f:
		f.write('PATH OF DATASET:\n'+filepath+' \n\n\nNUMBER OF MARKERS:\n1\n\n\n')
		f.write('CLASSGROUPID:\n3\nNAME:\n'+marker_name+'\nCOMMENT:\n\nCOLOR:\ngreen\nEDITABLE:\nYes\nCLASSID:\n1\nNUMBER OF SAMPLES:\n' + str(len(onset_list_detected_spikes)) + '\nLIST OF SAMPLES:\nTRIAL NUMBER      TIME FROM SYNC POINT (in seconds)\n')
		for annot in onset_list_detected_spikes:
			f.write('                  +0                    +'+str(annot))
			f.write('\n')
		f.write('\n\n')


def generate_mrk_from_preds(path_output, subject):
	df = pd.read_csv(path_output+'subject_'+str(params.subject_number)+'_predictions.csv')

	#raw_names = [r for r in os.listdir(params.subject) if ('.ds' in r) and not ('._' in r) and not ('misc' in r)]
	#raw_names.sort()
	
	#print(raw_names)
	raw_names = list([subject])
	for ind,i in enumerate(raw_names):
		current_block_df = df.loc[df['block'] == ind]
		detected_spike_only = current_block_df.loc[current_block_df['pred'] == 1]
		onset_detected_spikes = (detected_spike_only['timing']/150).tolist()
		write_mrk_file(subject,i,onset_detected_spikes)
