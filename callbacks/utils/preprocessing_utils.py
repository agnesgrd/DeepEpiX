from math import *
import static.constants as c
from pathlib import Path
import pickle

def update_chunk_limits(total_duration):
    # Define chunk duration (in seconds) and total duration
    chunk_duration = c.CHUNK_RECORDING_DURATION

    # Calculate start and end times for the selected chunk
    chunk_limits = []
    chunk_number = ceil(total_duration/chunk_duration)
    for chunk_idx in range(chunk_number):
        start_time = chunk_idx * chunk_duration
        end_time = min(start_time + chunk_duration, total_duration)
        chunk_limits.append([start_time, end_time])
        
    # Return chunk limits in a dictionary to store in the dcc.Store
    return chunk_limits

# tentative function to interpolate missing channels using mne 
def interpolate_missing_channels(raw):
	print("interpolate missing or bad channel")
	
    # open a file containing the good 274 channels
	with open(Path.cwd() / "model_pipeline/good_channels", 'rb') as fp:
		good_channels = pickle.load(fp)

	with open(Path.cwd() / "model_pipeline/loc_meg_channels.pkl", 'rb') as fp: #path to the file.pkl containing for each channel name its location
		loc_meg_channels = pickle.load(fp)
		
	existing_channels = raw.info['ch_names'] # returns the list of chanel names that are present in the data
	missing_channels = list(set(good_channels) - set(existing_channels)) # gets the list of missing channels by comparing the existing channel names with the list of good channels
	new_raw = raw.copy() 

	# creates fake channels and set them to "bad channels", rename them with the name of the missing channels, 
	#then mne is supposed to be able to reconstruct bad channels with "interpolate_bads" 
	for miss in missing_channels:
		to_copy = raw.info['ch_names'][50] #pick a random channel
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

