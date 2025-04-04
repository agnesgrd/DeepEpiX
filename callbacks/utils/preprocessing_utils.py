import static.constants as c
from pathlib import Path
import pickle
import mne
from sklearn.preprocessing import StandardScaler
import dash
from flask_caching import Cache
import math

app = dash.get_app()

cache = Cache(app.server, config={
    # 'CACHE_TYPE': 'redis',
    # 'CACHE_REDIS_HOST': 'localhost',    # Redis server hostname
    # 'CACHE_REDIS_PORT': 6379,          # Redis server port
    # 'CACHE_REDIS_DB': 0,               # Redis database index
    # 'CACHE_REDIS_URL': 'redis://localhost:6379/0',  # Redis connection URL
    'CACHE_TYPE': 'FileSystemCache',
    # 'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 84000,
    # 'CACHE_THRESHOLD': 50 # higher numbers will store more data in the filesystem / redis cache
})

def get_preprocessed_dataframe(folder_path, freq_data, start_time, end_time, raw=None):
    """
    Preprocess the MEG data in chunks and cache them.

    :param folder_path: Path to the raw data file.
    :param freq_data: Dictionary containing frequency parameters for preprocessing.
    :param chunk_duration: Duration of each chunk in seconds (default is 3 minutes).
    :param cache: Cache object to store preprocessed chunks.
    :return: Processed dataframe in JSON format.
    """
    # Helper function to preprocess a chunk of the data
    def preprocess_chunk(start_time, end_time, raw, freq_data):
        try:
            if raw is None:
                raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)
                resample_freq = freq_data.get("resample_freq")
                low_pass_freq = freq_data.get("low_pass_freq")
                high_pass_freq = freq_data.get("high_pass_freq")
                notch_freq = freq_data.get("notch_freq")
                # Apply filtering and resampling
                raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
                raw.notch_filter(freqs=notch_freq)
                raw.resample(resample_freq)

            # Crop the raw data to the chunk's time range
            raw_chunk = raw.copy().crop(tmin=start_time, tmax=end_time)

            # Transform the raw data into a dataframe
            raw_df = raw_chunk.to_data_frame(picks="meg", index="time")  # Get numerical data (channels × time)

            # Standardization per channel
            scaler = StandardScaler()
            raw_df_standardized = raw_df - raw_df.mean(axis = 0) #.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)

            return raw_df_standardized

        except Exception as e:
            return f"Error during preprocessing chunk: {str(e)}"

    # Function to load and process the data in chunks, caching each piece
    @cache.memoize()
    def process_data_in_chunks(folder_path, freq_data, start_time, end_time):
        try:
            chunk_df = preprocess_chunk(start_time, end_time, raw, freq_data)
            return chunk_df

        except Exception as e:
            return f"Error during processing: {str(e)}"

    # Process and return the result in JSON format
    return process_data_in_chunks(folder_path, freq_data, start_time, end_time)
 
def update_chunk_limits(total_duration):
    # Define chunk duration (in seconds) and total duration
    chunk_duration = c.CHUNK_RECORDING_DURATION

    # Calculate start and end times for the selected chunk
    chunk_limits = []
    chunk_number = math.ceil(total_duration/chunk_duration)
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

