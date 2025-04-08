import static.constants as c
from pathlib import Path
import pickle
import mne
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

cache.clear()


################################# RAW DATA PREPROCESSING (FILTERING, SUBSAMPLING) #################################################################

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
    def preprocess_chunk(folder_path, freq_data, start_time, end_time, raw):
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
            print(raw.times)
            raw_chunk = raw.copy().crop(tmin=start_time, tmax=end_time)

            # Transform the raw data into a dataframe
            raw_df = raw_chunk.to_data_frame(picks="meg", index="time")  # Get numerical data (channels × time)

            # Standardization per channel
            raw_df_standardized = raw_df - raw_df.mean(axis = 0) #.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)

            return raw_df_standardized

        except Exception as e:
            return f"Error during preprocessing chunk: {str(e)}"

    # Function to load and process the data in chunks, caching each piece
    @cache.memoize()
    def process_data_in_chunks(folder_path, freq_data, start_time, end_time, raw):
        try:
            chunk_df = preprocess_chunk(folder_path, freq_data, start_time, end_time, raw)
            return chunk_df

        except Exception as e:
            return f"Error during processing: {str(e)}"

    # Process and return the result in JSON format
    return process_data_in_chunks(folder_path, freq_data, start_time, end_time, raw)

###################################################### ICA ########################################################################

@cache.memoize()
def compute_ica(folder_path, n_components, ica_method, max_iter, decim):

    path = f"{Path(folder_path).stem}_{n_components}_{ica_method}_{max_iter}_{decim}-ica.fif"

    if Path(path).exists():
         print("it already exists)")
         return str(path)

    raw = mne.io.read_raw_ctf(folder_path, preload=True)
    raw = raw.pick_types(meg=True)
    raw = raw.filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=ica_method,
        max_iter=max_iter,
        random_state=97
    )
    ica.fit(raw, decim=decim)
    ica.save(path)

    return path

@cache.memoize()
def get_ica_sources_for_chunk(folder_path, start_time, end_time, n_components, ica_method, max_iter, decim):
    # Get the cached ICA from step 1
    ica_path = compute_ica(folder_path, n_components, ica_method, max_iter, decim)

    # Load and crop raw data for the chunk
    raw = mne.io.read_raw_ctf(folder_path, preload=True)
    raw = raw.pick_types(meg=True)
    raw = raw.filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.read_ica(ica_path)

    # Use cached ICA to get sources for the chunk
    sources = ica.get_sources(raw)

    data = sources.get_data()
    sfreq = sources.info['sfreq']
    ch_names = sources.ch_names
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='misc')  # ICA comps = misc
    new_sources = mne.io.RawArray(data, info)

    sources = new_sources.resample(300)

    # Transform the raw data into a dataframe
    ica_df = sources.to_data_frame(index="time")  # Get numerical data (channels × time)

    # Standardization per channel
    ica_df_standardized = (ica_df - ica_df.mean()) / ica_df.std()

    return ica_df_standardized


# def get_ica(folder_path, ica, start_time, end_time, raw=None):
#     """
#     Compute ICA of the MEG data and cache each chunks.
#     """
    
#     # Helper function to preprocess a chunk of the data
#     def ica_by_chunk(folder_path, ica, start_time, end_time, raw):

#         if raw is None:
#             # Load the MEG data 
#             # Assuming folder_path is where the data is stored and chunk_limits is the relevant chunk for analysis
#             raw = mne.io.read_raw_ctf(folder_path, preload=True).pick("meg")  # Example for FIF format

#             # High-pass filtering to remove low-frequency drifts (1 Hz cutoff recommended)
#             raw = raw.filter(l_freq=1.0, h_freq=None)  # Apply 1 Hz high-pass filter

#         raw_chunk = raw.copy().crop(tmin=start_time, tmax=end_time)
            
#         ica_sources = ica.get_sources(raw_chunk)

#         ica_sources.resample(300)

#         # Transform the raw data into a dataframe
#         ica_df = ica_sources.to_data_frame(index="time")  # Get numerical data (channels × time)

#         # Standardization per channel
#         ica_df_standardized = (ica_df - ica_df.mean()) / ica_df.std()


#         return ica_df_standardized

#     # Function to load and process the data in chunks, caching each piece
#     @cache.memoize()
#     def compute_ica_on_chunk(folder_path, ica, start_time, end_time, raw):
#         try:
#             chunk_df = ica_by_chunk(folder_path, ica, start_time, end_time, raw)
#             return chunk_df
#         except Exception as e:
#             return f"Error during ICA: {str(e)}"

#     # Process and return the result in JSON format
#     return compute_ica_on_chunk(folder_path, ica, start_time, end_time, raw)
 
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

