# Dash & Plotly
import dash
from dash import dcc
import plotly.graph_objects as go

# Standard library
import math
import pickle
from pathlib import Path
import numpy as np

# Third-party libraries
import mne
from flask_caching import Cache

# Local modules
import config

app = dash.get_app()

cache = Cache(app.server, config={
    'CACHE_TYPE': config.CACHE_TYPE,
    'CACHE_DIR': config.CACHE_DIR,
    'CACHE_DEFAULT_TIMEOUT': config.CACHE_DEFAULT_TIMEOUT
})

################################# RAW DATA PREPROCESSING (FILTERING, SUBSAMPLING) #################################################################

def filter_resample(folder_path, freq_data):
      
    raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)
    
    resample_freq = freq_data.get("resample_freq")
    low_pass_freq = freq_data.get("low_pass_freq")
    high_pass_freq = freq_data.get("high_pass_freq")
    notch_freq = freq_data.get("notch_freq")

    # Apply filtering and resampling
    raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
    raw.notch_filter(freqs=notch_freq)
    raw.resample(resample_freq)

    return raw


############################### MAIN CACHED FUNCTIONS (PREPROCESSING AND ICA) ##############################################
      

def get_preprocessed_dataframe(folder_path, freq_data, start_time, end_time, raw=None):
    """
    Preprocess the MEG data in chunks and cache them.

    :param folder_path: Path to the raw data file.
    :param freq_data: Dictionary containing frequency parameters for preprocessing.
    :param chunk_duration: Duration of each chunk in seconds (default is 3 minutes).
    :param cache: Cache object to store preprocessed chunks.
    :return: Processed dataframe in pandas format.
    """
    # Function to load and process the data in chunks, caching each piece

    @cache.memoize(make_name=f"{folder_path}:{freq_data}:{start_time}:{end_time}")
    def process_data_in_chunks(folder_path, freq_data, start_time, end_time, raw=None):
        try:
            if raw is None:
                prep_raw = filter_resample(folder_path, freq_data)

            # Crop the raw data to the chunk's time range
            raw_chunk = prep_raw.copy().crop(tmin=start_time, tmax=end_time)

            # Transform the raw data into a dataframe
            raw_df = raw_chunk.to_data_frame(picks="meg", index="time")  # Get numerical data (channels × time)

            # Standardization per channel
            raw_df_standardized = raw_df - raw_df.mean(axis = 0)

            return raw_df_standardized

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

 
def update_chunk_limits(total_duration):
    # Define chunk duration (in seconds) and total duration
    chunk_duration = config.CHUNK_RECORDING_DURATION

    # Calculate start and end times for the selected chunk
    chunk_limits = []
    chunk_number = math.ceil(total_duration/chunk_duration)
    for chunk_idx in range(chunk_number):
        start_time = chunk_idx * chunk_duration
        end_time = min(start_time + chunk_duration, total_duration)
        chunk_limits.append([start_time, end_time])

    return chunk_limits


################################## POWER SPECTRUM DECOMPOSITION ######################################################

def compute_power_spectrum_decomposition(folder_path, freq_data):
    raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)

    resample_freq = freq_data.get("resample_freq")
    low_pass_freq = freq_data.get("low_pass_freq")
    high_pass_freq = freq_data.get("high_pass_freq")
    notch_freq = freq_data.get("notch_freq")

    if not low_pass_freq or not high_pass_freq or not notch_freq:
        return dash.no_update
    
    raw.notch_filter(freqs=notch_freq)

    # Compute Power Spectral Density (PSD)
    psd_data = raw.compute_psd(method='welch', fmin=high_pass_freq, fmax=low_pass_freq, n_fft=2048, picks='meg', n_jobs=-1)
    psd, freqs = psd_data.get_data(return_freqs=True)

    # Convert PSD to dB (as MNE does by default)
    psd_dB = 10 * np.log10(psd)

    # Create a Plotly figure
    psd_fig = go.Figure()

    # Plot multiple channels with transparency for better readability
    for ch_idx, ch_name in enumerate(config.ALL_CH_NAMES_PREFIX):
        psd_fig.add_trace(go.Scatter(
            x=freqs,
            y=psd_dB[ch_idx],  
            mode='lines',
            line=dict(width=1),
            name=ch_name
        ))

    # Update layout to match MNE’s default style
    psd_fig.update_layout(
        title="Power Spectral Density (PSD)",
        xaxis=dict(
            title="Frequency (Hz)",
            type="linear",  # MNE uses linear frequency scale
            showgrid=True
        ),
        yaxis=dict(
            title="Power (dB)",  # Log scale power in dB
            type="linear",
            showgrid=True
        ),
        # template="plotly_white"
    )

    return dcc.Graph(figure = psd_fig, style={"padding": "10px", "borderRadius": "10px", "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"})

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

