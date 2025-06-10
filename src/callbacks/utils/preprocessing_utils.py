# Dash & Plotly
import dash
from dash import dcc
import plotly.graph_objects as go

# Standard Library
import os
import hashlib

# Third-party Libraries
import numpy as np
import mne
import dask.dataframe as dd
from dask import delayed
from flask_caching import Cache

# Local Modules
import config
from callbacks.utils import folder_path_utils as fpu
from callbacks.utils import cache_utils as cu

app = dash.get_app()

cache = Cache(app.server, config={
    'CACHE_TYPE': config.CACHE_TYPE,
    'CACHE_DIR': config.CACHE_DIR,
    'CACHE_DEFAULT_TIMEOUT': config.CACHE_DEFAULT_TIMEOUT
})

################################# RAW DATA PREPROCESSING (FILTERING, SUBSAMPLING) #################################################################

def filter_resample(folder_path, freq_data):
      
    raw = fpu.read_raw(folder_path, preload=True, verbose=False)
    
    resample_freq = freq_data.get("resample_freq")
    low_pass_freq = freq_data.get("low_pass_freq")
    high_pass_freq = freq_data.get("high_pass_freq")
    notch_freq = freq_data.get("notch_freq")

    # Apply filtering and resampling
    raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
    raw.notch_filter(freqs=notch_freq)
    raw.resample(resample_freq)

    # raw.rename_channels({ch['ch_name']: ch['ch_name'].split('-')[0] for ch in raw.info['chs']})

    return raw


############################### MAIN CACHED FUNCTIONS (PREPROCESSING) ##############################################

def get_max_length(raw, resample_freq):
    return raw.n_times/raw.info['sfreq']-1/resample_freq

def update_chunk_limits(total_duration):
    chunk_duration = config.CHUNK_RECORDING_DURATION
    chunk_limits= [
        [start, min(start + chunk_duration, total_duration)]
        for start in range(0, int(total_duration), chunk_duration)
    ]
    return chunk_limits

def get_cache_filename(folder_path, freq_data, start_time, end_time, cache_dir=f"{config.CACHE_DIR}"):
    # Create a unique hash key
    hash_input = f"{folder_path}_{freq_data}_{start_time}_{end_time}"
    hash_key = hashlib.md5(hash_input.encode()).hexdigest()
    return os.path.join(cache_dir, f"{hash_key}.parquet")

def get_preprocessed_dataframe_dask(folder_path, freq_data, start_time, end_time, prep_raw=None, cache_dir=f"{config.CACHE_DIR}"):
    os.makedirs(cache_dir, exist_ok=True)
    cu.clear_old_cache_files(cache_dir)
    cache_file = get_cache_filename(folder_path, freq_data, start_time, end_time, cache_dir)

    # If cache exists, load and return
    if os.path.exists(cache_file):
        return dd.read_parquet(cache_file)

    # Otherwise, compute and save
    @delayed
    def load_and_filter():
        return filter_resample(folder_path, freq_data)

    @delayed
    def crop_and_to_df(prep_raw):
        raw_chunk = prep_raw.copy().crop(tmin=start_time, tmax=end_time)
        return raw_chunk.to_data_frame(picks="meg", index="time")

    @delayed
    def standardize(raw_df):
        return raw_df - raw_df.mean(axis=0)
    
    # Chain and compute
    if prep_raw is None:
        prep_raw = load_and_filter()
    raw_df = crop_and_to_df(prep_raw)
    raw_df_std = standardize(raw_df)

    df = raw_df_std.compute()
    ddf = dd.from_pandas(df, npartitions=10)
    ddf.to_parquet(cache_file)

    return ddf

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
    def process_data_in_chunks(folder_path, freq_data, start_time, end_time, prep_raw=None):
        try:
            if prep_raw is None:
                prep_raw = filter_resample(folder_path, freq_data)

            raw_chunk = prep_raw.copy().crop(tmin=start_time, tmax=end_time)
            raw_df = raw_chunk.to_data_frame(picks="meg", index="time")  # Get numerical data (channels × time)
            raw_df_standardized = raw_df - raw_df.mean(axis = 0) # Standardization per channel

            return raw_df_standardized

        except Exception as e:
            return f"⚠️ Error during processing: {str(e)}"

    # Process and return the result in JSON format
    return process_data_in_chunks(folder_path, freq_data, start_time, end_time, raw)

###################################################### ICA ########################################################################

def get_cache_filename_ica(folder_path, start_time, end_time, ica_result_path, cache_dir):
    hash_input = f"{folder_path}_{start_time}_{end_time}_{ica_result_path}"
    hash_key = hashlib.md5(hash_input.encode()).hexdigest()
    return os.path.join(cache_dir, f"{hash_key}.parquet")

def get_ica_dataframe_dask(folder_path, start_time, end_time, ica_result_path, raw=None, cache_dir=f"{config.CACHE_DIR}"):
    os.makedirs(cache_dir, exist_ok=True)
    cu.clear_old_cache_files(cache_dir)

    cache_file = get_cache_filename_ica(folder_path, start_time, end_time, ica_result_path, cache_dir)

    if os.path.exists(cache_file):
        return dd.read_parquet(cache_file)

    if raw is None:
        raw = fpu.read_raw(folder_path, preload=True, verbose=False).pick_types(meg=True)
        raw.filter(l_freq=1.0, h_freq=None)

    raw = raw.copy().crop(tmin=start_time, tmax=end_time)

    ica = mne.preprocessing.read_ica(ica_result_path)
    sources = ica.get_sources(raw)

    data = sources.get_data()
    sfreq = sources.info['sfreq']
    ch_names = sources.ch_names
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='misc')  # ICA comps = misc

    new_sources = mne.io.RawArray(data, info)
    sources = new_sources.resample(300)
    sources_df = sources.to_data_frame(index="time")  # Get numerical data (channels × time)

    # Convert to Dask array
    ddf = dd.from_pandas(sources_df, npartitions=10)
    ddf = ddf - ddf.mean(axis=0)
    ddf.to_parquet(cache_file)

    return ddf

################################## POWER SPECTRUM DECOMPOSITION ######################################################

def compute_power_spectrum_decomposition(folder_path, freq_data):
    raw = fpu.read_raw(folder_path, preload=True, verbose=False)

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

    psd_fig = go.Figure()

    for ch_idx, ch_name in enumerate(psd_data.ch_names):
        psd_fig.add_trace(go.Scatter(
            x=freqs,
            y=psd_dB[ch_idx],  
            mode='lines',
            line=dict(width=1),
            name=ch_name
        ))

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
        template="plotly_dark"
    )

    return dcc.Graph(figure = psd_fig)

