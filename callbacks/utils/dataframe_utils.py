from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_raw_df_filtered_on_time_dask(time_range, raw_ddf):
    raw_ddf = raw_ddf.reset_index()  # Ensure time is a column
    raw_ddf['time_seconds'] = (raw_ddf['time'] - raw_ddf['time'].iloc[0]).dt.total_seconds()

    filtered_ddf = raw_ddf[(raw_ddf['time_seconds'] >= time_range[0]) & (raw_ddf['time_seconds'] <= time_range[1])]
    
    return filtered_ddf['time_seconds'], filtered_ddf.set_index('time')

def get_annotations_df_filtered_on_time_dask(time_range, annotations_ddf):
    annotations_ddf = annotations_ddf.reset_index()
    filtered = annotations_ddf[
        (annotations_ddf['time'] >= time_range[0]) &
        (annotations_ddf['time'] <= time_range[1])
    ]
    return filtered.set_index('time')

def get_raw_df_filtered_on_time(time_range, raw_df):

    times = (raw_df.index - raw_df.index[0]).total_seconds()
    time_mask = (times >= time_range[0]) & (times <= time_range[1])
    filtered_times = times[time_mask]
    filtered_raw_df = raw_df[time_mask]
    
    return filtered_times, filtered_raw_df

def get_annotations_df_filtered_on_time(time_range, annotations_df):

    times = annotations_df.index
    time_mask = (times >= time_range[0]) & (times <= time_range[1])
    filtered_annotations_df = annotations_df[time_mask]
    
    return filtered_annotations_df

def get_shifted_time_axis_dask(time_range, raw_ddf):
    # Get the first timestamp from the index
    index = raw_ddf.index.compute()
    times = index - index[0]
    adjusted_times = times + time_range[0]
    return adjusted_times

def get_shifted_time_axis(time_range, raw_df):
    """ Adjusts the time axis of a DataFrame to include an offset based on the given time range."""
    times = (raw_df.index - raw_df.index[0])
    adjusted_times = times + time_range[0]
    return adjusted_times