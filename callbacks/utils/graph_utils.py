import numpy as np

import static.constants as c
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler


def calculate_channel_offset(num_channels, plot_height=500, min_gap=50):
    """
    Calculate the optimal channel offset to avoid overlap of traces in the plot.
    
    Parameters:
    - num_channels: The number of channels selected for the plot.
    - plot_height: The height of the plot (figure).
    - min_gap: The minimum gap (in pixels) between channels.
    
    Returns:
    - optimal_channel_offset: The calculated offset between channels.
    """
    # Ensure there's enough space between each trace to avoid overlap.
    # We leave space for the desired minimum gap between traces.
    total_gap_needed = (num_channels - 1) * min_gap
    
    # Calculate the optimal offset based on the plot height and the required gap.
    optimal_channel_offset = (plot_height - total_gap_needed) / (num_channels - 1) if num_channels > 1 else min_gap
    
    # Ensure the offset is a positive value
    optimal_channel_offset = max(optimal_channel_offset, min_gap)
    
    return optimal_channel_offset

def get_y_axis_ticks(selected_channels, channel_offset = c.DEFAULT_Y_AXIS_OFFSET):
    
    channel_offset = (channel_offset if channel_offset != None else c.DEFAULT_Y_AXIS_OFFSET)
    y_axis_ticks = np.arange(len(selected_channels)) * channel_offset
    return y_axis_ticks

def normalize_dataframe_columns(df):
    scaler = StandardScaler()

    df_standardized = df.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)

    return df_standardized

def get_filtered_df(time_range, raw_df, annotations_df):

    times = (raw_df.index - raw_df.index[0]).total_seconds()
    time_mask = (times >= time_range[0]) & (times <= time_range[1])
    filtered_times = times[time_mask]
    filtered_raw_df = raw_df[time_mask]
    annotations_times = annotations_df.index
    annotations_time_mask = (annotations_times >= time_range[0]) & (annotations_times <= time_range[1])
    filtered_annotations_df = annotations_df[annotations_time_mask]
    
    return filtered_times, filtered_raw_df, filtered_annotations_df

