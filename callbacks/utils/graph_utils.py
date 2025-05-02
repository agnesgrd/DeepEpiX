import numpy as np
import plotly.graph_objects as go

import config
from layout import DEFAULT_FIG_LAYOUT, REGION_COLOR_PALETTE
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import dataframe_utils as du
from callbacks.utils import smoothgrad_utils as su

def calculate_channel_offset_std(signal_df, scale_factor=1, max_scale_factor=11, min_offset=10):
    stds = signal_df.std(skipna=True)
    mean_std = stds.mean()*2
    return max(min_offset, mean_std * (max_scale_factor-scale_factor))

def get_y_axis_ticks_with_gap(channel_names, base_offset, group_gap=2):
    """
    Compute y-axis ticks with additional spacing when the side (L/R/Z) changes.

    Parameters:
    - channel_names: list of channel names (e.g., ['MRC61-2805', 'MLC23-2805'])
    - base_offset: float, base vertical spacing between channels
    - group_gap: multiplier to insert larger gap when group changes

    Returns:
    - np.array of y-axis positions with extra group separation
    """
    y_ticks = []
    current_y = 0
    previous_side = None

    for name in channel_names:
        side = name[1] if len(name) > 1 else ""

        if previous_side is not None and side != previous_side:
            current_y += base_offset * group_gap  # Add extra space

        y_ticks.append(current_y)
        current_y += base_offset
        previous_side = side

    return np.array(y_ticks)

def apply_default_layout(fig, xaxis_range, time_range, folder_path):
    """
    Apply default layout to a Plotly figure with dynamic values for axis range, title, etc.

    Parameters:
    - fig: The Plotly figure to update.
    - xaxis_range: Range for the x-axis (list or tuple).
    - time_range: Time range for the x-axis (tuple or list with two values [start, end]).
    - folder_path: Title for the plot.

    Returns:
    - Updated Plotly figure with applied layout.
    """
    layout = DEFAULT_FIG_LAYOUT.copy()
    layout['xaxis']['range'] = xaxis_range
    layout['xaxis']['minallowed'] = time_range[0]
    layout['xaxis']['maxallowed'] = time_range[1]
    fig.update_layout(layout)
    return fig

def generate_graph_time_channel(selected_channels, offset_selection, time_range, folder_path, freq_data, color_selection, xaxis_range, channels_region, filter = {}):
    """Handles the preprocessing and figure generation for the MEG signal visualization."""
    import time  # For logging execution times

    start_time = time.time()
    raw_ddf = pu.get_preprocessed_dataframe_dask(folder_path, freq_data, time_range[0], time_range[1])
    print(f"Step 1: Preprocessing completed in {time.time() - start_time:.2f} seconds.")

    # Filter time range
    filter_start_time = time.time()
    shifted_times = du.get_shifted_time_axis_dask(time_range, raw_ddf)
    print(f"Step 2: Time shifting completed in {time.time() - filter_start_time:.2f} seconds.")

    # Filter the dataframe based on the selected channels
    filter_df_start_time = time.time()
    print(raw_ddf.compute())
    filtered_raw_df = raw_ddf[selected_channels].compute()
    print(f"Step 3: Dataframe filtering completed in {time.time() - filter_df_start_time:.2f} seconds.")

    # Offset channel traces along the y-axis
    offset_start_time = time.time()
    channel_offset = calculate_channel_offset_std(filtered_raw_df, offset_selection)
    y_axis_ticks = get_y_axis_ticks_with_gap(selected_channels, channel_offset)
    shifted_filtered_raw_df = filtered_raw_df + np.tile(y_axis_ticks, (len(filtered_raw_df), 1))
    print(f"Step 4: Channel offset calculation completed in {time.time() - offset_start_time:.2f} seconds.")

    # Create a dictionary mapping channels to their colors
    if color_selection == "rainbow":
        region_to_color = {
            region: REGION_COLOR_PALETTE[i % len(REGION_COLOR_PALETTE)]
            for i, region in enumerate(channels_region.keys())
        }

        # Build channel-to-color mapping
        channel_to_color = {
            channel: region_to_color[region]
            for region, channels in channels_region.items()
            for channel in channels
        }

        # Final color map only for selected channels
        color_map = {
            channel: channel_to_color[channel]
            for channel in selected_channels
            if channel in channel_to_color
        }
    elif color_selection == "white":
        color_map = {channel: "white" for channel in selected_channels}
    elif "smoothGrad" in color_selection:
        color_map = {channel: "#00008B" for channel in selected_channels}

    # Use Plotly Express for efficient figure generation
    fig_start_time = time.time()
    shifted_filtered_raw_df["Time"] = shifted_times  # Add time as a column for Plotly Express

    fig = go.Figure()

    for col in shifted_filtered_raw_df.columns[:-1]:  # Exclude Time
        fig.add_trace(go.Scattergl(
            x=shifted_filtered_raw_df["Time"],
            y=shifted_filtered_raw_df[col],
            mode="lines",
            name=col,
            line=dict(color=color_map.get(col, None), width=1)
        ))

    if 'smoothGrad' in color_selection: 
        fig = su.add_smoothgrad_scatter(
            fig, 
            shifted_filtered_raw_df, 
            time_range, 
            selected_channels, 
            filter=filter
        )

    print(f"Step 5: Figure creation completed in {time.time() - fig_start_time:.2f} seconds.")

    layout_start_time = time.time()
    fig = apply_default_layout(fig, xaxis_range, time_range, None)
    print(f"Step 6: Layout update completed in {time.time() - layout_start_time:.2f} seconds.")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
    return fig