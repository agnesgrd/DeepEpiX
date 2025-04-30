import numpy as np
import config
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import dataframe_utils as du
from callbacks.utils import smoothgrad_utils as su
import plotly.express as px
import plotly.graph_objects as go
from layout import DEFAULT_FIG_LAYOUT

def calculate_channel_offset(num_channels, plot_height=900, min_gap=30):
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

def get_y_axis_ticks(selected_channels, channel_offset = config.DEFAULT_Y_AXIS_OFFSET):
    
    channel_offset = (channel_offset if channel_offset != None else config.DEFAULT_Y_AXIS_OFFSET)
    y_axis_ticks = np.arange(len(selected_channels)) * channel_offset
    return y_axis_ticks

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
    
    # Apply dynamic values to layout
    layout['xaxis']['range'] = xaxis_range
    layout['xaxis']['minallowed'] = time_range[0]
    layout['xaxis']['maxallowed'] = time_range[1]
    # layout['title']['text'] = folder_path if folder_path else 'Select a folder path'
    
    # Update the layout of the figure
    fig.update_layout(layout)
    
    return fig

def generate_graph_time_channel(selected_channels, offset_selection, time_range, folder_path, freq_data, color_selection, xaxis_range, filter = {}):
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
    filtered_raw_df = raw_ddf[selected_channels].compute()
    print(f"Step 3: Dataframe filtering completed in {time.time() - filter_df_start_time:.2f} seconds.")

    # Offset channel traces along the y-axis
    offset_start_time = time.time()
    channel_offset = calculate_channel_offset(len(selected_channels))*(11-offset_selection)*9
    y_axis_ticks = get_y_axis_ticks(selected_channels, channel_offset)
    shifted_filtered_raw_df = filtered_raw_df + np.tile(y_axis_ticks, (len(filtered_raw_df), 1))
    print(f"Step 4: Channel offset calculation completed in {time.time() - offset_start_time:.2f} seconds.")

    # Create a dictionary mapping channels to their colors
    if color_selection == "rainbow":
        # Create a reverse mapping to quickly find the region of a channel
        def map_channel_to_color():
            channel_to_region = {}
            for region, channels in config.GROUP_CHANNELS_BY_REGION.items():
                for channel in channels:
                    channel_to_region[channel] = region

            channel_to_color = {}
            for channels, region in channel_to_region.items():
                channel_to_color[channels] = config.REGION_COLORS[region]
            return channel_to_color

        CHANNEL_TO_COLOR = map_channel_to_color()

        color_map = {channel: CHANNEL_TO_COLOR[channel] for channel in selected_channels}
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

    # fig.update_layout(
    #     xaxis_title="Time (s)",
    #     yaxis_title="Value",
    #     showlegend=True
    # )
    # fig = px.line(
    #     shifted_filtered_raw_df,
    #     x="Time",
    #     y=shifted_filtered_raw_df.columns[:-1],  # Exclude the Time column from y
    #     labels={"value": "Value", "variable": "Channel", "Time": "Time (s)"},
    #     color_discrete_map=color_map
    # )

    if 'smoothGrad' in color_selection: 
        fig = su.add_smoothgrad_scatter(
            fig, 
            shifted_filtered_raw_df, 
            time_range, 
            selected_channels, 
            filter=filter
        )

    print(f"Step 5: Figure creation completed in {time.time() - fig_start_time:.2f} seconds.")

    # Update layout with x-axis range and other customizations
    layout_start_time = time.time()
    fig = apply_default_layout(fig, xaxis_range, time_range, None)

    print(f"Step 6: Layout update completed in {time.time() - layout_start_time:.2f} seconds.")

    # Total execution time
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

    return fig