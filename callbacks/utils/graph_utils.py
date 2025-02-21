import numpy as np
import static.constants as c
from sklearn.preprocessing import StandardScaler
from pages.home import get_preprocessed_dataframe
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB
from plotly.subplots import make_subplots



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

def get_y_axis_ticks(selected_channels, channel_offset = c.DEFAULT_Y_AXIS_OFFSET):
    
    channel_offset = (channel_offset if channel_offset != None else c.DEFAULT_Y_AXIS_OFFSET)
    y_axis_ticks = np.arange(len(selected_channels)) * channel_offset
    return y_axis_ticks

def normalize_dataframe_columns(df):
    scaler = StandardScaler()

    df_standardized = df.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)

    return df_standardized

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

def get_shifted_time_axis(time_range, raw_df):
    """
    Adjusts the time axis of a DataFrame to include an offset based on the given time range.
    
    Parameters:
        time_range (tuple): The (start_time, end_time) range to apply as an offset.
        raw_df (pd.DataFrame): A DataFrame with a time-based index.

    Returns:
        pd.Series: A series of adjusted times in seconds, with the offset applied.
    """
    # Calculate elapsed time in seconds from the first timestamp in the DataFrame
    times = (raw_df.index - raw_df.index[0])
    
    # Add the starting time of the given range as an offset
    adjusted_times = times + time_range[0]
    
    return adjusted_times

def generate_graph_time_channel(selected_channels, offset_selection, time_range, folder_path, freq_data, color_selection, sensitivity_analysis, xaxis_range):
    """Handles the preprocessing and figure generation for the MEG signal visualization."""
    import time  # For logging execution times

    start_time = time.time()
    # Preprocess data

    raw_df = get_preprocessed_dataframe(folder_path, freq_data, time_range[0], time_range[1])
    print(f"Step 1: Preprocessing completed in {time.time() - start_time:.2f} seconds.")

    # Filter time range
    filter_start_time = time.time()
    shifted_times = get_shifted_time_axis(time_range, raw_df)
    print(f"Step 2: Time shifting completed in {time.time() - filter_start_time:.2f} seconds.")

    # Filter the dataframe based on the selected channels
    filter_df_start_time = time.time()
    filtered_raw_df = raw_df[selected_channels]
    print(f"Step 4: Dataframe filtering completed in {time.time() - filter_df_start_time:.2f} seconds.")

    # Offset channel traces along the y-axis
    offset_start_time = time.time()
    channel_offset = calculate_channel_offset(len(selected_channels))*(10-offset_selection)*10 #/10 #/ 12
    y_axis_ticks = get_y_axis_ticks(selected_channels, channel_offset)
    shifted_filtered_raw_df = filtered_raw_df + np.tile(y_axis_ticks, (len(filtered_raw_df), 1))
    print(f"Step 5: Channel offset calculation completed in {time.time() - offset_start_time:.2f} seconds.")

    # Create a dictionary mapping channels to their colors
    if color_selection == "rainbow":
        color_map = {channel: c.CHANNEL_TO_COLOR[channel] for channel in selected_channels}
    elif color_selection == "unified":
        color_map = {channel: "#00008B" for channel in selected_channels}
    elif "smoothGrad" in color_selection:
        color_map = {channel: "#00008B" for channel in selected_channels}

    # Use Plotly Express for efficient figure generation
    fig_start_time = time.time()
    shifted_filtered_raw_df["Time"] = shifted_times  # Add time as a column for Plotly Express

    fig = px.line(
        shifted_filtered_raw_df,
        x="Time",
        y=shifted_filtered_raw_df.columns[:-1],  # Exclude the Time column from y
        labels={"value": "Value", "variable": "Channel", "Time": "Time (s)"},
        color_discrete_map=color_map
    )

    # Create a resampler-aware figure
    # Create the resampled figure
    # fig = FigureResampler(
    #     go.Figure(),
    #     default_downsampler=MinMaxLTTB(parallel=True),
    #     resampled_trace_prefix_suffix=("<b style='color:#2bb1d6'>[R]</b> ", ''),
    #     # show_mean_aggregation_size=False,
    #     # default_n_shown_samples=17000,
    #     create_overview=True,
    #     # Specify the subplot rows that will be used for the overview axis of each column
    #     overview_row_idxs=[1],
    #     # Additonal kwargs for the overview axis
    #     overview_kwargs={"height": 200},
    #     )

    # # Ensure contiguous NumPy arrays for performance
    # time_values = np.ascontiguousarray(shifted_filtered_raw_df["Time"].to_numpy())

    # for channel in selected_channels:
    #     y_values = np.ascontiguousarray(shifted_filtered_raw_df[channel].to_numpy())  # Fix here

    #     fig.add_trace(
    #         go.Scattergl(
    #             # x=time_values,
    #             # y=y_values,
    #             mode="lines",
    #             name=channel,
    #             line=dict(color=color_map[channel], width=1)
    #         ),
    #         hf_x=time_values,
    #         hf_y=y_values,  # Pass contiguous array

    #     )

    if 'smoothGrad' in color_selection:

        # Add scatter plot using px.scatter
        # Convert time range to integer indices
        time_range_indices = np.arange(round(time_range[0] * 150), round(time_range[1] * 150)+1).astype(int)
        channel_indices = np.where(np.isin(c.ALL_CH_NAMES, selected_channels))[0]
        print('time range indices',time_range_indices)
        print('channel indices', channel_indices)
        print(sensitivity_analysis.shape)
        filtered_sa_array = sensitivity_analysis[time_range_indices[:, None], channel_indices]
        print(filtered_sa_array.shape)
        scatter_df = shifted_filtered_raw_df.melt(id_vars=["Time"], var_name="Channel", value_name="Value")

        scatter_df["Color"] = filtered_sa_array.flatten('F')  # Flatten color array

        # # Add a break (NaN row) between each channel
        # scatter_df_sorted = scatter_df.sort_values(by=["Channel", "Time"])  # Ensure sorting within each channel

        # # Function to add a break row (NaN) after each channel
        # def add_break(x):
        #     break_row = pd.DataFrame([{col: None for col in x.columns}])  # Create a row of NaNs
        #     return pd.concat([x, break_row], ignore_index=True)

        # scatter_df_sorted = scatter_df_sorted.groupby("Channel", group_keys=True).apply(add_break).reset_index(drop=True)
        # Ensure time is numeric (in seconds)
        # scatter_df = scatter_df.sort_values(by='Time')  # Sort just in case

        # # Define new time range at 600 Hz (1/600 seconds step)
        # time_min = scatter_df['Time'].min()
        # time_max = scatter_df['Time'].max()
        # new_time_grid = np.arange(time_min, time_max, 1/600)  # Step size = 1/600 seconds
        # print(new_time_grid)
        # # Reindex DataFrame
        # scatter_df_interpolated = pd.DataFrame({'Time': new_time_grid})
        # scatter_df_interpolated = scatter_df_interpolated.merge(scatter_df, on='Time', how='left')

        # # Interpolate missing "color" values
        # scatter_df_interpolated['Color'] = scatter_df_interpolated['Color'].interpolate(method='linear')

        scatter_df_filtered = scatter_df[scatter_df["Color"] > 0]
        print(scatter_df_filtered)

        scatter_fig = px.scatter(
            scatter_df_filtered,
            x="Time",
            y="Value",
            color="Color",
            color_continuous_scale="Reds",
            labels={"value": "Value", "variable": "Channel", "Time": "Time (s)"},
            opacity=1
        )

        # Add scatter traces to the line plot
        fig.add_traces(scatter_fig.data)

        # fig.update_traces(mode='markers', marker_size = 3)  # Set your desired width

    print(f"Step 6: Figure creation completed in {time.time() - fig_start_time:.2f} seconds.")

    # Update layout with x-axis range and other customizations
    layout_start_time = time.time()
    fig.update_layout(
        autosize=True,
        xaxis=dict(
            title='Time (s)',
            range=xaxis_range,
            minallowed=time_range[0],
            maxallowed=time_range[1],
            fixedrange=False,
            rangeslider=dict(visible=True, thickness=0.02),
            showspikes=True,
            spikemode="across+marker",
            spikethickness = 1,
        ),
        yaxis=dict(
            title='Channels',
            showgrid=True,
            tickvals=y_axis_ticks,
            ticktext=[f'{selected_channels[i]}' for i in range(len(selected_channels))],
            ticklabelposition="outside right",
            side="right",
            automargin=True,
            spikethickness = 0
        ),
        title={
            'text': folder_path if folder_path else 'Select a folder path in Home Page',
            'x': 0.5,
            'font': {'size': 12},
            'automargin': True,
            'yref': 'paper',
        },
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode =  'select',
        selectdirection = 'h',
        hovermode = 'closest'
    )
    # Update the line width after creation


    fig.update_traces(line=dict(width=1))

    if "smoothGrad" in color_selection:
        fig.update_layout(           
            coloraxis_colorbar=dict(
            title=dict(text="SmoothGrad"),
            thicknessmode="pixels", thickness=10,
            lenmode="fraction", len=0.15,
            y=0,
            x=0.9,
            orientation="h",
            ticks="outside",
            dtick=1),
            coloraxis=dict(cmin=0, cmax=1)  # Set color range from 0 to 1
        )


        fig.update_traces(line=dict(width=1), marker=dict(size=3))


    

    print(f"Step 7: Layout update completed in {time.time() - layout_start_time:.2f} seconds.")

    # Total execution time
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

    return fig

def generate_small_graph_time_channel(selected_channels, time_range, folder_path, freq_data, time_points, page_selector, chunk_limits):
    """Handles the preprocessing and figure generation for the MEG signal visualization."""
    import time  # For logging execution times

    start_time = time.time()
    # Preprocess data
    chunk_start_time, chunk_end_time = chunk_limits[int(page_selector)]
    raw_df = get_preprocessed_dataframe(folder_path, freq_data, chunk_start_time, chunk_end_time)
    print(f"Step 1: Preprocessing completed in {time.time() - start_time:.2f} seconds.")

    # Filter time range
    filter_start_time = time.time()
    filtered_times, filtered_raw_df = get_raw_df_filtered_on_time(time_range, raw_df)
    print(f"Step 2: Time filtering completed in {time.time() - filter_start_time:.2f} seconds.")

    # Filter the dataframe based on the selected channels
    filter_df_start_time = time.time()
    filtered_raw_df = filtered_raw_df[selected_channels]
    print(f"Step 4: Dataframe filtering completed in {time.time() - filter_df_start_time:.2f} seconds.")

    # Offset channel traces along the y-axis
    offset_start_time = time.time()
    channel_offset = calculate_channel_offset(len(selected_channels)) / 12
    y_axis_ticks = get_y_axis_ticks(selected_channels, channel_offset)
    shifted_filtered_raw_df = filtered_raw_df + np.tile(y_axis_ticks, (len(filtered_raw_df), 1))
    print(f"Step 5: Channel offset calculation completed in {time.time() - offset_start_time:.2f} seconds.")
    # Create a dictionary mapping channels to their colors
    color_map = {channel: c.CHANNEL_TO_COLOR[channel] for channel in selected_channels}
    # Use Plotly Express for efficient figure generation
    fig_start_time = time.time()
    shifted_filtered_raw_df["Time"] = filtered_times  # Add time as a column for Plotly Express
    fig = px.line(
        shifted_filtered_raw_df,
        x="Time",
        y=shifted_filtered_raw_df.columns[:-1],  # Exclude the Time column from y
        labels={"value": "Value", "variable": "Channel", "Time": "Time (s)"},
        color_discrete_map=color_map
    )

    print(f"Step 6: Figure creation completed in {time.time() - fig_start_time:.2f} seconds.")

    # Update layout with x-axis range and other customizations
    layout_start_time = time.time()
    fig.update_layout(
        autosize=True,
        xaxis=dict(
            tickvals=time_points,
            ticks="outside", 
            tickwidth=2, 
            tickcolor='crimson', 
            ticklen=10
        ),
        yaxis=dict(
            showgrid=True,
            automargin=True
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    # Update the line width after creation
    for trace in fig.data:
        trace.update(line=dict(width=1))
    print(f"Step 7: Layout update completed in {time.time() - layout_start_time:.2f} seconds.")

    # Total execution time
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

    return fig


