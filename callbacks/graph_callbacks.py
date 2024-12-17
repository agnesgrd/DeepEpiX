from dash_extensions.enrich import Serverside, Output, Input, State, callback, clientside_callback, ctx, no_update
import traceback
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from pages.home import get_preprocessed_dataframe
import static.constants as c
import callbacks.utils.graph_utils as gu
import numpy as np
import traceback
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB
from pages.home import get_preprocessed_dataframe
import pandas as pd


def generate_meg_signal_figure(time_range, channel_range, annotations_to_show, folder_path, freq_data, annotations):
    """Handles the preprocessing and figure generation for the MEG signal visualization."""
    # Preprocess data
    raw_df = get_preprocessed_dataframe(folder_path, freq_data)

    # Convert annotations_data back to a DataFrame
    annotations_df = pd.DataFrame(annotations).set_index("onset")
    print(annotations_df)

    # Filter time range
    filtered_times, filtered_raw_df, filtered_annotations_df = gu.get_filtered_df(time_range, raw_df, annotations_df)
    print(filtered_annotations_df)

    # Filter channel range
    start_channel, end_channel = channel_range
    selected_channels = c.ALL_CH_NAMES[start_channel:end_channel + 1]
    filtered_raw_df = filtered_raw_df[selected_channels]

    filtered_raw_df = gu.normalize_dataframe_columns(filtered_raw_df)

    # Offset channel traces along the y-axis
    channel_offset = gu.calculate_channel_offset(len(selected_channels))/6
    y_axis_ticks = gu.get_y_axis_ticks(selected_channels, channel_offset)
    filtered_raw_df += np.tile(y_axis_ticks, (len(filtered_raw_df), 1))

    # Create the resampled figure
    fig = FigureResampler(
        go.Figure(),
        default_downsampler=MinMaxLTTB(parallel=True),
        resampled_trace_prefix_suffix=("<b style='color:#2bb1d6'>[R]</b> ", ''),
        show_mean_aggregation_size=False
    )

    # Add traces to the figure
    for i, channel_data in enumerate(filtered_raw_df):
        # For each channel, add its trace to the figure
        fig.add_trace(
            go.Scattergl(name=f"Channel {i}", mode="lines"),  # Scattergl ensures fast rendering
            hf_x=filtered_times,
            hf_y=filtered_raw_df[channel_data],  # High-frequency y data
            max_n_samples=600, # Adjust this for the maximum resolution you want per trace,
)
        
    fig.update_traces(line = {"width":0.8})

    # # Customize layout
    # fig.update_layout(
    #     autosize=True,
    #     xaxis=dict(title='Time (s)'),
    #     yaxis=dict(
    #         title='Channels',
    #         showgrid=True,
    #         tickvals=y_axis_ticks,  # Align ticks with channels
    #         ticktext=[f'{selected_channels[i]}' for i in range(len(selected_channels))],  # Label each channel
    #         ticklabelposition="outside right"
    #     ),
    #     title='MEG Signal Visualization',
    #     height=600,
    #     showlegend=True
    # )

    # Get the min and max y-values of the filtered data
    y_min = filtered_raw_df.min().min()
    y_max = filtered_raw_df.max().max()

    # Add vertical lines for annotations
    for _, row in filtered_annotations_df.iterrows():
        onset = row.name
        description = row['description']
        color = c.ANNOTATIONS_COLORS.get(description, 'black')  # Default to black if description not found
        if description in annotations_to_show:
            fig.add_shape(
                showlegend = True,
                type='line',
                x0=onset,
                x1=onset,
                y0=y_min,
                y1=y_max,
                xref='x',
                yref='y',  # Use the y-axis coordinate system
                line=dict(
                    color=color,
                    width=2,
                    dash='dot'
                ),
                label=dict(
                    text=description,  # Use description as the label text
                    textposition="end",
                    xanchor = "center",
                    yanchor = "top",
                    font=dict(size=12, color=color)
                ),
                opacity=0.25,
                legendgroup=description
            )

    # Update the layout with the legend and any other customizations
    fig.update_layout(
        autosize=True,
        xaxis=dict(title='Time (s)'),
        yaxis=dict(
            title='Channels',
            showgrid=True,
            tickvals=y_axis_ticks,  # Align ticks with channels
            ticktext=[f'{selected_channels[i]}' for i in range(len(selected_channels))],  # Label each channel
            ticklabelposition="outside right"
        ),
        title='MEG Signal Visualization',
        height=600,
        showlegend=False
        )

    return fig  # Return the figure to be rendered in the graph

def register_update_meg_signal():  
    @dash.callback(
        Output("meg-signal-graph", "figure"),
        Output("python-error", "children"),
        Input("time-slider", "value"),  # Time range selection
        Input("channel-slider", "value"),  # Channel range selection as state
        Input("annotation-checkboxes", "value"),
        State("folder-store", "data"),
        State("frequency-store", "data"),
        State("annotations-store", "data"),
        prevent_initial_call=False
        )
    def update_meg_signal_time(time_range, channel_range, annotations_to_show, folder_path, freq_data, annotations):
        """Update MEG signal visualization based on time and channel selection."""
        try:
            fig = generate_meg_signal_figure(time_range, channel_range, annotations_to_show, folder_path, freq_data, annotations)

            return fig, None
        except FileNotFoundError:
            return go.Figure(), f"Error: Folder not found."
        except ValueError as ve:
            return go.Figure(), f"Error: {str(ve)}.\n Details: {traceback.format_exc()}"
        except Exception as e:
            return go.Figure(), f"Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}"
        
def register_move_time_slider():

    @dash.callback(
    Output("time-slider", "value"),
    Input("time-left", "n_clicks"),
    Input("time-right", "n_clicks"),
    State("time-slider", "value"),
    prevent_initial_call=True
    )
    def shift_time_range(left_clicks, right_clicks, current_range):
        """Shift the time range left or right."""

        # Define shift step 
        start, end = current_range
        shift_step = end - start

        # Determine how much to shift based on button clicks
        shift = (right_clicks - left_clicks) * shift_step

        # Update the time range0
        if shift > 0:
            new_range = [shift, shift + shift_step]
        else:
            new_range = [0, shift_step]

        # Ensure the range stays within bounds
        new_range = [
            max(0, new_range[0]),  # Prevent start from going below 0
            min(180, new_range[1])  # Prevent end from exceeding max time (100 in this case)
        ]

        return new_range


