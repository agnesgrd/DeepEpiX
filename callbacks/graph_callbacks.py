# from dash_extensions.enrich import Output, Input, State, Patch
import traceback
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash import Patch
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


def generate_graph_time_channel(time_range, channel_range, annotations_to_show, folder_path, freq_data, annotations):
    """Handles the preprocessing and figure generation for the MEG signal visualization."""
    # Preprocess data
    raw_df = get_preprocessed_dataframe(folder_path, freq_data)

    # Filter time range
    filtered_times, filtered_raw_df = gu.get_raw_df_filtered_on_time(time_range, raw_df)

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

def register_update_graph_time_channel():  
    @dash.callback(
        Output("meg-signal-graph", "figure"),
        Output("python-error", "children"),
        Input("time-slider", "value"),  # Time range selection
        Input("channel-slider", "value"),  # Channel range selection as state
        State("annotation-checkboxes", "value"),
        State("folder-store", "data"),
        State("frequency-store", "data"),
        State("annotations-store", "data"),
        prevent_initial_call=False
        )
    def update_graph_time_channel(time_range, channel_range, annotations_to_show, folder_path, freq_data, annotations):
        """Update MEG signal visualization based on time and channel selection."""
        try:
            fig = generate_graph_time_channel(time_range, channel_range, annotations_to_show, folder_path, freq_data, annotations)

            return fig, None
        except FileNotFoundError:
            return go.Figure(), f"Error: Folder not found."
        except ValueError as ve:
            return go.Figure(), f"Error: {str(ve)}.\n Details: {traceback.format_exc()}"
        except Exception as e:
            return go.Figure(), f"Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}"
        
def register_update_annotations():
    @dash.callback(
        Output("meg-signal-graph", "figure", allow_duplicate=True),
        Input("annotation-checkboxes", "value"),  # Annotations to show based on the checklist
        State("meg-signal-graph", "figure"),  # Current figure to update
        State("time-slider", "value"),
        State("annotations-store", "data"),
        prevent_initial_call=True
    )
    def update_annotations(annotations_to_show, fig_dict, time_range, annotations):
        """Update annotations visibility based on the checklist selection."""
        # Create a Patch for the figure
        fig_patch = Patch()

        # Get the current y-axis range from the figure
        y_min, y_max = (
            fig_dict.get("layout", {}).get("yaxis", {}).get("range", [0, 1])
        )

        # Convert annotations to DataFrame
        annotations_df = pd.DataFrame(annotations).set_index("onset")

        filtered_annotations_df = gu.get_annotations_df_filtered_on_time(time_range, annotations_df)

        # Filter and update shapes
        # Prepare the shapes for the selected annotations
        new_shapes = []
        new_annotations = []
        for _, row in filtered_annotations_df.iterrows():
            description = row["description"]
            if description in annotations_to_show:
                new_shapes.append(
                    dict(
                        type="line",
                        x0=row.name,
                        x1=row.name,
                        y0=y_min,
                        y1=y_max,
                        xref="x",
                        yref="y",
                        line=dict(color="red", width=2, dash="dot"),
                        opacity=0.25
                        )
                    )
                # Add the label in the margin
                new_annotations.append(
                    dict(
                        x=row.name,
                        y=1.05,  # Slightly above the graph in the margin
                        xref="x",
                        yref="paper",  # Use paper coordinates for the y-axis (margins)
                        text=description,  # Annotation text
                        showarrow=False,  # No arrow needed
                        font=dict(size=10, color="red"),  # Customize font
                        align="center",
                    )
                )

        # Update only the shapes in the figure using Patch
        fig_patch["layout"]["shapes"] = new_shapes
        fig_patch["layout"]["annotations"] = new_annotations

        return fig_patch
    
# def register_update_annotations():
#     @dash.callback(
#         Output("meg-signal-graph", "figure", allow_duplicate=True),
#         Input("annotation-checkboxes", "value"),  # Annotations to show based on the checklist
#         State("meg-signal-graph", "figure"),  # Current figure to update
#         State("time-slider", "value"),
#         State("annotations-store", "data"),
#         prevent_initial_call=True
#     )
#     def update_annotations(annotations_to_show, fig_dict, time_range, annotations):
#         """Update annotations visibility based on the checklist selection."""
#         try:
#             fig_patch = generate_annotations(annotations_to_show, fig_dict, time_range, annotations)
#         except Exception as e:
#             pass
#         return fig_patch
    
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


