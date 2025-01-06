# from dash_extensions.enrich import Output, Input, State, Patch
import traceback
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash import Patch
from pages.home import get_preprocessed_dataframe
import static.constants as c
import callbacks.utils.graph_utils as gu
import callbacks.utils.annotation_utils as au
import numpy as np
import traceback
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB
import pandas as pd

def register_callbacks_annotation_names():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("annotation-checkboxes", "options"),
        Output("annotation-checkboxes", "value"),
        Input("annotations-store", "data"),
        prevent_initial_call = False
    )
    def display_annotation_names_checklist(annotations_store):
        annotation_names = au.get_annotation_descriptions(annotations_store)
        options = [{'label': name, 'value': name} for name in annotation_names]
        return options, annotation_names  # Set all annotations as default selected
    

# Modify your graph update function to take into account the time range from the scrollable X-axis
def generate_graph_time_channel(time_range, channel_region, annotations_to_show, folder_path, freq_data, annotations):
    """Handles the preprocessing and figure generation for the MEG signal visualization."""
    # Preprocess data
    raw_df = get_preprocessed_dataframe(folder_path, freq_data)

    # Filter time range
    filtered_times, filtered_raw_df = gu.get_raw_df_filtered_on_time([0,180], raw_df)

    # Get the selected channels based on region
    selected_channels = []
    for region_code in channel_region:
        if region_code in c.GROUP_CHANNELS_BY_REGION:
            selected_channels.extend(c.GROUP_CHANNELS_BY_REGION[region_code])

    # If no channels are selected, raise an error
    if not selected_channels:
        raise ValueError("No channels selected from the given regions.")
    
    # Filter the dataframe based on the selected channels
    filtered_raw_df = filtered_raw_df[selected_channels]

    # Offset channel traces along the y-axis
    channel_offset = gu.calculate_channel_offset(len(selected_channels))/12
    y_axis_ticks = gu.get_y_axis_ticks(selected_channels, channel_offset)
    shifted_filtered_raw_df = filtered_raw_df + np.tile(y_axis_ticks, (len(filtered_raw_df), 1))

    # Create the resampled figure
    fig = FigureResampler(
        go.Figure(),
        default_downsampler=MinMaxLTTB(parallel=True),
        resampled_trace_prefix_suffix=("<b style='color:#2bb1d6'>[R]</b> ", ''),
        show_mean_aggregation_size=False
    )

    # Add traces to the figure
    for i, channel_data in enumerate(filtered_raw_df):
        fig.add_trace(
            go.Scattergl(
                name=f"Channel {i}", 
                mode="lines", 
                line=dict(color=c.CHANNEL_TO_COLOR[channel_data], width=1),
                text=[
                    f"Time: {time} s<br>Value: {value}<br>Channel: {channel_data}"
                    for time, value in zip(filtered_times, filtered_raw_df[channel_data])
                ],  # Create text for each data point
                hovertemplate="%{text}<extra></extra>",  # Use the 'text' for hover display
                ),
            hf_x=filtered_times,
            hf_y=shifted_filtered_raw_df[channel_data],
            max_n_samples=27000,
        )

    # Update layout with x-axis range and other customizations
    fig.update_layout(
        autosize=True,
        xaxis=dict(
            title='Time (s)',
            range=time_range,  # Set dynamic time range based on the scrollable x-axis
            fixedrange=False,
            rangeslider=dict(
                visible=True,
                thickness=0.02,  # Narrow slider (default is 0.15)
            ),
            autorange = False
        ),
        yaxis=dict(
            title='Channels',
            showgrid=True,
            tickvals=y_axis_ticks,
            ticktext=[f'{selected_channels[i]}' for i in range(len(selected_channels))],
            ticklabelposition="outside right"
        ),
        title={
            'text': folder_path if folder_path else 'Select a folder path in Home Page',
            'x' : 0.5,
            'font': {'size': 12},
            'automargin': True,
            'yref': 'paper'
        },
        height=1000,
        showlegend=False
    )

    return fig

def register_update_graph_time_channel(): 
    @dash.callback(
        Output("meg-signal-graph", "figure"),
        Output("python-error", "children"),
        Input("channel-region-checkboxes", "value"),
        Input("annotation-checkboxes", "value"),
        Input("folder-store", "data"),
        State("frequency-store", "data"),
        State("annotations-store", "data"),
        prevent_initial_call=False
    )
    def update_graph_time_channel(channel_region, annotations_to_show, folder_path, freq_data, annotations):
        """Update MEG signal visualization based on time and channel selection."""
        try:
            if not channel_region or not folder_path or not freq_data:  # Check if data is missing
                return go.Figure(), "Missing data for graph rendering."
            time_range = [0,10]
            print("here")
            fig = generate_graph_time_channel(time_range, channel_region, annotations_to_show, folder_path, freq_data, annotations)

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
        Output("first-page-loading", "data"),
        Input("meg-signal-graph", "figure"),  # Current figure to update
        Input("annotation-checkboxes", "value"),  # Annotations to show based on the checklist
        State("annotations-store", "data"),
        State("first-page-loading", "data"),
        prevent_initial_call=True,
        supress_callback_exceptions=True
    )
    def update_annotations(fig_dict, annotations_to_show, annotations, first_load):
        """Update annotations visibility based on the checklist selection."""
        # Default time range in case the figure doesn't contain valid x-axis range data
        time_range = [0, 180]

        # ctx = dash.callback_context

        # if not ctx.triggered:
        #     return dash.no_update
        # triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        # if triggered_id == "meg-signal-graph":
        if first_load == 1:
            return None, 1
        
        else:
        
            print('updating annotations')
            # Check if fig_dict contains the x-axis range
            # if fig_dict and "layout" in fig_dict and "xaxis" in fig_dict["layout"]:
            #     xaxis_range = fig_dict["layout"]["xaxis"].get("range", None)
            #     if xaxis_range and len(xaxis_range) == 2:
            #         time_range = [xaxis_range[0], xaxis_range[1]]

            # Create a Patch for the figure
            fig_patch = Patch()

            # Check if fig_dict is None (i.e., if it is the initial empty figure)
            if fig_dict is None or 'layout' not in fig_dict or 'yaxis' not in fig_dict['layout']:
                # Set default y_min and y_max if the figure layout is not available
                y_min, y_max = 0, 1  # Set default range for the y-axis
            else:
                # Get the current y-axis range from the figure
                y_min, y_max = fig_dict['layout']['yaxis'].get('range', [0, 1])

            # Convert annotations to DataFrame
            annotations_df = pd.DataFrame(annotations).set_index("onset")

            # Filter annotations based on the current time range
            filtered_annotations_df = gu.get_annotations_df_filtered_on_time(time_range, annotations_df)

            # Prepare the shapes and annotations for the selected annotations
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

            # Update the figure with the new shapes and annotations
            fig_patch["layout"]["shapes"] = new_shapes
            fig_patch["layout"]["annotations"] = new_annotations
            # fig_patch["layout"]["xaxis"] = dict(
            #         rangeslider=dict(
            #         visible=True,
            #         thickness=0.02,  # Narrow slider (default is 0.15)
            #     ),
            #     )

            return fig_patch, 1

def register_manage_channels_checklist():
    @dash.callback(
        Output("channel-region-checkboxes", "value"),
        [Input("check-all-btn", "n_clicks"),
        Input("clear-all-btn", "n_clicks")],
    )
    def manage_checklist(check_all_clicks, clear_all_clicks):
        # Determine which button was clicked
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        all_regions = list(c.GROUP_CHANNELS_BY_REGION.keys())

        if triggered_id == "check-all-btn":
            return all_regions  # Select all regions
        elif triggered_id == "clear-all-btn":
            return []  # Clear all selections

        return dash.no_update
    
def register_move_time_slider():
    @dash.callback(
        Output("meg-signal-graph", "figure", allow_duplicate = True),
        Input("keyboard", "keydown"),
        State("meg-signal-graph", "figure"),
        prevent_initial_call=True,
        supress_callback_exceptions=True
    )
    def move_time_slider(keydown, fig):
        print(f'Key Pressed: {keydown}')

        # Get the current x-axis range
        xaxis_range = fig["layout"]["xaxis"]["range"]
        move_amount = 1/3*(xaxis_range[1]-xaxis_range[0])  # Number of seconds to move

        # Define the bounds for the x-axis (adjust based on your data)
        min_bound = 0
        max_bound = 180

        # Update the range based on the key press
        if keydown["key"] == "ArrowLeft":
            print("left")
            new_range = [xaxis_range[0] - move_amount, xaxis_range[1] - move_amount]
            if new_range[0] < min_bound:
                new_range = [min_bound, min_bound + move_amount]
        elif keydown["key"] == "ArrowRight":
            new_range = [xaxis_range[0] + move_amount, xaxis_range[1] + move_amount]
            print('right')
            if new_range[1] > max_bound:
                new_range = [max_bound - move_amount, max_bound]
        else:
            print("problem")
            return fig  # Return the figure unchanged if the key is not handled

        fig_patch = Patch()
        # Update the figure with the new x-axis range
        fig_patch["layout"]["xaxis"]["range"] = new_range

        return fig_patch





