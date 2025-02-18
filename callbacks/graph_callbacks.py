# from dash_extensions.enrich import Output, Input, State, Patch
import traceback
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash import Patch
import static.constants as c
import callbacks.utils.graph_utils as gu
import callbacks.utils.sensitivity_analysis_utils as sau
import traceback
import plotly.graph_objects as go
import pandas as pd
import itertools
   
    
def register_callbacks_sensivity_analysis():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("colors-radio", "options"),
        Output("colors-radio", "value"),
        Input("sensitivity-analysis-store", "data"),
        State("colors-radio", "options"),
        State("colors-radio", "value"),
        prevent_initial_call=False
    )
    def display_sensitivity_analysis_checklist(sa_store, default_options, value):
        # Create options for the checklist from the channels in montage_store
        options = [{'label': key, 'value': key} for key in sa_store.keys()]
        updated_options = default_options + options

        # If value is valid, keep the current selection
        return updated_options, dash.no_update
            
   
def register_update_graph_time_channel(): 
    @dash.callback(
        Output("meg-signal-graph", "figure"),
        Output("python-error", "children"),
        Input("page-selector", "value"),
        Input("montage-radio", "value"),
        Input("channel-region-checkboxes", "value"),
        Input("folder-store", "data"),
        Input("offset-display", "children"),
        Input("colors-radio", "value"),
        State("chunk-limits-store", "data"),
        State("frequency-store", "data"),
        State("montage-store", "data"),
        State("meg-signal-graph", "figure"),
        State("sensitivity-analysis-store", "data"),
        prevent_initial_call=False
    )
    def update_graph_time_channel(page_selection, montage_selection, channel_selection, folder_path, offset_selection, color_selection, chunk_limits,freq_data, montage_store, graph, sensitivity_analysis):
        """Update MEG signal visualization based on time and channel selection."""

        time_range = chunk_limits[int(page_selection)]

        # Get the current x-axis center
        xaxis_range = graph["layout"]["xaxis"].get("range", [])
        if xaxis_range[1] < time_range[0] or xaxis_range[0] > time_range[1]:
            xaxis_range = [time_range[0], time_range[0]+10]


        # Reading back
        if color_selection == "smoothGrad across channels":
            sensitivity_analysis = sau.deserialize_array(sensitivity_analysis['smoothGrad across channels'][0], sensitivity_analysis['smoothGrad across channels'][1])
        elif color_selection == "smoothGrad across time":
            sensitivity_analysis = sau.deserialize_array(sensitivity_analysis['smoothGrad across time'][0], sensitivity_analysis['smoothGrad across time'][1])

        try:
            if montage_selection == "channel selection" and not channel_selection or not folder_path or not freq_data:  # Check if data is missing
                return go.Figure(), "Missing data for graph rendering."
            
            else:
                if montage_selection == "channel selection":
                    # Get the selected channels based on region
                    selected_channels = [
                        channel
                        for region_code in channel_selection
                        if region_code in c.GROUP_CHANNELS_BY_REGION
                        for channel in c.GROUP_CHANNELS_BY_REGION[region_code]
                    ]

                    if not selected_channels:
                        raise ValueError("No channels selected from the given regions.")

                else: 

                    # If montage selection is not "channel selection", use montage's corresponding channels
                    selected_channels = montage_store.get(montage_selection, [])
                    
                    # If there are no channels for the selected montage
                    if not selected_channels:
                        raise ValueError(f"No channels available for the selected montage: {montage_selection}")
                
                    if offset_selection is None:
                        offset_selection = 5
                        
                fig = gu.generate_graph_time_channel(selected_channels, float(offset_selection), time_range, folder_path, freq_data, color_selection, sensitivity_analysis, xaxis_range)

                return fig, None
            
        except FileNotFoundError:
            return go.Figure(), f"Error: Folder not found."
        except ValueError as ve:
            return go.Figure(), f"Error: {str(ve)}.\n Details: {traceback.format_exc()}"
        except Exception as e:
            return go.Figure(), f"Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}"
            
def register_move_time_slider():
    @dash.callback(
        Output("meg-signal-graph", "figure", allow_duplicate = True),
        Input("keyboard", "keydown"),
        State("page-selector", "value"),
        State("chunk-limits-store", "data"),
        State("meg-signal-graph", "figure"),
        prevent_initial_call=True,
        supress_callback_exceptions=True
    )
    def move_time_slider(keydown, page_selection, chunk_limits, fig):

        # Get the current x-axis range
        xaxis_range = fig["layout"]["xaxis"]["range"]
        move_amount = 1/3*(xaxis_range[1]-xaxis_range[0])  # Number of seconds to move

        # Define the bounds for the x-axis (adjust based on your data)
        time_range = chunk_limits[int(page_selection)]
        min_bound = time_range[0]
        max_bound = time_range[1]

        # Update the range based on the key press
        if keydown["key"] == "ArrowLeft":
            new_range = [xaxis_range[0] - move_amount, xaxis_range[1] - move_amount]
            if new_range[0] < min_bound:
                new_range = [min_bound, min_bound + move_amount]
        elif keydown["key"] == "ArrowRight":
            new_range = [xaxis_range[0] + move_amount, xaxis_range[1] + move_amount]
            if new_range[1] > max_bound:
                new_range = [max_bound - move_amount, max_bound]
        else:
            return fig  # Return the figure unchanged if the key is not handled

        fig_patch = Patch()
        # Update the figure with the new x-axis range
        fig_patch["layout"]["xaxis"]["range"] = new_range

        return fig_patch
    
def register_move_to_spike():
    @dash.callback(
        Output("meg-signal-graph", "figure", allow_duplicate=True),
        Input("prev-spike", "n_clicks"),
        Input("next-spike", "n_clicks"),
        State("meg-signal-graph", "figure"),
        State("annotation-checkboxes", "value"),
        State("annotations-store", "data"),
        State("page-selector", "value"),
        State("chunk-limits-store", "data"),
        prevent_initial_call = True
    )
    def move_to_spike(prev_spike, next_spike, graph, annotations_to_show, annotations_data, page_selection, chunk_limits):

        if not annotations_data or not annotations_to_show:
            return dash.no_update  # No annotations available, return the same graph
        
        if len(annotations_to_show)==0 or len(annotations_data)==0:
            return dash.no_update

        # Extract x-coordinates (onset times) of spikes from annotations
        spike_x_positions = [
            ann["onset"] for ann in annotations_data if ann["description"] in annotations_to_show
        ]
        spike_x_positions = sorted(spike_x_positions)  # Ensure sorted order

        if not spike_x_positions:
            return graph  # No spikes to navigate

        # Get the current x-axis center
        xaxis_range = graph["layout"]["xaxis"].get("range", [])
        if xaxis_range:
            current_x_center = sum(xaxis_range) / 2  # Midpoint of current view
        else:
            current_x_center = spike_x_positions[0]  # Default to first spike

        # Determine next or previous spike
        if dash.ctx.triggered_id == "next-spike":
            next_spike_x = next((x for x in spike_x_positions if x > current_x_center), spike_x_positions[-1])
        elif dash.ctx.triggered_id == "prev-spike":
            next_spike_x = next((x for x in reversed(spike_x_positions) if x < current_x_center), spike_x_positions[0])
        else:
            return graph  # No valid button click
        
        time_range_limits = chunk_limits[int(page_selection)]
        
        # Extract time range limits
        time_range_min, time_range_max = time_range_limits

        # Compute x-axis range offset
        x_range_offset = (xaxis_range[1] - xaxis_range[0]) / 2 if xaxis_range else 10

        # Default centered range
        proposed_x_min = next_spike_x - x_range_offset
        proposed_x_max = next_spike_x + x_range_offset

        # Adjust if near the edges
        if proposed_x_min < time_range_min:
            x_min, x_max = time_range_min, time_range_min + 2 * x_range_offset
        elif proposed_x_max > time_range_max:
            x_min, x_max = time_range_max - 2 * x_range_offset, time_range_max
        else:
            x_min, x_max = proposed_x_min, proposed_x_max

        # Ensure the adjusted range is within valid limits
        x_min = max(x_min, time_range_min)
        x_max = min(x_max, time_range_max)

        # Update the graph layout
        graph["layout"]["xaxis"]["range"] = [x_min, x_max]

        return graph