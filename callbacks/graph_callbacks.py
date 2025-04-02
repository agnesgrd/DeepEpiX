import traceback
import plotly.graph_objects as go
import dash
from dash import Input, Output, State, Patch, callback
import static.constants as c
import callbacks.utils.graph_utils as gu
import traceback
import plotly.graph_objects as go
import pickle
   


def register_update_graph_time_channel(): 
    @callback(
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
        State("anomaly-detection-store", "data"),
        prevent_initial_call=False
    )
    def update_graph_time_channel(page_selection, montage_selection, channel_selection, folder_path, offset_selection, color_selection, chunk_limits,freq_data, montage_store, graph, sensitivity_analysis_store, anom_detect_store):
        """Update MEG signal visualization based on time and channel selection."""
        
        if folder_path is None:
            return dash.no_update, "Please choose a subject to display in Home page."

        # if not chunk_limits:
        #     return dash.no_update, "Please choose a subject to display in Home page."
        time_range = chunk_limits[int(page_selection)]

        # Get the current x-axis center
        xaxis_range = graph["layout"]["xaxis"].get("range", [])
        if xaxis_range[1] < time_range[0] or xaxis_range[0] > time_range[1]:
            xaxis_range = [time_range[0], time_range[0]+10]


        # Reading back
        if "smoothGrad" in color_selection:
            with open(sensitivity_analysis_store['smoothGrad'], 'rb') as f:
                filter = pickle.load(f)

        elif "anomDetect" in color_selection:
            with open(anom_detect_store['anomDetect'], 'rb') as f:
                filter = pickle.load(f)
        
        else:
            filter = {}

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
                        
                fig = gu.generate_graph_time_channel(selected_channels, float(offset_selection), time_range, folder_path, freq_data, color_selection, xaxis_range, filter)

                return fig, None
            
        except FileNotFoundError:
            return dash.no_update, f"Error: Folder not found."
        except ValueError as ve:
            return dash.no_update, f"Error: {str(ve)}.\n Details: {traceback.format_exc()}"
        except Exception as e:
            return dash.no_update, f"Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}"
            
def register_move_time_slider():
    @callback(
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
    
