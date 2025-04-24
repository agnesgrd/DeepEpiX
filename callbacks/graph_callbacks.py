import traceback
import plotly.graph_objects as go
import dash
from dash import Input, Output, State, Patch, callback, ClientsideFunction, clientside_callback
import config
import callbacks.utils.graph_utils as gu
import traceback
import plotly.graph_objects as go
import pickle
   


def register_update_graph_time_channel(): 
    @callback(
        Output("meg-signal-graph", "figure"),
        Output("python-error", "children"),
        # Output("loading-graph", "children"),
        Input("update-button", "n_clicks"),  # Trigger the callback with the button
        Input("page-selector", "value"),
        State("montage-radio", "value"),
        State("channel-region-checkboxes", "value"),
        State("folder-store", "data"),
        State("offset-display", "children"),
        State("colors-radio", "value"),
        State("chunk-limits-store", "data"),
        State("frequency-store", "data"),
        State("montage-store", "data"),
        State("meg-signal-graph", "figure"),
        State("sensitivity-analysis-store", "data"),
        State("anomaly-detection-store", "data"),
        running=[(Output("update-button", "disabled"), True, False)],
        prevent_initial_call=False
    )
    def update_graph_time_channel(n_clicks, page_selection, montage_selection, channel_selection, folder_path, offset_selection, color_selection, chunk_limits,freq_data, montage_store, graph, sensitivity_analysis_store, anom_detect_store):
        """Update MEG signal visualization based on time and channel selection."""
        # if graph and 'data' in graph and graph['data']:  # if there's already data in the figure
        #     return graph, None
        print(chunk_limits)
        print(page_selection)
        
        if n_clicks == 0:
            return dash.no_update, dash.no_update
        
        if not folder_path:
            return dash.no_update, "Please choose a subject to display on Home page."
        
        if None in (page_selection, offset_selection, color_selection, freq_data) or not chunk_limits:
            return dash.no_update, "You have a subject in memory but its recording has not been preprocessed yet. Please go back on Home page to reprocess the signal."
        
        if (montage_selection == "channel selection" and not channel_selection):  # Check if data is missing
                return dash.no_update, "Missing channel selection for graph rendering."
        
        if montage_selection == "channel selection":
            # Get the selected channels based on region
            selected_channels = [
                channel
                for region_code in channel_selection
                if region_code in config.GROUP_CHANNELS_BY_REGION
                for channel in config.GROUP_CHANNELS_BY_REGION[region_code]
            ]

            if not selected_channels:
                return dash.no_update, "No channels selected from the given regions"
            
        # If montage selection is not "channel selection", use montage's corresponding channels
        elif montage_selection != "montage selection":
            selected_channels = montage_store.get(montage_selection, [])

            if not selected_channels:
                return dash.no_update, f"No channels available for the selected montage: {montage_selection}"

        time_range = chunk_limits[int(page_selection)]

        # Get the current x-axis center
        xaxis_range = graph["layout"]["xaxis"].get("range", [])
        print(xaxis_range)
        print(time_range)
        if xaxis_range[1] <= time_range[0] or xaxis_range[0] >= time_range[1]:
            xaxis_range = [time_range[0], time_range[0]+10]

        print(xaxis_range)

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
            fig = gu.generate_graph_time_channel(selected_channels, float(offset_selection), time_range, folder_path, freq_data, color_selection, xaxis_range, filter)

            return fig, None
            
        except FileNotFoundError:
            return dash.no_update, f"Error: Folder not found."
        except ValueError as ve:
            return dash.no_update, f"Error: {str(ve)}.\n Details: {traceback.format_exc()}"
        except Exception as e:
            return dash.no_update, f"Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}"