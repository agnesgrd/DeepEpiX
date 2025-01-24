import dash
from dash import Input, Output, State, html
import mne
from callbacks.utils import topomap_utils as tu
import numpy as np
from callbacks.utils import history_utils as hu
import static.constants as c
import time
from pages.home import get_preprocessed_dataframe
import plotly.graph_objects as go
from dash import dcc


# Callback to handle the plotting of the topomap
def register_display_topomap():
    @dash.callback(
        Output("topomap-img", "src"),
        Output("topomap-modal", "is_open"),
        [Input("plot-topomap-button", "n_clicks")],
        [State("topomap-timepoint", "value"),
        State("folder-store", "data"),
        State("frequency-store", "data") ,
        State("topomap-modal", "is_open")],
        prevent_initial_call=True
    )
    def plot_topomap(n_clicks, timepoint, folder_path,freq_data, is_open):
        if n_clicks>0 and timepoint is not None:
            try:
                # Load raw data (metadata only)
                raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)
                
                img_str = tu.create_topomap(raw, timepoint)
                return f"data:image/png;base64,{img_str}", not is_open
                
            except Exception as e:
                print(f"Error in plot_topomap: {str(e)}")
                return "https://via.placeholder.com/150", is_open
            
        # If no button click or invalid input, return a placeholder image
        return "https://via.placeholder.com/150", is_open
    

def register_enable_topomap_button():
    @dash.callback(
        Output("plot-topomap-button-range", "disabled"),
        Input("topomap-min-range", "value"),
        Input("topomap-max-range", "value")
    )
    def enable_topomap_button(min_range, max_range):
        if min_range is not None and max_range is not None:
            return False  # Enable the button if both inputs are provided
        return True  # Disable the button if either input is missing
     
def register_display_topomap_video():
    @dash.callback(
        Output("topomap-result", "children"),
        Output("topomap-modal-content", "children"),
        Output("topomap-range-modal", "is_open"),
        Output("history-store", "data", allow_duplicate=True),
        [Input("plot-topomap-button-range", "n_clicks")],
        [State("topomap-min-range", "value"),
        State("topomap-max-range", "value"),
        State("channel-region-checkboxes", "value"),
        State("folder-store", "data"),
        State("frequency-store", "data") ,
        State("topomap-range-modal", "is_open")],
        State("history-store", "data"),
        prevent_initial_call=True
    )
    def plot_topomap(n_clicks, min_time, max_time, channel_region, folder_path, freq_data, is_open, history_data):
        if n_clicks>0 and min_time is not None and max_time is not None:
            try:

                # Load raw data (metadata only)
                raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)
                raw.pick_types(meg=True, ref_meg=False)


                # Generate signal image (selected signal visualization)
                # signal_img_str = tu.create_signal_plot(raw, min_time, max_time)  # Returns base64-encoded string
                # signal_img_src = f"data:image/png;base64,{signal_img_str}"
                # signal_image = html.Img(src=signal_img_src, style={
                #     'width': '100%',  # Full width
                #     'height': '100%',  # Consistent height
                #     'marginBottom': '10px',  # Space between rows
                # })
                step_size = max(1 / 100, (max_time-min_time)/3)  #freq_data.get("resample_freq")
                time_points = np.arange(float(min_time), float(max_time)+step_size, step_size)

                signal_graph = tu.create_small_graph_time_channel(min_time, max_time, folder_path, freq_data, time_points)

                # Generate images
                topomap_images = []

                for t in time_points:
                    img_str = tu.create_topomap(raw, t)  # Returns base64-encoded string
                    img_src = f"data:image/png;base64,{img_str}"
                    topomap_images.append(html.Img(src=img_src, style={
                        'height': '180px',  # Consistent height
                        'margin': '0 5px',  # Horizontal spacing between images
                    }))

                # Combine signal image and topomap images
                modal_content = html.Div(
                    children=[
                        html.Div(
                            children = dcc.Graph(figure=signal_graph), 
                            style={"textAlign": "center"}),  # Row 1: Signal
                        html.Div(  # Row 2: Topomap images
                            children=topomap_images, style={
                                "display": "flex",  # Horizontal alignment
                                "justifyContent": "center", # Center horizontally
                                "overflowX": "auto",  # Enable horizontal scrolling
                            } 
                        )
                    ],
                )
                
                history_data = hu.fill_history_data(history_data, f"Plotted topomap on [{min_time}, {max_time}].\n")

                return True, modal_content, not is_open, history_data

            except Exception as e:
                print(f"Error in plot_topomap: {str(e)}")
                return None, None, is_open, dash.no_update
            
        # If no button click or invalid input, return a placeholder image
        return None, None, is_open, dash.no_update
    
def register_range_on_selection():   
    @dash.callback(
        [Output("topomap-min-range", "value"),
        Output("topomap-max-range", "value")],
        [Input("meg-signal-graph", "selectedData")]  # Capture selection data from the graph
    )
    def update_range_on_selection(selected_data):
        if selected_data:
            
            # Get the selected range (from selectedData)
            try:
                x_range = selected_data.get('range', {}).get('x')
                if x_range and len(x_range) == 2:  # Validate that x_range exists and has two elements
                    min_range = round(x_range[0], 3)  # Get the minimum time value from the selection
                    max_range = round(x_range[1], 3)  # Get the maximum time value from the selection
                    return min_range, max_range  # Update the min and max range for the topomap
                else:
                    # Handle case where x_range is invalid
                    print("x_range is missing or invalid:", x_range)
                    return dash.no_update, dash.no_update
            except Exception as e:
                # Log any unexpected exceptions
                print(f"Error processing selected data: {e}")
                return dash.no_update, dash.no_update
        else:
            return dash.no_update, dash.no_update  # Default range if no selection has been made
        