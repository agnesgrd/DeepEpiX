import dash
from dash import Input, Output, State, html
import mne
from callbacks.utils import topomap_utils as tu
import numpy as np


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
    
def register_close_topomap():
    # Callback to close the modal when the close button inside the modal is clicked
    @dash.callback(
        Output("topomap-modal", "is_open", allow_duplicate=True),
        [Input("close-topomap-modal", "n_clicks")],
        [State("topomap-modal", "is_open")],
        prevent_initial_call=True
    )
    def close_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open

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
        [Input("plot-topomap-button-range", "n_clicks")],
        [State("topomap-min-range", "value"),
        State("topomap-max-range", "value"),
        State("folder-store", "data"),
        State("frequency-store", "data") ,
        State("topomap-range-modal", "is_open")],
        prevent_initial_call=True
    )
    def plot_topomap(n_clicks, min_time, max_time, folder_path, freq_data, is_open):
        if n_clicks>0 and min_time is not None and max_time is not None:
            try:
                # Load raw data (metadata only)
                raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)

                step_size = max(1 / 100, (max_time-min_time)/20)  #freq_data.get("resample_freq")

                time_points = np.arange(float(min_time), float(max_time), step_size)

                # Generate images
                images = []
                for t in time_points:
                    img_str = tu.create_topomap(raw, t)  # Returns base64-encoded string
                    img_src = f"data:image/png;base64,{img_str}"
                    images.append(html.Img(src=img_src, style={'width': '200px', 'margin': '0px'}))

                # Add all images to modal content
                modal_content = html.Div(
                    children=images,
                    style={
                        'display': 'flex',
                        'flex-wrap': 'wrap',
                        'justify-content': 'center',
                        # 'position': 'fixed',
                        # 'top': '0',
                        # 'right': '0',  # Modal slides in
                        # 'width': '50%',
                        # 'height': '50%'
                    }
                )

                return True, modal_content, not is_open

            except Exception as e:
                print(f"Error in plot_topomap: {str(e)}")
                return None, None, is_open
            
        # If no button click or invalid input, return a placeholder image
        return None, None, is_open
    
def register_range_on_selection():   
    @dash.callback(
        [Output("topomap-min-range", "value"),
        Output("topomap-max-range", "value")],
        [Input("meg-signal-graph", "selectedData")]  # Capture selection data from the graph
    )
    def update_range_on_selection(selectedData):
        if selectedData:
            # Get the selected range (from selectedData)
            x_vals = [point['x'] for point in selectedData['points']]  # Extract the x values (time points)
            min_range = round(min(x_vals), 3)  # Get the minimum time value from the selection
            max_range = round(max(x_vals), 3)  # Get the maximum time value from the selection
            return min_range, max_range  # Update the min and max range for the topomap
        else:
            return dash.no_update, dash.no_update  # Default range if no selection has been made