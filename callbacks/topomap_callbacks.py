import dash
from dash import Input, Output, State
import mne
from callbacks.utils import topomap_utils as tu
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