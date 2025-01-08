import dash
from dash import Input, Output, State

# Callback to open the modal when the "Plot Topomap" button is clicked
def register_display_topomap():
    @dash.callback(
        Output("topomap-modal", "is_open"),
        [Input("plot-topomap-btn-1", "n_clicks")],
        [State("topomap-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_modal(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

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