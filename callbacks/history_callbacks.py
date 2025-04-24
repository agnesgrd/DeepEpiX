from dash import Input, Output, callback, html
from callbacks.utils import history_utils as hu
import dash_bootstrap_components as dbc

def register_update_history():
    @callback(
        Output("history-log", "children"),
        Input("url", "pathname"),
        Input("history-store", "data"),
        prevent_initial_call=False
    )
    def update_history(pathname, history_data):
        category='annotations'
        return html.Div([
            dbc.ListGroup([
                dbc.ListGroupItem(entry) for entry in hu.read_history_data_by_category(history_data, category)]) 
                if hu.read_history_data_by_category(history_data, category) 
                else html.P("No entries yet.")
        ])
    
def register_clean_history():
    @callback(
        Output("history-store", "clear_data", allow_duplicate=True),   # Clears the stored history data
        Input("clean-history-button", "n_clicks"),  # Triggered by button clicks
        prevent_initial_call=True  # Avoid triggering the callback on initial load
    )
    def clean_history(n_clicks):
        # Return empty values for both the log and the store
        if n_clicks > 0:
            return True
