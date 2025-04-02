from dash import html, Input, Output, callback
import dash
import plotly.graph_objects as go



def register_update_history():
    @callback(
        Output("history-log", "children"),
        Input("history-store", "data"),
        prevent_initial_call=False
    )
    def update_history(history_data):
        # Check if history_data is a list
        if not isinstance(history_data, list) or len(history_data)==0:
            return ["No history available."]
        
        # Return each string in the list as a separate line
        return [html.Div(line, style={"whiteSpace": "pre-wrap"}) for line in history_data]
    
def register_clean_history():
    @callback(
        Output("history-store", "data", allow_duplicate=True),   # Clears the stored history data
        Input("clean-history-button", "n_clicks"),  # Triggered by button clicks
        prevent_initial_call=True  # Avoid triggering the callback on initial load
    )
    def clean_history(n_clicks):
        # Return empty values for both the log and the store
        if n_clicks > 0:
            return []
