import dash
from dash import Input, Output

# Callback to display the stored folder path
def register_callbacks_folder_path():
    @dash.callback(
        Output("display-folder-path", "children"),
        Input("folder-store", "data"),  # Access the stored data
        prevent_initial_call=False
    )
    def display_folder_path(folder_path):
        """Display the stored folder path."""
        return f"{folder_path}" if folder_path else "No folder path has been selected."