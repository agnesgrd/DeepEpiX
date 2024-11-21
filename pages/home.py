import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import os

# This is where the app will get the context from
# No need to instantiate app here, Dash handles the pages via the page registration system
dash.register_page(__name__)

layout = html.Div([
    html.H1("HOME: Select MEG Data Folder"),

    # Explanation of what the user needs to do
    html.Div([
        html.P("Please select a folder containing .ds data files."),
    ], style={"padding": "10px"}),

    # Button to choose folder (using file selection dialog)
    dbc.Button("Choose Folder", id="choose-folder-button", color="primary", style={"margin-bottom": "20px"}),

    # Display the selected folder path
    html.Div([
        html.H4("Selected Folder:"),
        html.Div(id="folder-path", style={"font-style": "italic", "color": "#555"}),
    ], style={"padding": "10px"}),

    # Hidden div to store the path of the folder for further processing
    dcc.Store(id="folder-store"),

    # Option to validate and load the folder contents (for demonstration)
    html.Div([
        dbc.Button("Load and Analyze", id="load-analyze-button", color="success", disabled=True),
        html.Div(id="loading-message", style={"margin-top": "20px", "font-size": "16px"}),
    ])
])

# Callback to handle the folder selection
@dash.callback(
    [Output("folder-path", "children"),
     Output("folder-store", "data"),
     Output("load-analyze-button", "disabled")],
    [Input("choose-folder-button", "n_clicks")],
    prevent_initial_call=True
)
def choose_folder(n_clicks):
    """ Function to handle the folder selection event (simulated) """
    
    # Simulate selecting a folder
    selected_folder = "/path/to/your/data/"  # Adjust this path
    
    if os.path.isdir(selected_folder):
        folder_files = os.listdir(selected_folder)
        ds_files = [f for f in folder_files if f.endswith(".ds")]
        if ds_files:
            folder_message = f"Folder selected: {selected_folder}"
            is_disabled = False
        else:
            folder_message = "No .ds files found in this folder."
            is_disabled = True
    else:
        folder_message = "Invalid folder path."
        is_disabled = True

    return folder_message, selected_folder, is_disabled
