import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import os

# Register the page
dash.register_page(__name__, path = "/")

layout = html.Div([
    html.H1("HOME: Choose MEG Data Folder"),

    # Explanation of what the user needs to do
    html.Div([
        html.P("Please enter the full path to the .ds folder containing the data to analyze."),
    ], style={"padding": "10px"}),

    html.Div([
        html.P("Example: /home/admin_mel/Code/DeepEpiX/data/berla_Epi-001_20100413_07.ds"),
    ], style={"padding": "10px"}),

    # Input field for folder path
    html.Div([
        dcc.Input(
            id="folder-path-input",
            type="text",
            placeholder="Enter folder path here...",
            style={
                "width": "100%",
                "padding": "10px",
                "fontSize": "16px",
                "borderWidth": "1px",
                "borderStyle": "solid",
                "borderRadius": "5px",
            }
        )
    ], style={"padding": "10px"}),

    # Display the entered folder path
    html.Div([
        html.H4("Entered Folder Path:"),
        html.Div(id="entered-folder", style={"font-style": "italic", "color": "#555"}),
    ], style={"padding": "10px"}),

    # Hidden store to keep the folder path
    # dcc.Store(id="folder-store"),

    # Button to load and proceed to the next page
    html.Div([
        dbc.Button(
            "Load and Proceed",
            id="load-analyze-button",
            color="success",
            disabled=True,
            href="/view"  # Link to the next page
        )
    ], style={"margin-top": "20px"})
])

@dash.callback(
    Output("entered-folder", "children"),
    Output("folder-store", "data"),
    Output("load-analyze-button", "disabled"),
    Input("folder-path-input", "value"),
    prevent_initial_call=True
)
def handle_entered_folder(folder_path):
    """Validate entered folder path for .ds"""
    if folder_path:
        # Simulate the processing of the entered folder path
        # Check if folder exists and finish by .ds
        if os.path.isdir(folder_path):            
            if folder_path.endswith(".ds"):
                return (
                    f"{folder_path} (valid)",
                    folder_path,
                    False  # Enable the load button
                )
            return (
                f"{folder_path} should end with .ds.",
                None,
                True  # Keep the load button disabled
            )
        else:
            return (
                f"{folder_path} does not exist. Please try again.",
                None,
                True  # Keep the load button disabled
            )
    return ("Invalid input. Please enter a valid folder path.", None, True)
