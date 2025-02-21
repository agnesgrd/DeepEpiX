from dash import html, dcc, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles, label_styles
from callbacks.utils import folder_path_utils as fpu
from callbacks.utils import markerfile_utils as mu

def create_save():
    layout = html.Div([
        html.Div([
            # Input for annotation name
            html.Label("Select Annotations:", style={**label_styles["classic"]}),
            # Button container with flexbox to align buttons side by side
            html.Div([
                dbc.Button(
                    "Check All",
                    id="check-all-annotations-btn",
                    color="success",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style={
                        "fontSize": "12px",
                        "padding": "6px 12px",
                        "borderRadius": "5px",
                        "width": "48%",  # Adjusted width
                    }
                ),
                dbc.Button(
                    "Clear All",
                    id="clear-all-annotations-btn",
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style={
                        "fontSize": "12px",
                        "padding": "6px 12px",
                        "borderRadius": "5px",
                        "width": "48%",  # Adjusted width
                    }
                ),
            ], style={"display": "flex", "justifyContent": "space-between", "gap": "4%", "marginBottom": "6px"}),  # Align buttons side by side

            dcc.Checklist(
                id="saving-annotation-checkboxes",
                inline=False,
                style={"margin": "10px 0", "fontSize": "12px"},
                persistence=True,
                persistence_type="local"
            ),
        ], style = box_styles["classic"]),

        html.Div([
            # Dropdown for choosing folder
            html.Label("Select Saving Folder:", style={**label_styles["classic"]}),
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dbc.Button("📂", id="open-folder-button", color="primary", className="me-2"),
                        width=4
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="saving-folder-path-dropdown",
                            options=fpu.get_folder_path_options(),
                            placeholder="Select ...",
                        ),
                        width=8
                    ),
                ])
            ])
        ], style = box_styles["classic"]),

        html.Div([
            # Dropdown for choosing folder
            html.Label("Enter Old MarkerFile.mrk Name:", style={**label_styles["classic"]}),
            dbc.Input(id="old-mrk-name", type="text", value="OldMarkerFile_test", style={**input_styles["small-number"]}),

        ], style = box_styles["classic"]),

        html.Div([
            # Dropdown for choosing folder
            html.Label("Enter New MarkerFile.mrk Name:", style={**label_styles["classic"]}),
            dbc.Input(id="new-mrk-name", type="text", value="MarkerFile", style={**input_styles["small-number"]}),

        ], style = box_styles["classic"]),

        html.Div([

            html.Label("By default, the old marker file is renamed to OldMarkerFile.mrk, and the new one is saved as MarkerFile.mrk in the subject folder, ensuring backward compatibility.", style={**label_styles["info"]}),
            
            # Run Prediction Button
            html.Div([
                dbc.Button(
                    "Save",
                    id="save-annotation-button",
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    disabled=False,
                    style=button_styles["big"]
                ),
            ]),

            # Loading spinner wraps only the elements that require loading
            dcc.Loading(
                id="loading",
                type="default", 
                children=[
                    html.Div(id="saving-mrk-status", style={"margin-top": "10px"})
                ]),

        ], style = box_styles["classic"]),

    ])

    return layout

@dash.callback(
    Output("saving-folder-path-dropdown", "value"),
    # Input("annotation-checkboxes", "options"),
    Input("folder-store", "data")
)
def enter_default_saving_folder(folder_path):
    if folder_path:
        return folder_path
    

# Callback function to save the annotation file
@dash.callback(
    Output("saving-mrk-status", "children"),  # Display a message in the saving status area
    Input("save-annotation-button", "n_clicks"),  # Trigger when the Save button is clicked
    State("saving-folder-path-dropdown", "value"),  # Get the selected folder path from the dropdown
    State("old-mrk-name", "value"),
    State("new-mrk-name", "value"),
    State("annotations-store", "data")  # Assuming annotations are stored somewhere
)
def save_annotation_file(n_clicks, folder_path, old_mrk_name, new_mrk_name, annotations):
    if n_clicks > 0:
        # Check if folder path and annotations are valid
        if not folder_path:
            return "Error: No folder path selected."
        if not annotations:
            return "Error: No annotations found."

        # Rename old marker file to OldMarkerFile.mrk
        mu.modify_name_oldmarkerfile(folder_path, old_mrk_name)

        # Save the new marker file
        try:
            mu.save_mrk_file(folder_path, old_mrk_name, new_mrk_name, annotations)
            return "File saved successfully!"
        except Exception as e:
            return f"Error saving the file: {str(e)}"
    return ""
    

