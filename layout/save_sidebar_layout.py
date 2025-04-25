from dash import html, dcc
import dash_bootstrap_components as dbc
from layout import input_styles, box_styles, button_styles, label_styles
from callbacks.utils import folder_path_utils as fpu

def create_save():
    layout = html.Div([
        html.Div([
            # Input for annotation name
            html.Label("Select Annotations:", style={**label_styles["classic"]}),
            # Button container with flexbox to align buttons side by side
            html.Div([
                dbc.Button(
                    "Check All",
                    id="check-all-annotations-to-save-btn",
                    color="success",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style=button_styles["tiny"]
                ),
                dbc.Button(
                    "Clear All",
                    id="clear-all-annotations-to-save-btn",
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style=button_styles["tiny"]
                ),
            ], style={"display": "flex", "justifyContent": "space-between", "gap": "4%", "marginBottom": "6px"}),  # Align buttons side by side

            dcc.Checklist(
                id="annotations-to-save-checkboxes",
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
                dbc.Button("📂", id="open-folder-button", color="primary", className="me-2"), 
                dcc.Dropdown(
                    id="saving-folder-path-dropdown",
                    options=fpu.get_folder_path_options(),
                    placeholder="Select ...",
                )],style={"display": "block", "justifyContent": "space-between", "gap": "4%", "marginBottom": "10px"}
            ),

            

            # Dropdown for choosing folder
            html.Label("Enter Old MarkerFile.mrk Name:", style={**label_styles["classic"]}),
            dbc.Input(id="old-mrk-name", type="text", value="OldMarkerFile_test", style={**input_styles["small-number"]}),

            # Dropdown for choosing folder
            html.Label("Enter New MarkerFile.mrk Name:", style={**label_styles["classic"]}),
            dbc.Input(id="new-mrk-name", type="text", value="MarkerFile", style={**input_styles["small-number"]}),
                            
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

            dbc.Tooltip("By default, the old marker file is renamed to OldMarkerFile.mrk, and the new one is saved as MarkerFile.mrk in the subject folder, ensuring backward compatibility.", target="save-annotation-button", placement="left"),

            # Loading spinner wraps only the elements that require loading
            dcc.Loading(
                id="loading",
                type="default", 
                children=[
                    html.Div(id="saving-mrk-status", style={"margin-top": "10px"})
                ]),
        ])
            
    ])

    return layout