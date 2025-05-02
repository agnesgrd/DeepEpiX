from dash import html, dcc
import dash_bootstrap_components as dbc

from layout import INPUT_STYLES, BOX_STYLES, BUTTON_STYLES, LABEL_STYLES
from callbacks.utils import folder_path_utils as fpu

def create_save():
    layout = html.Div([
        html.Div([
            html.Label("Select Annotations:", style={**LABEL_STYLES["classic"]}),

            html.Div([
                dbc.Button(
                    "Check All",
                    id="check-all-annotations-to-save-btn",
                    color="success",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style=BUTTON_STYLES["tiny"]
                ),
                dbc.Button(
                    "Clear All",
                    id="clear-all-annotations-to-save-btn",
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style=BUTTON_STYLES["tiny"]
                ),
            ], style={"display": "flex", "justifyContent": "space-between", "gap": "4%", "marginBottom": "6px"}),  # Align buttons side by side

            dcc.Checklist(
                id="annotations-to-save-checkboxes",
                inline=False,
                style={"margin": "10px 0", "fontSize": "12px"},
                persistence=True,
                persistence_type="local"
            ),
        ], style = BOX_STYLES["classic"]),

        html.Div([
            html.Label("Select Saving Folder:", style={**LABEL_STYLES["classic"]}),

            html.Div([
                dbc.Button("📂", id="open-folder-button", color="primary", className="me-2"), 
                dcc.Dropdown(
                    id="saving-folder-path-dropdown",
                    options=fpu.get_folder_path_options(),
                    placeholder="Select ...",
                )],style={"display": "block", "justifyContent": "space-between", "gap": "4%", "marginBottom": "10px"}
            ),

            html.Label("Enter Old MarkerFile.mrk Name:", style={**LABEL_STYLES["classic"]}),
            dbc.Input(id="old-mrk-name", type="text", value="OldMarkerFile", style={**INPUT_STYLES["small-number"]}),

            html.Label("Enter New MarkerFile.mrk Name:", style={**LABEL_STYLES["classic"]}),
            dbc.Input(id="new-mrk-name", type="text", value="MarkerFile", style={**INPUT_STYLES["small-number"]}),
                            
            dbc.Button(
                "Save",
                id="save-annotation-button",
                color="warning",
                outline=True,
                size="sm",
                n_clicks=0,
                disabled=False,
                style=BUTTON_STYLES["big"]
            ),

            dbc.Tooltip("By default, the old marker file is renamed to OldMarkerFile.mrk, and the new one is saved as MarkerFile.mrk in the subject folder, ensuring backward compatibility.", target="save-annotation-button", placement="left"),

            dcc.Loading(
                id="loading",
                type="default", 
                children=[
                    html.Div(id="saving-mrk-status", style={"margin-top": "10px"})
                ]),
        ])
            
    ])

    return layout