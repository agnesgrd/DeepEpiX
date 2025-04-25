# --- Dash ---
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# --- Local Utilities ---
from callbacks.utils import folder_path_utils as fpu

# --- Layout Config ---
from layout import input_styles, box_styles, flexDirection

# --- Callbacks ---
from callbacks.storage_callbacks import register_populate_memory_tab_contents
from callbacks.folder_path_callbacks import (
    register_handle_valid_folder_path,
    register_update_dropdown,
    register_store_folder_path_and_clear_data,
    register_populate_tab_contents,
)
from callbacks.preprocessing_callbacks import (
    register_handle_frequency_parameters,
    register_preprocess_meg_data,
)
from callbacks.psd_callbacks import register_display_psd

dash.register_page(__name__, path = "/")

layout = html.Div([

    html.Div([
        html.H3([
            html.I(className="bi bi-sd-card", style={"marginRight": "10px", "fontSize": "1.2em"}),
            "Internal Storage"]),

        html.Div(
                id="subject-memory", 
                children = [

                    dbc.Tabs(id="subject-tabs-memory", active_tab='subject-tab-memory', children=[

                        dbc.Tab(label='General', tab_id="subject-tab-memory", children=[

                            html.Div(id="subject-container-memory", children = [html.Span("No subject in memory. Please choose one below.")])], style={"margin-top": "10px", "width": "40%", "maxHeight": "450px",  "overflowY": "auto"}),

                        dbc.Tab(label='Raw Info', tab_id="raw-info-tab-memory", children=[

                            html.Div(id="raw-info-container-memory", children = [html.Label("No subject in memory. Please choose one below.")])], style={"margin-top": "10px", "width": "40%", "maxHeight": "450px",  "overflowY": "auto"}),
                        
                        dbc.Tab(label='Event Statistics', tab_id="events-tab-memory", children=[

                            html.Div(id="event-stats-container-memory", children = [html.Label("No subject in memory. Please choose one below.")])], style={"margin-top": "10px", "width": "40%", "maxHeight": "450px",  "overflowY": "auto"}),

                        dbc.Tab(label='History', tab_id="history-tab-memory", children=[
                            
                            html.Div(id ="history-container-memory", children = [html.Label("No history in memory. Please do further analysis.")])], style={"margin-top": "10px", "width": "40%", "maxHeight": "450px",  "overflowY": "auto"}),

                    ], style={**flexDirection["row-tabs"], "width": "40%"}),

                ])
    ], style=box_styles["classic"]),

    html.Div([
        html.H4([
            html.I(className="bi bi-1-circle-fill", style={"marginRight": "10px", "fontSize": "1.2em"}),
            "Choose MEG Data Folder"]),

        html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Button("📂 Open Folder", id="open-folder-button", color="primary", className="me-2"),
                    width="auto"
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="folder-path-dropdown",
                        options=fpu.get_folder_path_options(),
                        placeholder="Select ...",
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Button(
                        "Load",
                        id="load-button",
                        color="success",
                        disabled=True,
                        n_clicks=0
                    )
                )
            ])
        ]),

        html.Div(id="folder-path-warning", className="text-danger", style={"marginTop": "10px"})

    ], style=box_styles["classic"]),

    html.Div(
        id="frequency-container",
        children=[

            html.Div(
                id="frequency-inputs",
                children=[
                    html.H4([
                        html.I(className="bi bi-2-circle-fill", style={"marginRight": "10px", "fontSize": "1.2em"}),
                         "Set Frequency Parameters for Signal Preprocessing"]),

                    html.Div([
                        html.Label("Resampling Frequency (Hz): "),
                        dbc.Input(id="resample-freq", type="number", value=150, step=1, min=50, style=input_styles["number-in-box"]),
                    ]),

                    html.Div([
                        html.Label("High-pass Frequency (Hz): "),
                        dbc.Input(id="high-pass-freq", type="number", value=0.5, step=0.1, min=0.1, style=input_styles["number-in-box"]),
                    ]),

                    html.Div([
                        html.Label("Low-pass Frequency (Hz): "),
                        dbc.Input(id="low-pass-freq", type="number", value=50, step=1, min=1, style=input_styles["number-in-box"]),
                    ]),

                    html.Div([
                        html.Label("Notch filter Frequency (Hz): "),
                        dbc.Input(id="notch-freq", type="number", value=50, step=1, min=0, style=input_styles["number-in-box"]),
                    ]),

                    html.H4([
                        html.I(className="bi bi-3-circle-fill", style={"marginRight": "10px", "fontSize": "1.2em"}),
                         "Give Hint on Channel for Heartbeat Detection"]),

                    html.Div([
                        html.Label("Channel Name:"),
                        dbc.Input(id="heartbeat-channel", type="text", placeholder="None", style=input_styles["number-in-box"]),
                        dbc.Tooltip(
                                """
                                ch_name : None | str\n

                                - The name of the channel to use for MNE ECG peak detection. 
                                - If None (default), ECG channel is used if present. 
                                - If None and no ECG channel is present, a synthetic ECG channel is created from the cross-channel average. 
                                - This synthetic channel can only be created from MEG channels.
                                """,
                                target="heartbeat-channel",
                                placement="right",
                                class_name="custom-tooltip"
                            ),
                    ]),

                    html.Div([
                        dbc.Button("Preprocess & Display", id="preprocess-display-button", color="success", disabled=True, n_clicks=0),
                        dcc.Loading(
                            id="loading",
                            type="default",
                            children=[html.Div(id="preprocess-status")]
                        ),
                        dcc.Location(id="url", refresh=True),
                    ], style={"margin-top": "20px"})

                ], style={**box_styles["classic"], "width": "40%"}),

            html.Div(
                id="analysis", 
                children = [

                    dbc.Tabs(id="tabs", active_tab='raw-info-tab', children=[

                            dbc.Tab(label='Raw Info', tab_id="raw-info-tab", children=[

                                html.Div(id="raw-info-container"),
                            ]),

                            dbc.Tab(label='Event Statistics', tab_id="events-tab", children=[

                                html.Div(id="event-stats-container", style = {"width":"70%"}),
                            ]),

                            dbc.Tab(label='Power Spectral Density', tab_id="psd-tab", children=[

                                html.Div(id="psd-container", children=[

                                    dbc.Button("Compute & Display", id="compute-display-psd-button", color="success", n_clicks=0, style = {"marginTop": "15px"}),

                                    dcc.Loading(id="loading", type="default", children=[
                                        
                                        html.Div(id="psd-status", style={"margin-top": "10px"})
                                    ]),    
                                ]),
                            ]),

                    ], style=flexDirection["row-tabs"]),

                ], style={**box_styles["classic"], "width": "60%"}),

        ], style={"display": "none"})
    ])

register_populate_memory_tab_contents()

register_update_dropdown()

register_handle_valid_folder_path()

register_store_folder_path_and_clear_data()

register_populate_tab_contents()

register_handle_frequency_parameters()

register_display_psd()

register_preprocess_meg_data()