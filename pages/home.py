# Dash & Plotly
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# External Libraries
import os
import mne
import numpy as np

# Local Imports
from layout import input_styles, box_styles, flexDirection
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import folder_path_utils as fpu
from callbacks.utils import annotation_utils as au
from callbacks.utils import history_utils as hu
import config

# Register the page
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

                            html.Div(id="subject-container-memory", children = [html.Span("No subject in memory. Please choose one below.")])], style={"margin-top": "10px", "width": "40%"}),

                        dbc.Tab(label='Raw Info', tab_id="raw-info-tab-memory", children=[

                            html.Div(id="raw-info-container-memory", children = [html.Label("No subject in memory. Please choose one below.")])], style={"margin-top": "10px", "width": "40%"}),
                        
                        dbc.Tab(label='Event Statistics', tab_id="events-tab-memory", children=[

                            html.Div(id="event-stats-container-memory", children = [html.Label("No subject in memory. Please choose one below.")])], style={"margin-top": "10px", "width": "40%"}),

                        dbc.Tab(label='History', tab_id="history-tab-memory", children=[
                            
                            html.Div(id ="history-container-memory", children = [html.Label("No history in memory. Please do further analysis.")])], style={"margin-top": "10px", "width": "40%", "height": "500px",  "overflowY": "auto"}),

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

                        ],
                        style=flexDirection["row-tabs"]
                    ),

                ], style={**box_styles["classic"], "width": "60%"}),
            ],

            style={**flexDirection["row-flex"], "display": "none"}
        )
    ])

@callback(
    Output("subject-container-memory", "children"),
    Output("raw-info-container-memory", "children"),
    Output("event-stats-container-memory", "children"),
    Output("history-container-memory", "children"),
    Input("url", "pathname"),
    Input("subject-tabs-memory", "active_tab"),
    State("folder-store", "data"),
    State("chunk-limits-store", "data"),
    State("frequency-store", "data"),
    State("annotations-store", "data"),
    State("history-store", "data"),
    prevent_initial_call=False
)
def populate_memory_tab_contents(pathname, selected_tab, folder_path, chunk_limits, freq_data, annotations_data, history_data):
    """Populate memory tab content based on selected tab and stored folder path."""
    if not folder_path or not chunk_limits or not freq_data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    subject_content = dash.no_update
    raw_info_content = dash.no_update
    event_stats_content = dash.no_update
    history_content = dash.no_update

    if selected_tab == "subject-tab-memory":
            subject_content = dbc.Card(
                dbc.CardBody([
                    html.H5([html.I(className="bi bi-person-rolodex", style={"marginRight": "10px", "fontSize": "1.2em"}), "Subject"], className="card-title"),
                    html.Hr(),
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Strong(folder_path)
                        ]),
                    ]),

                    html.H5([html.I(className="bi bi-sliders", style={"marginRight": "10px", "fontSize": "1.2em"}), "Frequency Parameters"], className="card-title"),
                    html.Hr(),
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Strong("Resample Frequency: "),
                            html.Span(f"{freq_data.get('resample_freq', 'N/A')} Hz")
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Low-pass Filter: "),
                            html.Span(f"{freq_data.get('low_pass_freq', 'N/A')} Hz")
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("High-pass Filter: "),
                            html.Span(f"{freq_data.get('high_pass_freq', 'N/A')} Hz")
                        ]),
                        dbc.ListGroupItem([
                            html.Strong("Notch Filter: "),
                            html.Span(f"{freq_data.get('notch_freq', 'N/A')} Hz")
                        ]),
                    ])
                ])
            )

    if selected_tab == "raw-info-tab-memory":
        raw_info_content = fpu.build_table_raw_info(folder_path)

    if selected_tab == "events-tab-memory":
        event_stats_content = au.build_table_events_statistics(annotations_data)

    if selected_tab == "history-tab-memory":

        icon_map = {
            "annotations": "bi-activity",
            "models": "bi-stars",
            "ica": "bi-noise-reduction"
        }

        history_content = dbc.Card(
                dbc.CardBody([
                    html.Div([
                        html.H5([
                            html.I(className=f"bi {icon_map[category]}", style={"marginRight": "10px", "fontSize": "1.2em"}),
                            category.capitalize()
                        ], className="card-title"),
                        html.Hr(),
                        dbc.ListGroup([
                            dbc.ListGroupItem(entry)
                            for entry in hu.read_history_data_by_category(history_data, category)
                        ]) if hu.read_history_data_by_category(history_data, category) else
                        html.P("No entries yet.", className="text-muted")
                    ], style={"marginBottom": "10px"})

                    for category in ['annotations', 'models', 'ica']
                    ]
                )
            )


    return subject_content, raw_info_content, event_stats_content, history_content


@callback(
    Output("folder-path-dropdown", "options"),
    Output("folder-path-dropdown", "value"),
    Input("open-folder-button", "n_clicks"),
    State("folder-path-dropdown", "options"),
    prevent_initial_call=True
)
def update_dropdown(n_clicks, folder_path_list):
    """Update the dropdown menu after a user selects files from the file explorer."""
    if n_clicks > 0:
        folder_path = fpu.browse_folder()
        if folder_path:
            if fpu.test_ds_folder(folder_path):
                folder_path_list.append({"label": fpu.get_ds_folder(folder_path), "value": folder_path})
                return folder_path_list, folder_path
    return dash.no_update, dash.no_update

@callback(
    Output("load-button", "disabled"),
    Input("folder-path-dropdown", "value"),
    prevent_initial_call=True
)
def handle_valid_folder_path(folder_path):
    """Validate entered folder path : Check if folder exists and finish by .ds, then make "load" button clickable."""
    if folder_path:
        if os.path.isdir(folder_path):            
            if folder_path.endswith(".ds"):
                return False
    return dash.no_update

@callback(
    Output("frequency-container", "style"),
    Output("folder-store", "data"),
    Output("sensitivity-analysis-store", "clear_data"),
    Output("chunk-limits-store", "clear_data"),
    Output("frequency-store", "clear_data"),
    Output("annotations-store", "clear_data"),
    Output("anomaly-detection-store", "clear_data"),
    Input("load-button", "n_clicks"),
    State("folder-path-dropdown", "value"),
    prevent_initial_call=True
)
def store_folder_path_and_clear_data(n_clicks, folder_path):
    """Clear all stores and display frequency section on load."""
    if not folder_path:
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)
    return (
        {**flexDirection["row-flex"], "display": "flex"},
        folder_path, 
        True, True, True, True, True
    )

@callback(
    Output("raw-info-container", "children"),
    Output("event-stats-container", "children"),
    Input("tabs", "active_tab"),
    Input("folder-store", "data"),
    prevent_initial_call=False
)
def populate_tab_contents(selected_tab, folder_path):
    """Populate tab content based on selected tab and stored folder path."""
    if not folder_path:
        return dash.no_update, dash.no_update

    raw_info_content = dash.no_update
    event_stats_content = dash.no_update

    if selected_tab == "raw-info-tab":
        raw_info_content = fpu.build_table_raw_info(folder_path)

    if selected_tab == "events-tab":
        event_stats_content = fpu.build_table_events_statistics(folder_path)

    return raw_info_content, event_stats_content

@callback(
    Output("preprocess-status", "children"),
    Input("resample-freq", "value"),
    Input("high-pass-freq", "value"),
    Input("low-pass-freq", "value"),
    Input("notch-freq", "value"),
    prevent_initial_call=True
)
def handle_frequency_parameters(resample_freq, high_pass_freq, low_pass_freq, notch_freq):
    """Retrieve frequency parameters and store them."""

    if not low_pass_freq or not high_pass_freq or not notch_freq:
        return f"⚠️ Please fill in all frequency parameters."
    
    elif high_pass_freq >= low_pass_freq:
        return f"⚠️ High-pass frequency must be less than low-pass frequency."

    return dash.no_update
  
@callback(
    Output("psd-status", "children"),
    Input("compute-display-psd-button", "n_clicks"),
    State("folder-store", "data"),
    State("frequency-store", "data"),
    running=[
        (Output("compute-display-psd-button", "disabled"), True, False)],
    prevent_initial_call=True
)
def display_psd(n_clicks, folder_path, freq_data):
    """ Compute and display power spectrum decomposition depending on the frequency parameters stored."""

    if folder_path is None or freq_data is None:
        return "Please fill in all frequency parameters."
    
    if n_clicks > 0:
        return pu.compute_power_spectrum_decomposition(folder_path, freq_data)

@callback(
    Output("preprocess-status", "children", allow_duplicate=True),
    Output("frequency-store", "data"),
    Output("annotations-store", "data"),
    Output("chunk-limits-store", "data"),
    Output("url", "pathname"),
    Input("preprocess-display-button", "n_clicks"),
    State("folder-store", "data"),
    State("resample-freq", "value"),
    State("high-pass-freq", "value"),
    State("low-pass-freq", "value"),
    State("notch-freq", "value"),
    State("heartbeat-channel", "value"),
    running=[
        (Output("preprocess-display-button", "disabled"), True, False),
        (Output("load-button", "disabled"), True, False),
        (Output("compute-display-psd-button", "disabled"), True, False)],
    prevent_initial_call=True
)
def preprocess_meg_data(n_clicks, folder_path, resample_freq, high_pass_freq, low_pass_freq, notch_freq, heartbeat_ch_name):
    """Preprocess MEG data and save it, store annotations and chunk limits in memory."""

    if n_clicks > 0:

        try:
            raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)

            annotations_dict, max_length = au.get_annotations_dataframe(raw, heartbeat_ch_name)
            chunk_limits = pu.update_chunk_limits(max_length)

            # Store the frequency values when the folder is valid
            freq_data = {
                "resample_freq": resample_freq,
                "low_pass_freq": low_pass_freq,
                "high_pass_freq": high_pass_freq,
                "notch_freq": notch_freq
            }

            # Apply filtering and resampling
            raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
            raw.notch_filter(freqs=notch_freq)
            raw.resample(resample_freq)

            for chunk_idx in chunk_limits:
                start_time, end_time = chunk_idx
                raw_df = pu.get_preprocessed_dataframe(folder_path, freq_data, start_time, end_time, raw)
            return "Preprocessed and saved data", freq_data, annotations_dict, chunk_limits, "/viz/raw-signal"
        
        except Exception as e:
            return f"Error during preprocessing : {str(e)}", dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update