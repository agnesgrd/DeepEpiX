# Dash & Plotly
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# External Libraries
import os
import mne
import numpy as np
from collections import Counter

# Local Imports
from layout import input_styles, box_styles, flexDirection
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import folder_path_utils as fpu
from callbacks.utils import annotation_utils as au

# Config and Exceptions
import config
from dash.exceptions import PreventUpdate

# Register the page
dash.register_page(__name__, path = "/")

layout = html.Div([

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

                                    # dcc.Graph(id="psd-graph", style={"display": "none"}),

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
    Output("folder-store", "data"),
    Output("load-button", "disabled"),
    Input("folder-path-dropdown", "value"),
    prevent_initial_call=True
)
def handle_valid_folder_path(folder_path):
    """Validate entered folder path : Check if folder exists and finish by .ds, then make "load" button clickable."""
    if folder_path:
        if os.path.isdir(folder_path):            
            if folder_path.endswith(".ds"):
                return folder_path, False
    return dash.no_update, dash.no_update

@callback(
    Output("raw-info-container", "children"),
    Input("load-button", "n_clicks"),
    State("folder-store", "data"),
    prevent_initial_call=True
)
def populate_raw_info(n_clicks, folder_path):
    """Fill table with info contained by raw object."""
    if not folder_path:
        return dash.no_update

    raw = mne.io.read_raw_ctf(folder_path, preload=False, verbose=False)
    info = raw.info

    data = [
        {"Property": "File name", "Value": raw.filenames[0] if raw.filenames else "Unknown"},
        {"Property": "Number of channels", "Value": info['nchan']},
        {"Property": "Sampling frequency (Hz)", "Value": info['sfreq']},
        {"Property": "Highpass filter", "Value": info['highpass']},
        {"Property": "Lowpass filter", "Value": info['lowpass']},
        {"Property": "Duration (s)", "Value": round(raw.times[-1], 2)},
        {"Property": "Channel names (preview)", "Value": ', '.join(info['ch_names'][:5]) + "..." if len(info['ch_names']) > 5 else ', '.join(info['ch_names'])},
        {"Property": "Bad channels", "Value": ', '.join(info['bads']) if info['bads'] else "None"},
        {"Property": "Measurement date", "Value": str(info['meas_date'])},
        {"Property": "Experimenter", "Value": info.get('experimenter', 'Unknown')},
        {"Property": "Comps (SSP/ICA)", "Value": f"{len(info.get('comps', []))} components"},
        {"Property": "Projections (SSP)", "Value": f"{len(info.get('projs', []))} projections"},
        {"Property": "Digitized points", "Value": f"{len(info.get('dig', []))} points" if info.get('dig') else "None"},
        {"Property": "CTF Head Transform", "Value": "Available" if info.get('ctf_head_t') else "None"},
        {"Property": "Device to Head Transform", "Value": "Available" if info.get('dev_head_t') else "None"},
    ]

    info_table_header = html.Thead(html.Tr([html.Th("Property"), html.Th("Value")]))
    info_table_body = html.Tbody([
        html.Tr([html.Td(row["Property"]), html.Td(str(row["Value"]))]) for row in data
    ])
    info_table = dbc.Table(
        [info_table_header, info_table_body],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
    )

    return html.Div([
        info_table
    ])

@callback(
    Output("event-stats-container", "children"),
    Input("tabs", "active_tab"),
    State("folder-store", "data"),
    prevent_initial_call=True
)
def populate_events_statistics(selected_tab, folder_path):
    """Fill table with statistics on annotations."""
    if selected_tab != "events-tab" or not folder_path:
        return dash.no_update

    raw = mne.io.read_raw_ctf(folder_path, preload=False, verbose=False)
    annotations = raw.annotations

    if len(annotations) == 0:
        return html.P("No annotations found in this recording.")

    # Count annotation descriptions
    description_counts = Counter(annotations.description)

    # Build a stats table
    table_header = [html.Thead(html.Tr([html.Th("Event Name"), html.Th("Count")]))]
    table_body = [
        html.Tr([html.Td(desc), html.Td(count)]) for desc, count in description_counts.items()
    ]
    annotation_table = dbc.Table(table_header + [html.Tbody(table_body)], bordered=True, striped=True, hover=True, size="sm")

    # Show total number and a few more stats
    stats_summary = html.Ul([
        html.Li(f"Total annotations: {len(annotations)}"),
        html.Li(f"Unique event types: {len(description_counts)}"),
        html.Li(f"First event starts at {annotations.onset[0]:.2f} s"),
        html.Li(f"Last event ends at {(annotations.onset[-1] + annotations.duration[-1]):.2f} s"),
    ])

    return html.Div([
        annotation_table,
        html.Hr(),
        html.H5("Event Summary"),
        stats_summary,
    ])

@callback(
    [Output("frequency-container", "style"),
    Output("sensitivity-analysis-store", "clear_data")],
    Input("load-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_load_button(n_clicks):
    """Display frequency parameters when button load is clicked"""
    if n_clicks > 0:
        # Clean cache or temporary files
        #TODO
        return {**flexDirection["row-flex"], "display": "flex"}, True
    return dash.no_update, dash.no_update

@callback(
    Output("preprocess-status", "children"),
    Output("frequency-store", "data"),
    Input("resample-freq", "value"),
    Input("high-pass-freq", "value"),
    Input("low-pass-freq", "value"),
    Input("notch-freq", "value"),
    prevent_initial_call=True
)
def handle_frequency_parameters(resample_freq, high_pass_freq, low_pass_freq, notch_freq):
    """Retrieve frequency parameters and store them."""

    if not low_pass_freq or not high_pass_freq or not notch_freq:
        return f"⚠️ Please fill in all frequency parameters.", dash.no_update
    
    elif high_pass_freq >= low_pass_freq:
        error = "High-pass frequency must be less than low-pass frequency."
        return error, dash.no_update
    else:
        # Store the frequency values when the folder is valid
        frequency_values = {
            "resample_freq": resample_freq,
            "low_pass_freq": low_pass_freq,
            "high_pass_freq": high_pass_freq,
            "notch_freq": notch_freq
        }
        return None, frequency_values
  
@callback(
    Output("psd-status", "children"),
    # Output("psd-graph", "figure"),
    # Output("psd-graph", "style"),
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
    
        raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)

        resample_freq = freq_data.get("resample_freq")
        low_pass_freq = freq_data.get("low_pass_freq")
        high_pass_freq = freq_data.get("high_pass_freq")
        notch_freq = freq_data.get("notch_freq")

        if not low_pass_freq or not high_pass_freq or not notch_freq:
            return dash.no_update
        
        raw.notch_filter(freqs=notch_freq)

        # Compute Power Spectral Density (PSD)
        psd_data = raw.compute_psd(method='welch', fmin=high_pass_freq, fmax=low_pass_freq, n_fft=2048, picks='meg', n_jobs=-1)
        psd, freqs = psd_data.get_data(return_freqs=True)

        # Convert PSD to dB (as MNE does by default)
        psd_dB = 10 * np.log10(psd)

        # Create a Plotly figure
        psd_fig = go.Figure()

        # Plot multiple channels with transparency for better readability
        for ch_idx, ch_name in enumerate(config.ALL_CH_NAMES_PREFIX):
            psd_fig.add_trace(go.Scatter(
                x=freqs,
                y=psd_dB[ch_idx],  
                mode='lines',
                line=dict(width=1),
                name=ch_name
            ))

        # Update layout to match MNE’s default style
        psd_fig.update_layout(
            title="Power Spectral Density (PSD)",
            xaxis=dict(
                title="Frequency (Hz)",
                type="linear",  # MNE uses linear frequency scale
                showgrid=True
            ),
            yaxis=dict(
                title="Power (dB)",  # Log scale power in dB
                type="linear",
                showgrid=True
            ),
            # template="plotly_white"
        )

        return dcc.Graph(figure = psd_fig, style={"padding": "10px", "borderRadius": "10px", "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"})

@callback(
    Output("preprocess-status", "children", allow_duplicate=True),
    Output("url", "pathname"),
    Output("annotations-store", "data"),
    Output("chunk-limits-store", "data"),
    Input("preprocess-display-button", "n_clicks"),
    State("folder-store", "data"),
    State("frequency-store", "data"),
    State("heartbeat-channel", "value"),
    running=[
        (Output("preprocess-display-button", "disabled"), True, False),
        (Output("load-button", "disabled"), True, False)],
    prevent_initial_call=True
)
def preprocess_meg_data(n_clicks, folder_path, freq_data, heartbeat_ch_name):
    """Preprocess MEG data and save it, store annotations and chunk limits in memory."""

    if n_clicks > 0:

        try:
            raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)

            annotations_dict, max_length = au.get_annotations_dataframe(raw, heartbeat_ch_name)
            chunk_limits = pu.update_chunk_limits(max_length)

            resample_freq = freq_data.get("resample_freq")
            low_pass_freq = freq_data.get("low_pass_freq")
            high_pass_freq = freq_data.get("high_pass_freq")
            notch_freq = freq_data.get("notch_freq")

            # Apply filtering and resampling
            raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
            raw.notch_filter(freqs=notch_freq)
            raw.resample(resample_freq)

            for chunk_idx in chunk_limits:
                start_time, end_time = chunk_idx
                raw_df = pu.get_preprocessed_dataframe(folder_path, freq_data, start_time, end_time, raw)
            return "Preprocessed and saved data", "/viz/raw-signal", annotations_dict, chunk_limits
        
        except Exception as e:
            return f"Error during preprocessing : {str(e)}", dash.no_update, None, None

    return None, dash.no_update, None, None