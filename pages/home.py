import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import os
import mne
from layout import input_styles, box_styles
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import folder_path_utils as fpu
from collections import Counter
from callbacks.utils import annotation_utils as au
import plotly.graph_objects as go
import numpy as np
import static.constants as c
from dash.exceptions import PreventUpdate

# Register the page
dash.register_page(__name__, path = "/")

layout = html.Div([
    
    # Explanation of what the user needs to do
    html.Div([
        html.H3([
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
        ], style={"padding": "10px"}),

    ], style={"padding": "15px", "backgroundColor": "#fff", "border": "1px solid #ddd","borderRadius": "8px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "marginBottom": "20px"}),

    html.Div(
        id="frequency-container",
        children=[
            # Frequency Inputs (Left Side)
            html.Div(
                id="frequency-inputs",
                children=[
                    html.H3([
                        html.I(className="bi bi-2-circle-fill", style={"marginRight": "10px", "fontSize": "1.2em"}),
                         "Set Frequency Parameters for Signal Preprocessing"]),
                    # html.H3("Frequency Parameters for Signal Processing", style={"margin-bottom": "15px"}),

                    html.Div([
                        html.Label("Resampling Frequency (Hz): "),
                        dbc.Input(id="resample-freq", type="number", value=150, step=1, min=50, style=input_styles["number-in-box"]),
                    ], style={"padding": "10px"}),

                    html.Div([
                        html.Label("High-pass Frequency (Hz): "),
                        dbc.Input(id="high-pass-freq", type="number", value=0.5, step=0.1, min=0.1, style=input_styles["number-in-box"]),
                    ], style={"padding": "10px"}),

                    html.Div([
                        html.Label("Low-pass Frequency (Hz): "),
                        dbc.Input(id="low-pass-freq", type="number", value=50, step=1, min=1, style=input_styles["number-in-box"]),
                    ], style={"padding": "10px"}),

                    html.Div([
                        html.Label("Notch filter Frequency (Hz): "),
                        dbc.Input(id="notch-freq", type="number", value=50, step=1, min=0, style=input_styles["number-in-box"]),
                    ], style={"padding": "10px"}),

                    html.H3([
                        html.I(className="bi bi-3-circle-fill", style={"marginRight": "10px", "fontSize": "1.2em"}),
                         "Give hint on channel name for heartbeat detection"]),
                    # html.H3("Frequency Parameters for Signal Processing", style={"margin-bottom": "15px"}),

                    html.Div([
                        html.Label("Channel Name for Heartbeat Detection (ex: MRF52-2805, default = None): "),
                        dbc.Input(id="heartbeat-channel", type="text", style=input_styles["number-in-box"]),
                    ], style={"padding": "10px"}),

                    html.Div([
                        dbc.Button("Preprocess & Display", id="preprocess-display-button", color="success", disabled=True, n_clicks=0),
                        dcc.Loading(
                            id="loading",
                            type="default",
                            children=[html.Div(id="preprocess-status", style={"margin-top": "10px"})]
                        ),
                        dcc.Location(id="url", refresh=True),
                    ], style={"padding": "10px", "margin-top": "20px"})
                ],
                style={
                    "padding": "15px",
                    "backgroundColor": "#fff",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                    "width": "40%",  # Set width for left panel
                    "marginRight": "20px"  # Add spacing between elements
                }
            ),

            html.Div(
                id="analysis", 
                children = [
                    dcc.Tabs(
                        id="tabs",
                        value='raw-info-tab',  # Default selected tab
                        children=[
                            dcc.Tab(label='Raw Info', value='raw-info-tab', children=[
                                html.Div(id="raw-info-container", children=[

                                    dcc.Loading(
                                        type="default",
                                        children=[
                                            dash.dash_table.DataTable(
                                                id='raw-info-table',
                                                columns=[
                                                    {"name": "Property", "id": "Property"},
                                                    {"name": "Value", "id": "Value"}
                                                ],
                                                data=[],  # To be filled via callback
                                                style_cell={
                                                    "textAlign": "left",
                                                    "padding": "10px",
                                                    "whiteSpace": "normal",
                                                    "height": "auto"
                                                },
                                                style_header={
                                                    "backgroundColor": "#f8f9fa",
                                                    "fontWeight": "bold"
                                                },
                                                style_table={"overflowX": "auto", "marginTop": "15px"},
                                            )
                                        ]
                                    )
                                ])
                            ]),
                            dcc.Tab(label='Power Spectral Density', value='psd-tab', children=[
                                # Channel Statistics Section
                                html.Div(id="psd-container", children=[

                                    dbc.Button("Compute & Display", id="compute-display-psd-button", color="success", n_clicks=0, style = {"marginTop": "15px"}),

                                    dcc.Graph(id="psd-graph", style={"display": "none"}),

                                    dcc.Loading(
                                            id="loading",
                                            type="default",
                                            children=[html.Div(id="psd-status", style={"margin-top": "10px"})]
                                        ),
                                    # You can add additional content related to channel statistics
                                ]),
                            ]),
                            dcc.Tab(label='Event Statistics', value='events-tab', children=[
                                # Event Statistics Section
                                html.Div(id="event-stats-container", children=[
                                    # You can add content related to event analysis, like event count over time
                                ], style = {"marginTop": "15px"}),
                            ]),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "row",  # Ensure tabs are displayed in a row (horizontal)
                            "alignItems": "center",  # Center the tabs vertically within the parent container
                            "width": "100%",  # Full width of the container
                            "borderBottom": "1px solid #ddd"  # Optional, adds a bottom border to separate from content
                        }
                    ),
                ],
                style={
                    "padding": "15px",
                    "backgroundColor": "#fff",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                    "width": "60%",  # Adjust the width as needed
                }
            ),
        ],


        style={
            "display": "none",
            "flexDirection": "row",  # Side-by-side layout
            "alignItems": "flex-start",  # Align to top
            "gap": "20px",  # Add spacing between elements
            "width": "100%"  # Ensure full width
        }
    )
])

# Callback to update dropdown when button is clicked
@callback(
    Output("folder-path-dropdown", "options"),
    Output("folder-path-dropdown", "value"),  # Set selected folder
    Input("open-folder-button", "n_clicks"),
    State("folder-path-dropdown", "options"),
    prevent_initial_call=True
)
def update_dropdown(n_clicks, folder_path_list):
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
    """Validate entered folder path for .ds"""
    # Check if folder exists and finish by .ds, then make "load" button clickable
    if folder_path:
        if os.path.isdir(folder_path):            
            if folder_path.endswith(".ds"):
                return folder_path, False
    return dash.no_update, dash.no_update

@callback(
    Output("raw-info-table", "data"),
    Input("load-button", "n_clicks"),
    State("folder-store", "data"),
    prevent_initial_call=True
)
def populate_raw_info(n_clicks, folder_path):
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

    return data

@callback(
    Output("event-stats-container", "children"),
    Input("tabs", "value"),
    State("folder-store", "data"),
    prevent_initial_call=True
)
def populate_events_statistics(selected_tab, folder_path):
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
    annotation_table = dbc.Table(table_header + [html.Tbody(table_body)], bordered=True, striped=True, hover=True)

    # Optionally show total number and a few more stats
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
    """Display frequency parameters when button is clicked"""
    if n_clicks > 0:
        style = {"display": "flex",
            "flexDirection": "row",  # Side-by-side layout
            "alignItems": "flex-start",  # Align to top
            "gap": "20px",  # Add spacing between elements
            "width": "100%"} # Ensure full width
        return style, True
    return dash.no_update, dash.no_update


@callback(
    Output("preprocess-status", "children"),
    Output("frequency-store", "data"),
    Output("preprocess-display-button", "disabled"),
    Input("resample-freq", "value"),
    Input("high-pass-freq", "value"),
    Input("low-pass-freq", "value"),
    Input("notch-freq", "value"),
    prevent_initial_call=True
)
def handle_frequency_parameters(resample_freq, high_pass_freq, low_pass_freq, notch_freq):
    """Retrieve frequency parameters and use them for analysis."""

    if not low_pass_freq or not high_pass_freq or not notch_freq:
        return "Please fill in all frequency parameters.", dash.no_update, True
    
    elif high_pass_freq >= low_pass_freq:
        error = "High-pass frequency must be less than low-pass frequency."
        return error, dash.no_update, True
    else:
        # Store the frequency values when the folder is valid
        frequency_values = {
            "resample_freq": resample_freq,
            "low_pass_freq": low_pass_freq,
            "high_pass_freq": high_pass_freq,
            "notch_freq": notch_freq
        }
        return None, frequency_values, False
  
@callback(
    Output("psd-status", "children"),
    Output("psd-graph", "figure"),
    Output("psd-graph", "style"),
    Input("compute-display-psd-button", "n_clicks"),
    State("folder-store", "data"),
    State("frequency-store", "data"),
    prevent_initial_call=True
)
def display_psd(n_clicks, folder_path, freq_data):

    if folder_path is None or freq_data is None:
        return "Please fill in all frequency parameters.", dash.no_update, dash.no_update
    
    if n_clicks > 0:
    
        raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)

        resample_freq = freq_data.get("resample_freq")
        low_pass_freq = freq_data.get("low_pass_freq")
        high_pass_freq = freq_data.get("high_pass_freq")
        notch_freq = freq_data.get("notch_freq")

        if not low_pass_freq or not high_pass_freq or not notch_freq:
            return dash.no_update
        
        raw.notch_filter(freqs=notch_freq)

        # Create the PSD plot using Plotly
        psd_fig = go.Figure()

        # Compute Power Spectral Density (PSD)
        psd_data = raw.compute_psd(method='welch', fmin=high_pass_freq, fmax=low_pass_freq, n_fft=2048, picks='meg', n_jobs=-1)
        psd, freqs = psd_data.get_data(return_freqs=True)

        # Convert PSD to dB (as MNE does by default)
        psd_dB = 10 * np.log10(psd)

        # Create a Plotly figure to mimic MNE’s PSD plot
        psd_fig = go.Figure()

        # Plot multiple channels with transparency for better readability
        for ch_idx, ch_name in enumerate(c.ALL_CH_NAMES_PREFIX):  # Plot only first 10 channels
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
            template="plotly_white"
        )

            # Example: calculate variance for each channel
        variances = np.var(raw.get_data(), axis=-1)
        threshold=0.001
        bad_channels = np.where(variances > threshold)[0]  # Apply a threshold for bad channels
        
        # Create a bar chart for channel variance
        fig = go.Figure(data=[go.Bar(x=raw.info['ch_names'], y=variances)])
        fig.update_layout(title="Channel Variance (Bad Channels Highlighted)",
                        xaxis_title="Channels",
                        yaxis_title="Variance")

        return None, fig, {}

@callback(
    Output("preprocess-status", "children", allow_duplicate=True),
    Output("url", "pathname"),
    Output("annotations-store", "data"),
    Output("chunk-limits-store", "data"),
    Input("preprocess-display-button", "n_clicks"),
    State("folder-store", "data"),
    State("frequency-store", "data"),
    State("heartbeat-channel", "value"),
    prevent_initial_call=True
)
def preprocess_meg_data(n_clicks, folder_path, freq_data, heartbeat_ch_name):
    """Preprocess MEG data and save it."""
    if n_clicks is None:
        raise PreventUpdate
    elif n_clicks > 0:
        # cache.clear()
        try:
            raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)

            annotations_dict, max_length = au.get_annotations_dataframe(raw, heartbeat_ch_name)
            chunk_limits = pu.update_chunk_limits(max_length)

            resample_freq = freq_data.get("resample_freq")
            low_pass_freq = freq_data.get("low_pass_freq")
            high_pass_freq = freq_data.get("high_pass_freq")
            notch_freq = freq_data.get("notch_freq")

            # # Apply filtering and resampling
            # raw=pu.interpolate_missing_channels(raw)
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
    








