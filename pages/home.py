import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import os
import mne
from layout import input_styles, box_styles
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import folder_path_utils as fpu
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
        html.H1("Choose MEG Data Folder"),
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
                    html.H3("Frequency Parameters for Signal Processing", style={"margin-bottom": "15px"}),

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

            # PSD Graph (Right Side)
            html.Div(
                id="psd",
                children=[
                    dcc.Graph(id="psd-graph")
                ],
                style={
                    "padding": "15px",
                    "backgroundColor": "#fff",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                    "width": "60%",  # Set width for right panel
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
  
# def get_preprocessed_dataframe(folder_path, freq_data):
#     @cache.memoize()
#     def preprocess_meg_data(folder_path, freq_data):
#         try:
#             resample_freq = freq_data.get("resample_freq")
#             low_pass_freq = freq_data.get("low_pass_freq")
#             high_pass_freq = freq_data.get("high_pass_freq")

#             raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)
#             raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
#             raw.resample(resample_freq)

#             # Transform the raw data into a serializable format
#             raw_df = raw.to_data_frame(picks="meg", index="time")  # Get numerical data (channels × time)
#             # Standardisation des données channel par channel
#             scaler = StandardScaler()

#             # Appliquer la standardisation à chaque canal (les colonnes de raw_df sont les canaux)
#             raw_df_standardized = raw_df.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)
            
#             return raw_df_standardized.to_json()
            
#         except Exception as e:
#             return f"Error during preprocessing : {str(e)}"
    
#     return pd.read_json(StringIO(preprocess_meg_data(folder_path, freq_data)))

@callback(
    Output("psd-graph", "figure"),
    Input("folder-store", "data"),
    Input("frequency-store", "data"),
    prevent_initial_call=True
)
def display_psd(folder_path, freq_data):

    if folder_path is None or freq_data is None:
        return dash.no_update
    
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
    psd_data = raw.compute_psd(method='welch', fmin=high_pass_freq, fmax=low_pass_freq, n_fft=2048, picks='meg')
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

    return psd_fig

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

            # Apply filtering and resampling
            raw=pu.interpolate_missing_channels(raw)
            raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
            raw.notch_filter(freqs=notch_freq)
            raw.resample(resample_freq)

            for chunk_idx in chunk_limits:
                start_time, end_time = chunk_idx
                raw_df = pu.get_preprocessed_dataframe(folder_path, freq_data, start_time, end_time, raw)
            return "Preprocessed and saved data", "/view", annotations_dict, chunk_limits
        
        except Exception as e:
            return f"Error during preprocessing : {str(e)}", dash.no_update, None, None

    return None, dash.no_update, None, None
    








