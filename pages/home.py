import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import os
import mne
from layout import input_styles
from dash.exceptions import PreventUpdate
from flask_caching import Cache
from io import StringIO
import pandas as pd
from dash import get_app
from sklearn.preprocessing import StandardScaler

# Register the page
dash.register_page(__name__, path = "/")
app=get_app()
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 1000,
    'CACHE_THRESHOLD': 10 # higher numbers will store more data in the filesystem / redis cache
})

layout = html.Div([
    html.H1("HOME: Choose MEG Data Folder"),

    # Explanation of what the user needs to do
    html.Div([
        html.P("Please enter the full path to the .ds folder containing the data to analyze."),
        html.P("Example: /home/admin_mel/Code/DeepEpiX/data/berla_Epi-001_20100413_07.ds"),
        html.Div([
            dbc.Input(
                id="folder-path-input",
                type="text",
                placeholder="Enter folder path here...",
                style=input_styles["path"]
            )
        ], style={"padding": "10px"}),

        # Display the entered folder path
        html.Div([
            html.H4("Entered Folder Path:"),
            html.Div(id="entered-folder", style={"font-style": "italic", "color": "#555"}),
        ], style={"padding": "10px"}),

        dbc.Button(
            "Load",
            id="load-button",
            color="success",
            disabled=True,
            n_clicks=0
        )
    ], style={"padding": "15px", "backgroundColor": "#fff", "border": "1px solid #ddd", 
              "borderRadius": "8px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "marginBottom": "20px"}),

    # Section for frequency parameters, initially hidden
    html.Div(
        id="frequency-inputs",
        children=[
            html.H3("Frequency Parameters for Signal Processing", style={"margin-bottom": "15px"}),

            # Inputs for frequency parameters
            html.Div([
                html.Label("Resampling Frequency (Hz): "),
                dbc.Input(id="resample-freq", type="number", value=150, step=50, min=50, style=input_styles["number"]),
            ], style={"padding": "10px"}),

            html.Div([
                html.Label("High-pass Frequency (Hz): "),
                dbc.Input(id="high-pass-freq", type="number", value=0.5, step=0.1, min=0.1, style=input_styles["number"]),
            ], style={"padding": "10px"}),

            html.Div([
                html.Label("Low-pass Frequency (Hz): "),
                dbc.Input(id="low-pass-freq", type="number", value=50, step=10, min=10, style=input_styles["number"]),
            ], style={"padding": "10px"}),

            # Button and status display with loading spinner
            html.Div([
                dbc.Button(
                    "Preprocess & Display",
                    id="preprocess-display-button",
                    color="success",
                    disabled=True,
                    n_clicks=0
                ),
                # Loading spinner wraps only the elements that require loading
                dcc.Loading(
                    id="loading",
                    type="default", 
                    children=[
                        html.Div(id="preprocess-status", style={"margin-top": "10px"})
                    ]
                ),
                # Location for URL refresh
                dcc.Location(id="url", refresh=True),
            ], style={"padding": "10px", "margin-top": "20px"})
        ],
        style={
            "padding": "15px",
            "backgroundColor": "#fff",
            "border": "1px solid #ddd",  # Grey border
            "borderRadius": "8px",  # Rounded corners
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            "marginBottom": "20px",
            "display": "none"
        }
    )
])


@dash.callback(
    Output("entered-folder", "children"),
    Output("folder-store", "data"),
    Output("load-button", "disabled"),
    Input("folder-path-input", "value"),
    prevent_initial_call=True
)
def handle_valid_folder_path(folder_path):
    """Validate entered folder path for .ds"""
    if folder_path:
        # Check if folder exists and finish by .ds, then make "load" button clickable
        if os.path.isdir(folder_path):            
            if folder_path.endswith(".ds"):
                return f"{folder_path} (valid)", folder_path, False
            return f"{folder_path} should end with .ds.", None, True
        return f"{folder_path} does not exist. Please try again.", None, True
    return "Invalid input. Please enter a valid folder path.", None, True

@dash.callback(
    Output("frequency-inputs", "style"),
    Input("load-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_load_button(n_clicks):
    """Display frequency parameters when button is clicked"""
    if n_clicks > 0:
        return {"padding": "15px",
            "backgroundColor": "#fff",
            "border": "1px solid #ddd",  # Grey border
            "borderRadius": "8px",  # Rounded corners
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            "marginBottom": "20px",
            "display": "block"}

@dash.callback(
    Output("frequency-store", "data"),
    Output("preprocess-display-button", "disabled"),
    Input("resample-freq", "value"),
    Input("high-pass-freq", "value"),
    Input("low-pass-freq", "value"),
    prevent_initial_call=True
)
def handle_frequency_parameters(resample_freq, high_pass_freq, low_pass_freq):
    """Retrieve frequency parameters and use them for analysis."""
    if high_pass_freq >= low_pass_freq:
        raise ValueError("High-pass frequency must be less than low-pass frequency.")
    else:
        # Store the frequency values when the folder is valid
        frequency_values = {
            "resample_freq": resample_freq,
            "low_pass_freq": low_pass_freq,
            "high_pass_freq": high_pass_freq
        }
        return frequency_values, False
  
def get_preprocessed_dataframe(folder_path, freq_data):
    @cache.memoize()
    def preprocess_meg_data(folder_path, freq_data):
        try:
            resample_freq = freq_data.get("resample_freq")
            low_pass_freq = freq_data.get("low_pass_freq")
            high_pass_freq = freq_data.get("high_pass_freq")

            raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)
            raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
            raw.resample(resample_freq)

            # Transform the raw data into a serializable format
            raw_df = raw.to_data_frame(picks="meg", index="time")  # Get numerical data (channels × time)
            # Standardisation des données channel par channel
            scaler = StandardScaler()

            # Appliquer la standardisation à chaque canal (les colonnes de raw_df sont les canaux)
            raw_df_standardized = raw_df.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten(), axis=0)
            
            return raw_df_standardized.to_json()
            
        except Exception as e:
            return f"Error during preprocessing : {str(e)}"
    
    return pd.read_json(StringIO(preprocess_meg_data(folder_path, freq_data)))

def get_annotations_dataframe(folder_path):
    raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)
    annotations_df = raw.annotations.to_data_frame()
    annotations_df['onset'] = pd.to_datetime(annotations_df['onset']).dt.tz_localize('UTC')
    origin_time = pd.Timestamp(raw.annotations.orig_time)
    annotations_df['onset'] = (annotations_df['onset'] - origin_time).dt.total_seconds()
    annotations_dict = annotations_df.to_dict(orient="records")
    return annotations_dict
    
@dash.callback(
    Output("preprocess-status", "children"),
    Output("url", "pathname"),
    Output("annotations-store", "data"),
    Input("preprocess-display-button", "n_clicks"),
    State("folder-store", "data"),
    State("frequency-store", "data"),
    prevent_initial_call=True
)
def preprocess_meg_data(n_clicks, folder_path, freq_data):
    """Preprocess MEG data and save it."""
    if n_clicks is None:
        raise PreventUpdate
    if n_clicks > 0:
        try:
            raw_df = get_preprocessed_dataframe(folder_path, freq_data)
            annotations_dict = get_annotations_dataframe(folder_path)
            return "Preprocessed and saved data", "/view", annotations_dict
        
        except Exception as e:
            return f"Error during preprocessing : {str(e)}", dash.no_update, None

    return None, dash.no_update, None
    








