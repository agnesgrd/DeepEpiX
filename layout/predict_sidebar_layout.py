import dash
from dash import html, Output, Input, State, dcc, dash_table
from layout import input_styles, box_styles, button_styles, label_styles
import numpy as np
import dash_bootstrap_components as dbc
import subprocess
import pickle
import pandas as pd
from model_pipeline.run_model import run_model_pipeline
from callbacks.utils import predict_utils as pu

def create_predict():
    layout = html.Div([

    # Model selection
    html.Div([
        html.Label("Available Models:", style={**label_styles["classic"]}),
        dcc.Dropdown(
            id="model-dropdown",
            options=pu.get_model_options(),
            placeholder="Select ...",
        ),
    ], style={"marginBottom": "20px"}),

    # Environment input
    html.Div([
        html.Label("Environment:", style={**label_styles["classic"]}),
        dbc.Input(id="venv", type="text", value="", disabled=True, style={**input_styles["small-number"]}),
    ], style={"marginBottom": "20px"}),

    # Detected spike name input
    html.Div([
        html.Label("Detected Spike Name:", style={**label_styles["classic"]}),
        dbc.Input(id="spike-name", type="text", value="detected_spikes_name", style={**input_styles["small-number"]}),
    ], style={"marginBottom": "20px"}),

        # Detected spike name input
    html.Div([
        html.Label("Threshold:", style={**label_styles["classic"]}),
        dbc.Input(id="threshold", type="number", value=0.5, step=0.1, min=0, max=1, style=input_styles["small-number"]),
    ], style={"marginBottom": "20px"}),



    # Run Prediction Button
    html.Div([
        dbc.Button(
            "Run Prediction",
            id="run-prediction-button",
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
            html.Div(id="prediction-status", style={"margin-top": "10px"})
        ]),

    # Prediction Output
    html.Div(id="prediction-output", style={"marginTop": "20px", "textAlign": "center"}),



    html.Div(id = "store-display-div", children = [
        dbc.Button(
            "Store & Display",
            id="store-display-button",
            color="success",
            outline=True,
            disabled=False,
            n_clicks=0,
            style = button_styles["big"]
        )], style = {"display": "none"}),
     
    ]),    
    return layout

@dash.callback(
    Output("venv", "value"),
    Input("model-dropdown", "value"),
    prevent_initial_call = True
)
def update_selected_model(selected_value):
    """Update the selected model path and detect the environment."""
    if not selected_value:
        return dash.no_update
    
    # Detect environment
    if selected_value.endswith((".keras", ".h5")):
        environment = "TensorFlow"
    elif selected_value.endswith(".pth"):
        environment = "PyTorch"
    else:
        environment = "Unknown"

    return environment

# def run_predict_script(param1, param2, param3, param4, param5, param6, param7):
#     process = subprocess.Popen(
#         ['python', '/home/admin_mel/Documents/DeepEpi/pipeline/main.py', 
#         str(param1), str(param2), str(param3), str(param4), 
#         str(param5), str(param6), str(param7)],
#         stdout=subprocess.PIPE,  
#         stderr=subprocess.PIPE,  
#         text=True,  
#         bufsize=1,  
#         universal_newlines=True
#     )

#     output_lines = []  # Store stdout
#     error_lines = []  # Store stderr

#     # Read and collect output in real-time
#     for line in process.stdout:
#         print(line, end="")  
#         output_lines.append(line.strip())  

#     # for line in process.stderr:
#     #     print("ERROR:", line, end="")  
#     #     error_lines.append(line.strip())  

#     process.wait()  # Ensure process finishes

#     # The last printed line should be the CSV file path
#     csv_path = "/home/admin_mel/Code/DeepEpiX/results/model_predictions.csv"

#     if csv_path and csv_path.endswith(".csv"):
#         try:
#             # Read the CSV file
#             df = pd.read_csv(csv_path)
#             timepoints = df['Timepoint'].tolist()
#             return timepoints  # Return the list of timepoints
#         except Exception as e:
#             return f"Error reading CSV: {e}"
#     else:
#         return "No valid output received from main.py."

# Dash callback
@dash.callback(
    Output("prediction-status", "children"),
    Output('prediction-output', 'children'),
    Output('run-prediction-button', 'n_clicks'),
    Output('store-display-div', 'style'),
    Input('run-prediction-button', 'n_clicks'),
    State('folder-store', 'data'),
    State('model-dropdown', 'value'),
    State('spike-name', 'value'),
    State('venv', 'value'),
    State('threshold', 'value'),
    prevent_initial_call = True
)
def execute_predict_script(n_clicks, subject_folder_path, model_path, spike_name, venv, threshold):
    if not n_clicks or n_clicks == 0:
        return None, dash.no_update, dash.no_update, dash.no_update

    # Validation: Check if all required fields are filled
    missing_fields = []
    if not subject_folder_path:
        missing_fields.append("Subject Folder")
    if not model_path:
        missing_fields.append("Model")
    if not spike_name:
        missing_fields.append("Spike Name")
    if not venv:
        missing_fields.append("Environment")
    if threshold is None:
        missing_fields.append("Threshold")

    if missing_fields:
        error_message = f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}"
        return error_message, dash.no_update, dash.no_update, dash.no_update
    
    result = run_model_pipeline(
        model_path, 
        venv, 
        '/home/admin_mel/Code/DeepEpiX/model_pipeline/good_channels', 
        subject_folder_path,
        "/home/admin_mel/Code/DeepEpiX/results/",
        threshold = float(threshold))
    
    prediction_table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in result.columns],
        data=result.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        page_size=5,  # Pagination
    )
    
    return True, prediction_table, 0, {"display": "block"}

@dash.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Output('sidebar-tabs', 'active_tab'),
    Input('store-display-button', 'n_clicks'),
    State('annotations-store', 'data'),
    State('prediction-output', 'children'),
    State('spike-name', 'value'),
    prevent_initial_call = True
)
def store_display_prediction(n_clicks, annotation_data, prediction_table, spike_name):
    if not n_clicks or n_clicks == 0:
        return dash.no_update, dash.no_update
    
    # Ensure annotation_data is initialized
    if annotation_data is None:
        annotation_data = []

    prediction = prediction_table['props']['data']

    # Extract predictions (assuming prediction_table is a DataTable component)
    if isinstance(prediction, list):  # Data is already a list of dicts
        prediction_df = pd.DataFrame(prediction)
    else:
        return dash.no_update, dash.no_update  # Invalid format

    # Convert predictions to annotation format
    new_annotations = prediction_df[['onset', 'duration']].copy()
    new_annotations['description'] = spike_name  # Set spike name as description

    # Convert to dictionary format for storage
    new_annotations_dict = new_annotations.to_dict(orient="records")

    # Append new annotations
    annotation_data.extend(new_annotations_dict)

    # Return updated annotations and switch tab
    return annotation_data, "selection-tab"
    
    




