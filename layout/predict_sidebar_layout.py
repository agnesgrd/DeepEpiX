import dash
from dash import html, Output, Input, State, dcc, dash_table
from layout import input_styles, box_styles, button_styles, label_styles
import numpy as np
import dash_bootstrap_components as dbc
import subprocess
import pickle
import pandas as pd
from model_pipeline.run_model import run_model_pipeline
from model_pipeline.smoothgrad import run_smoothgrad
from callbacks.utils import predict_utils as pu
import static.constants as c
from callbacks.utils import sensitivity_analysis_utils as sau

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
        dbc.Input(id="model-spike-name", type="text", value="detected_spikes_name", style={**input_styles["small-number"]}),
    ], style={"marginBottom": "20px"}),

    # Threshold
    html.Div([
        html.Label("Threshold:", style={**label_styles["classic"]}),
        dbc.Input(id="threshold", type="number", value=0.5, step=0.01, min=0, max=1, style=input_styles["small-number"]),
    ], style={"marginBottom": "20px"}),

    # Compute sensitvity analysis at the end
    html.Div([
        html.Label("Sensitivity Analysis (smoothGrad):", style={**label_styles["classic"]}),
        dbc.RadioItems(
            id="sensitivity-analysis",
            options=[
                {"label": "Yes", "value": "Yes"},
                {"label": "No", "value": "No"}
            ],
            value="Yes",  # Default selection
            inline=True,  # Display buttons in a row
            style={"margin-left": "10px"}
        )
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
    html.Div(id = "store-display-div", children = [

        html.Div(id="prediction-output", style={"marginTop": "20px", "textAlign": "center"}),

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

# Dash callback
@dash.callback(
    Output("prediction-status", "children"),
    Output('prediction-output', 'children'),
    Output('run-prediction-button', 'n_clicks'),
    Output('store-display-div', 'style'),
    Output('sensitivity-analysis-store', 'data'),
    Input('run-prediction-button', 'n_clicks'),
    State('folder-store', 'data'),
    State('model-dropdown', 'value'),
    State('model-spike-name', 'value'),
    State('venv', 'value'),
    State('threshold', 'value'),
    State('sensitivity-analysis', 'value'),
    prevent_initial_call = True
)
def execute_predict_script(n_clicks, subject_folder_path, model_path, spike_name, venv, threshold, sensitivity_analysis):
    if not n_clicks or n_clicks == 0:
        return None, dash.no_update, dash.no_update, dash.no_update, dash.no_update

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
        return error_message, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    y_pred, result = run_model_pipeline(
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

    if sensitivity_analysis == "Yes":
        grad = run_smoothgrad(model_path, y_pred)
        # grad_store = {'smoothGrad': [sau.serialize_array(grad),grad.shape]}
        grad_path = "results/smoothGrad.pkl"
        with open(grad_path, 'wb') as f:
            pickle.dump(grad, f)
    
    return True, prediction_table, 0, {"display": "block"}, {'smoothGrad': grad_path}

@dash.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Output('sidebar-tabs', 'active_tab'),
    Output('store-display-div', 'style', allow_duplicate=True),
    Input('store-display-button', 'n_clicks'),
    State('annotations-store', 'data'),
    State('prediction-output', 'children'),
    State('model-spike-name', 'value'),
    prevent_initial_call = True
)
def store_display_prediction(n_clicks, annotation_data, prediction_table, spike_name):
    if not n_clicks or n_clicks == 0 or prediction_table is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Ensure annotation_data is initialized
    if annotation_data is None:
        annotation_data = []

    prediction = prediction_table['props']['data']

    # Extract predictions (assuming prediction_table is a DataTable component)
    if isinstance(prediction, list):  # Data is already a list of dicts
        prediction_df = pd.DataFrame(prediction)
    else:
        return dash.no_update, dash.no_update, dash.no_update  # Invalid format

    # Convert predictions to annotation format
    new_annotations = prediction_df[['onset', 'duration']].copy()
    new_annotations['description'] = spike_name  # Set spike name as description

    # Convert to dictionary format for storage
    new_annotations_dict = new_annotations.to_dict(orient="records")

    # Append new annotations
    annotation_data.extend(new_annotations_dict)

    # Return updated annotations and switch tab
    return annotation_data, "selection-tab", {"display": "none"}
    
    




