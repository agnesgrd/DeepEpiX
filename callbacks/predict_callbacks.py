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
from pathlib import Path


def register_update_selected_model():
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
    
def register_execute_predict_script():
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
            Path.cwd() / "model_pipeline/good_channels", 
            subject_folder_path,
            Path.cwd() / "results",
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
    

def register_store_display_prediction():
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