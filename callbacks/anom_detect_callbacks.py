import dash
from dash import Output, Input, State, dash_table
import subprocess
import pandas as pd
import static.constants as c
from pathlib import Path
import static.constants as c
import os
import sys
import time


def register_update_selected_model_anom_detect():
    @dash.callback(
        Output("venv-ae", "value"),
        Input("model-ae-dropdown", "value"),
        prevent_initial_call = True
    )
    def update_selected_model(selected_value):
        """Update the selected model path and detect the environment."""
        if not selected_value:
            return dash.no_update
        
        # Detect environment
        if selected_value.endswith((".keras", ".h5")):
            environment = "TensorFlow (.tfenv)"
        elif selected_value.endswith(".pth"):
            environment = "PyTorch (.torchenv)"
        else:
            environment = "Unknown"

        return environment
    
def register_execute_predict_script_anom_detect():
    @dash.callback(
        Output("anomaly-detection-status", "children"),
        Output('anomaly-detection-output', 'children'),
        Output('run-anomaly-detection-button', 'n_clicks'),
        Output('store-display-anomaly-detection-div', 'style'),
        Output('anomaly-detection-store', 'data'),
        Input('run-anomaly-detection-button', 'n_clicks'),
        State('folder-store', 'data'),
        State('model-ae-dropdown', 'value'),
        State('model-detected-anomalies-name', 'value'),
        State('venv-ae', 'value'),
        State('anomaly-detection-threshold', 'value'),
        prevent_initial_call = True
    )
    def execute_predict_script(n_clicks, subject_folder_path, model_path, anom_name, venv, threshold):
        if not n_clicks or n_clicks == 0:
            return None, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Validation: Check if all required fields are filled
        missing_fields = []
        if not subject_folder_path:
            missing_fields.append("Subject Folder")
        if not model_path:
            missing_fields.append("Model")
        if not anom_name:
            missing_fields.append("Detected Anomalies Name")
        if not venv:
            missing_fields.append("Environment")
        if threshold is None:
            missing_fields.append("Threshold")

        if missing_fields:
            error_message = f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}"
            return error_message, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        working_dir = Path.cwd()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(working_dir)

        # Activate TensorFlow venv and run script
        if "CONDA_PREFIX" in os.environ:
            activate_env = f"conda run -n {c.TENSORFLOW_ENV} python"
        else:
            if os.name == "nt":
                activate_env = f"{Path.cwd()}/{c.TENSORFLOW_ENV}/Scripts/python.exe"
            else:
                activate_env = f"{Path.cwd()}/{c.TENSORFLOW_ENV}/bin/python"
            
        if "TensorFlow" in venv:

            command = [
                activate_env,
                f"model_pipeline/run_model.py",
                str(model_path),
                str(venv),
                str(subject_folder_path),
                str(Path.cwd() / "model_pipeline/good_channels"),
                str(Path.cwd() / "results"),
                str(threshold),  # Ensure threshold is passed as a string 
                "NO"
            ]
            print(command)

        elif "PyTorch" in venv:
            # Activate PyTorch venv and run script
            command = [
                str(Path.cwd() / f"{c.TORCH_ENV}/bin/python"),
                f"model_pipeline/run_model.py",
                str(model_path),
                str(venv),
                str(subject_folder_path),
                str(Path.cwd() / "model_pipeline/good_channels"),
                str(Path.cwd() / "results"),
                str(threshold),  # Ensure threshold is passed as a string 
                "no"
            ]

        try: 
                # Start timing
            start_time = time.time()

            subprocess.run(command, env=env, text=True) #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Model testing executed in {elapsed_time:.2f} seconds")

        except Exception as e:
            return f"Error running model: {e}", dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Load the DataFrame from CSV
        predictions_csv_path = Path.cwd() / f"results/{os.path.basename(model_path)}_predictions.csv"
        result = pd.read_csv(predictions_csv_path)
        
        prediction_table = dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in result.columns],
            data=result.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "8px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
            page_size=5,  # Pagination
        )

        mse_path = Path.cwd() / f"results/{os.path.basename(model_path)}_anomDetect.pkl"
        if os.path.exists(mse_path):
            return True, prediction_table, 0, {"display": "block"}, {"anomDetect": str(mse_path)}
        
        else:
            return True, prediction_table, 0, dash.no_update, dash.no_update
        

def register_display_anom_detect():
    @dash.callback(
        Output('annotations-store', 'data', allow_duplicate=True),
        Output('sidebar-tabs', 'active_tab', allow_duplicate=True),
        Output('store-display-anomaly-detection-div', 'style', allow_duplicate=True),
        Input('store-display-anomaly-detection-button', 'n_clicks'),
        State('annotations-store', 'data'),
        State('anomaly-detection-output', 'children'),
        State('model-spike-name', 'value'),
        prevent_initial_call = True
    )
    def store_display_prediction(n_clicks, annotation_data, prediction_table, spike_name):
        if not n_clicks or n_clicks == 0 or prediction_table is None:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Ensure annotation_data is initialized
        if not annotation_data:
            annotation_data = []

        prediction = prediction_table['props']['data']

        # Extract predictions (assuming prediction_table is a DataTable component)
        if isinstance(prediction, list):  # Data is already a list of dicts
            prediction_df = pd.DataFrame(prediction)
        else:
            return dash.no_update, dash.no_update, dash.no_update  # Invalid format

        # Convert predictions to annotation format
        new_annotations = prediction_df[['description', 'onset', 'duration']].copy()
        # new_annotations['description'] = spike_name  # Set spike name as description

        # Convert to dictionary format for storage
        new_annotations_dict = new_annotations.to_dict(orient="records")

        # Append new annotations
        annotation_data.extend(new_annotations_dict)

        # Return updated annotations and switch tab
        return annotation_data, "selection-tab", {"display": "none"}