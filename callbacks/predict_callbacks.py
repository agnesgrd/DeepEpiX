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
            environment = "TensorFlow (.tfenv)"
        elif selected_value.endswith(".pth"):
            environment = "PyTorch (.torchenv)"
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
        State('adjust-onset', 'value'),
        prevent_initial_call = True
    )
    def execute_predict_script(n_clicks, subject_folder_path, model_path, spike_name, venv, threshold, sensitivity_analysis, adjust_onset):
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
            
        # else:
        #     raise RuntimeError("No virtual environment detected. Please activate one before running the script.")
        


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
                str(adjust_onset)
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
                str(adjust_onset)
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
        predictions_csv_path = Path.cwd() / "results/predictions.csv"
        result = pd.read_csv(predictions_csv_path)

        # Assuming df is your DataFrame
        result_filtered = result[result["probas"] > threshold]
        
        prediction_table = dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in result_filtered.columns],
            data=result_filtered.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "8px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
            page_size=5,  # Pagination
        )

        if sensitivity_analysis == "Yes":

            command = [
                str(Path.cwd() / f"{c.TENSORFLOW_ENV}/bin/python"),
                f"model_pipeline/run_smoothgrad.py",
                str(model_path),
                str(venv),
                str(Path.cwd() / "results"),
                str(Path.cwd() / "results/predictions.csv"),
                str(threshold)  # Ensure threshold is passed as a string 
            ]

            try: 
                # Start timing for the second subprocess
                start_time = time.time()

                subprocess.run(command, env=env, text = True) # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # End timing for the second subprocess
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Smoothgrad executed in {elapsed_time:.2f} seconds")

            except Exception as e:
                return f"Error running smoothgrad: {e}", prediction_table, 0, {"display": "block"}, dash.no_update

            # grad_store = {'smoothGrad': [sau.serialize_array(grad),grad.shape]}

            return True, prediction_table, 0, {"display": "block"}, {'smoothGrad': str(Path.cwd() / "results/smoothGrad.pkl"),}

        else:
            return True, prediction_table, 0, {"display": "block"}, dash.no_update

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