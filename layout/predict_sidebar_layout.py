import dash
from dash import html, Output, Input, State, dcc
from layout import input_styles, box_styles, button_styles
import numpy as np
import dash_bootstrap_components as dbc
import subprocess
import pickle
import pandas as pd

def create_predict():
    layout = html.Div([
        html.Label(
                "Path to pipeline folder:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
        dbc.Input(id='param1', type='text', value='/home/admin_mel/Documents/DeepEpi/pipeline/', style={**input_styles["small-number"]}),
        
        html.Label(
                "Path to pipeline folder:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
        dbc.Input(id='param2', type='text', value='/home/admin_mel/Documents/DeepEpi/pipeline/good_channels', style={**input_styles["small-number"]}),
        
        html.Label(
                "Path to subject folder (.ds):",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
        dbc.Input(id='param3', type='text', value='/home/admin_mel/Code/DeepEpiX/data/exampleData/criso_Epi-001_20100322_03.ds', style={**input_styles["small-number"]}),
        
        html.Label("Enter parameter 4:"),
        dbc.Input(id='param4', type='text', value='1', style={**input_styles["small-number"]}),
        
        html.Label(
                "Path to model (.keras or .hp5):",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
        dbc.Input(id='param5', type='text', value='/home/admin_mel/Documents/DeepEpi/pipeline/tensorflow_light_features_model.keras', style={**input_styles["small-number"]}),
        
        html.Label(
                "Detected spike name:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
        dbc.Input(id='param6', type='text', value='detected_spikes_name', style={**input_styles["small-number"]}),

        html.Label(
                "Environment:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
        dbc.Input(id='param7', type='text', value='Tensorflow', style={**input_styles["small-number"]}),

        # Prediction Button
        dbc.Button(
                    "Run Prediction",
                    id="run-prediction-button",  # Unique ID for each button
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    disabled=False,
                    style=button_styles["big"]
                ),

        # Output container
        html.Div(id='prediction-output')
    ])
    return layout

def run_predict_script(param1, param2, param3, param4, param5, param6, param7):
    process = subprocess.Popen(
        ['python', '/home/admin_mel/Documents/DeepEpi/pipeline/main.py', 
        str(param1), str(param2), str(param3), str(param4), 
        str(param5), str(param6), str(param7)],
        stdout=subprocess.PIPE,  
        stderr=subprocess.PIPE,  
        text=True,  
        bufsize=1,  
        universal_newlines=True
    )

    output_lines = []  # Store stdout
    error_lines = []  # Store stderr

    # Read and collect output in real-time
    for line in process.stdout:
        print(line, end="")  
        output_lines.append(line.strip())  

    # for line in process.stderr:
    #     print("ERROR:", line, end="")  
    #     error_lines.append(line.strip())  

    process.wait()  # Ensure process finishes

    # The last printed line should be the CSV file path
    csv_path = "/home/admin_mel/Code/DeepEpiX/results/model_predictions.csv"

    if csv_path and csv_path.endswith(".csv"):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            timepoints = df['Timepoint'].tolist()
            return timepoints  # Return the list of timepoints
        except Exception as e:
            return f"Error reading CSV: {e}"
    else:
        return "No valid output received from main.py."

# Dash callback
@dash.callback(
    Output('prediction-output', 'children'),
    Input('run-prediction-button', 'n_clicks'),
    Input('param1', 'value'),
    Input('param2', 'value'),
    Input('param3', 'value'),
    Input('param4', 'value'),
    Input('param5', 'value'),
    Input('param6', 'value'),
    Input('param7', 'value')
)
def execute_predict_script(n_clicks, param1, param2, param3, param4, param5, param6, param7):
    if not n_clicks or n_clicks == 0:
        return "Click the button to start prediction."

    # Ensure parameters are valid (avoid None values)
    params = [param1, param2, param3, param4, param5, param6, param7]
    if any(p is None for p in params):
        return "Please provide all required parameters."

    result = run_predict_script(param1, param2, param3, param4, param5, param6, param7)
    return f"Prediction Output:\n{result}"




