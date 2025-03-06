from dash import html, dcc
from layout import input_styles, box_styles, button_styles, label_styles
import dash_bootstrap_components as dbc
from callbacks.utils import predict_utils as pu

def create_anom_detect():
    layout = html.Div([

    # Model selection
    html.Div([
        html.Label("Available Models:", style={**label_styles["classic"]}),
        dcc.Dropdown(
            id="model-ae-dropdown",
            options=pu.get_model_options(AE=True),
            placeholder="Select ...",
        ),
    ], style={"marginBottom": "20px"}),

    # Environment input
    html.Div([
        html.Label("Environment:", style={**label_styles["classic"]}),
        dbc.Input(id="venv-ae", type="text", value="", disabled=True, style={**input_styles["small-number"]}),
    ], style={"marginBottom": "20px"}),

    # Detected spike name input
    html.Div([
        html.Label("Detected Anomalies Name:", style={**label_styles["classic"]}),
        dbc.Input(id="model-detected-anomalies-name", type="text", value="detected_anomalies_name", style={**input_styles["small-number"]}),
    ], style={"marginBottom": "20px"}),

    # Threshold
    html.Div([
        html.Label("Threshold:", style={**label_styles["classic"]}),
        dbc.Input(id="anomaly-detection-threshold", type="number", value=0.5, step=0.01, min=0, max=1, style=input_styles["small-number"]),
    ], style={"marginBottom": "20px"}),

    # Compute sensitvity analysis at the end
    # html.Div([
    #     html.Label("Adjust onset (Global Field Power):", style={**label_styles["classic"]}),
    #     dbc.RadioItems(
    #         id="adjust-onset-anom",
    #         options=[
    #             {"label": "Yes", "value": "Yes"},
    #             {"label": "No", "value": "No"}
    #         ],
    #         value="Yes",  # Default selection
    #         inline=True,  # Display buttons in a row
    #         style={"margin-left": "10px"}
    #     )
    # ], style={"marginBottom": "20px"}),

    # Run Prediction Button
    html.Div([
        dbc.Button(
            "Run Prediction",
            id="run-anomaly-detection-button",
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
            html.Div(id="anomaly-detection-status", style={"margin-top": "10px"})
        ]),

    # Prediction Output
    html.Div(id = "store-display-anomaly-detection-div", children = [

        html.Div(id="anomaly-detection-output", style={"marginTop": "20px", "textAlign": "center"}),

        dbc.Button(
            "Display",
            id="store-display-anomaly-detection-button",
            color="success",
            outline=True,
            disabled=False,
            n_clicks=0,
            style = button_styles["big"]
        )], style = {"display": "none"}),
     
    ]),    
    return layout