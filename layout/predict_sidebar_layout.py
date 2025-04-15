from dash import html, dcc
from layout import input_styles, box_styles, button_styles, label_styles
import dash_bootstrap_components as dbc
from callbacks.utils import predict_utils as pu

def create_predict():
    layout = html.Div([

    # Model selection
    html.Div([
        html.Label("Available Models:", style={**label_styles["classic"]}),
        dcc.Dropdown(
            id="model-dropdown",
            options=pu.get_model_options('CNN'),
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
        dbc.Tooltip("Controls how confident the model must be to detect a spike.", target="threshold", placement="left"),
    ], style={"marginBottom": "20px"}),

    # Compute sensitvity analysis at the end
    html.Div([
        html.Label("Sensitivity Analysis (smoothGrad - only available for simple tensorflow models):", style={**label_styles["classic"]}),
        dbc.RadioItems(
            id="sensitivity-analysis",
            options=[
                {"label": "Yes", "value": "Yes"},
                {"label": "No", "value": "No"}
            ],
            value="Yes",  # Default selection
            inline=True,  # Display buttons in a row
            style={"margin-left": "10px"}
        ),
        dbc.Tooltip("SmoothGrad is an explainability technique that averages the gradients of the loss with respect to the input, over multiple noisy versions of the input. This reduces noise and highlights the regions in the time series that have the greatest influence on the model's predictions.", target="sensitivity-analysis", placement="left"),
    ], style={"marginBottom": "20px"}),

    # Compute sensitvity analysis at the end
    html.Div([
        html.Label("Adjust onset (Global Field Power):", style={**label_styles["classic"]}),
        dbc.RadioItems(
            id="adjust-onset",
            options=[
                {"label": "Yes", "value": "Yes"},
                {"label": "No", "value": "No"}
            ],
            value="Yes",  # Default selection
            inline=True,  # Display buttons in a row
            style={"margin-left": "10px"}
        ),
        dbc.Tooltip("Instead of setting the spike prediction to the middle of the window, this method adjusts the onset to the exact timestep where the Global Field Power (GFP) is at its maximum. This ensures more precise alignment with the probable spikes.", target="adjust-onset", placement="left"),
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
    




