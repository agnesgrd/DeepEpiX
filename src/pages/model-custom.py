import dash
from dash import html, callback, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from config import MODEL_PIPELINE_DIR
import callbacks.utils.model_utils as mu

dash.register_page(__name__, name="Model Custom", path="/model/custom")

layout = html.Div(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H4(
                                [
                                    "Your Models",
                                    html.I(
                                        className="bi bi-info-circle-fill",
                                        id="models-help-icon",
                                        style={
                                            "fontSize": "0.8em",
                                            "cursor": "pointer",
                                            "verticalAlign": "middle",
                                            "marginLeft": "15px",
                                        },
                                    ),
                                ],
                                style={"margin": "0px"},
                            ),
                            dbc.Tooltip(
                                "Here you can see your pretrained models and build your prediction pipeline for your own pretrained model.",
                                target="models-help-icon",
                                placement="right",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "gap": "20px",
                            "margin": "30px",
                        },
                    ),
                    mu.render_pretrained_models_table(),
                ]
            ),
            className="mb-5",
            style={
                "width": "100%",
                "border": "none",
                "boxShadow": "0px 4px 12px rgba(13, 110, 253, 0.3)",  # soft blue shadow
                "borderRadius": "12px",  # smooth corners
            },
        ),
        html.Div(
            [
                html.H6(
                    [
                        html.I(
                            className="bi bi-1-circle-fill",  # numbered icon
                            style={"marginRight": "10px", "fontSize": "1.2em"},
                        ),
                        "Place your model file (.keras, .pth, .h5, etc.) into the folder src → models so it can be detected automatically.",
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.H6(
                    [
                        html.I(
                            className="bi bi-2-circle-fill",
                            style={"marginRight": "10px", "fontSize": "1.2em"},
                        ),
                        "Generate a template by specifying your model name. The template provides the correct input and output format for proper functioning.",
                    ],
                    style={"marginBottom": "20px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Input(
                                id="file-name",
                                type="text",
                                placeholder="Enter you model name",
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Create Page",
                                id="create-btn",
                                color="primary",
                                n_clicks=0,
                                className="ms-2",
                            ),
                            width=4,
                        ),
                        html.Br(),
                        dcc.Loading(
                            id="add-model-loader",
                            type="default",
                            children=[
                                html.Div(
                                    id="add-model-status",
                                    style={"marginTop": "10px", "marginBottom": "20px"},
                                )
                            ],
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.H6(
                    [
                        html.I(
                            className="bi bi-3-circle-fill",
                            style={"marginRight": "10px", "fontSize": "1.2em"},
                        ),
                        "Fill the pipeline template with your preprocessing, inference, and postprocessing code.",
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.H6(
                    [
                        html.I(
                            className="bi bi-4-circle-fill",
                            style={"marginRight": "10px", "fontSize": "1.2em"},
                        ),
                        "Go to Visualization → Raw Signal and click on Predictions icon to run the pipeline with your model.",
                    ]
                ),
            ],
            style={"margin": "40px"},
        ),
        # dbc.Container(
        #     [
        #         dbc.Row(
        #             [
        #                 dbc.Col(
        #                     [
        #                         dbc.Card(
        #                             [
        #                                 dbc.CardHeader(
        #                                     html.H4(
        #                                         "Create a custom prediction pipeline for your model"
        #                                     )
        #                                 ),
        #                                 dbc.CardBody(
        #                                     [
        #                                         dbc.Form(
        #                                             [
        #                                                 dbc.Row(
        #                                                     [
        #                                                         dbc.Col(
        #                                                             dbc.Input(
        #                                                                 id="file-name",
        #                                                                 type="text",
        #                                                                 placeholder="Enter you model name",
        #                                                             ),
        #                                                             width=8,
        #                                                         ),
        #                                                         dbc.Col(
        #                                                             dbc.Button(
        #                                                                 "Create Page",
        #                                                                 id="create-btn",
        #                                                                 color="primary",
        #                                                                 n_clicks=0,
        #                                                                 className="ms-2",
        #                                                             ),
        #                                                             width=4,
        #                                                         ),
        #                                                     ]
        #                                                 )
        #                                             ]
        #                                         ),
        #                                         html.Br(),
        #                                         dbc.Alert(
        #                                             id="output",
        #                                             color="success",
        #                                             is_open=False,
        #                                         ),
        #                                     ]
        #                                 ),
        #                             ]
        #                         )
        #                     ],
        #                     width=8,
        #                 )
        #             ],
        #             justify="center",
        #             className="mt-5",
        #         )
        #     ],
        #     fluid=True,
        # ),
    ]
)


@callback(
    Output("add-model-status", "children"),
    Input("create-btn", "n_clicks"),
    State("file-name", "value"),
    prevent_initial_call=True,
)
def create_new_page(n_clicks, file_name):
    if n_clicks is None or not file_name:
        return "", dash.no_update

    # Ensure .py extension
    if not file_name.startswith("run"):
        file_name = "run_" + file_name
    if not file_name.endswith(".py"):
        file_name += ".py"

    # Template with professional structure
    template = '''
"""
Model Pipeline Template
Author: Your Name
Created: Date
Description: Standard template for running model pipelines with test_model.
"""

# === Import ===
import os
import pandas as pd
import gc
from tensorflow import keras


# === Helper functions specific to your model ===
def prepare_data():
    """
    Prepare dataset for the model.
    Replace with your actual preprocessing pipeline.
    """
    # TODO


def load_model(model_name):
    """
    Load a model from disk.
    Args:
        model_name: Path to the trained model file.
    Returns:
        Compiled model ready for inference.
    """
    # TODO


def predict_windows() -> list[float]:
    """
    Run predictions on the test dataset.
    Args:
        ?
    Returns:
        List of predicted probabilities.
    """
    # TODO


def get_adjusted_onsets() -> list[float]:
    """
    Compute adjusted onsets (e.g. aligning to GFP peaks).
    Args:
        ?
    Returns:
        List of adjusted onsets.
    """
    # TODO


def get_onsets(output_path):
    """
    Compute raw timing windows.
    Args:
        ?
    Returns:
        List of timing windows.
    """
    # TODO


def save_predictions(output_path, model_name, onsets, y_pred_probas):
    """Save predictions into a CSV file compatible with MNE annotations."""
    df = pd.DataFrame(
        {
            "onset": onsets,
            "duration": 0,
            "probas": y_pred_probas,
        }
    )
    output_file = os.path.join(
        output_path, f"{os.path.basename(model_name)}_predictions.csv"
    )
    df.to_csv(output_file, index=False)
    return output_file


# === Main function ===
def test_model(
    model_name,
    model_type,
    subject,
    output_path,
    threshold=0.5,
    adjust_onset=True,
    channel_groups=None,
):
    """Run the full pipeline: prepare data, predict, adjust onsets, and save results."""
    # 1. Data preparation
    X_test_ids = prepare_data(subject, output_path, channel_groups)

    # 2. Load model
    model = load_model(model_name)

    # 3. Predictions
    y_pred_probas = predict_windows(model, X_test_ids, model_name, output_path)

    # 4. Cleanup model & GPU memory
    del model
    gc.collect()
    keras.backend.clear_session()

    # 5. Adjust onset times
    if adjust_onset:
        onsets = get_adjusted_onsets(X_test_ids, y_pred_probas, output_path)
    else:
        onsets = get_onsets(output_path)

    # 6. Save predictions
    return save_predictions(output_path, model_name, onsets, y_pred_probas)


'''

    try:
        with open(MODEL_PIPELINE_DIR / file_name, "w") as f:
            f.write(template)
        return (
            f"File '{file_name}' created successfully with standard template in {str(MODEL_PIPELINE_DIR / file_name)}",
        )
    except Exception as e:
        return f"Error: {e}"
