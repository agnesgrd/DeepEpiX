import os
import time
import subprocess

import pandas as pd
import dash
from dash import Input, Output, State, html, callback
import dash_bootstrap_components as dbc

import config
from callbacks.utils import annotation_utils as au
from callbacks.utils import history_utils as hu


def register_update_selected_model():
    @callback(
        Output("venv", "value"),
        Input("model-dropdown", "value"),
        prevent_initial_call=True,
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
    @callback(
        Output("prediction-status", "children"),
        Output("run-prediction-button", "n_clicks"),
        Output("store-display-div", "style"),
        Output("model-probabilities-store", "data", allow_duplicate=True),
        Output("sensitivity-analysis-store", "data", allow_duplicate=True),
        Input("run-prediction-button", "n_clicks"),
        State("data-path-store", "data"),
        State("model-dropdown", "value"),
        State("venv", "value"),
        State("initial-threshold", "value"),
        State("sensitivity-analysis", "value"),
        State("adjust-onset", "value"),
        State("model-probabilities-store", "data"),
        State("sensitivity-analysis-store", "data"),
        State("channel-store", "data"),
        prevent_initial_call=True,
    )
    def _execute_predict_script(
        n_clicks,
        data_path,
        model_path,
        venv,
        threshold,
        sensitivity_analysis,
        adjust_onset,
        model_probabilities_store,
        sensitivity_analysis_store,
        channel_store,
    ):
        if not n_clicks or n_clicks == 0:
            return None, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Validation: Check if all required fields are filled
        if not data_path:
            error_message = "⚠️ Please choose a subject to display on Home page."
            return (
                error_message,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        missing_fields = []
        if not model_path:
            missing_fields.append("Model")
        if not venv:
            missing_fields.append("Environment")
        if threshold is None:
            missing_fields.append("Threshold")
        if missing_fields:
            error_message = (
                f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}"
            )
            return (
                error_message,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        cache_dir = config.CACHE_DIR
        predictions_csv_path = (
            cache_dir / f"{os.path.basename(model_path)}_predictions.csv"
        )
        smoothgrad_path = cache_dir / f"{os.path.basename(model_path)}_smoothGrad.pkl"

        # If already exists, skip execution
        if (
            predictions_csv_path.exists()
            and str(predictions_csv_path) in model_probabilities_store
        ):
            if (
                sensitivity_analysis
                and smoothgrad_path.exists()
                and str(smoothgrad_path) in model_probabilities_store
            ):
                return (
                    "✅ Reusing existing model predictions",
                    0,
                    {"display": "block"},
                    dash.no_update,
                    dash.no_update,
                )
            elif not sensitivity_analysis:
                return (
                    "✅ Reusing existing model predictions",
                    0,
                    {"display": "block"},
                    dash.no_update,
                    dash.no_update,
                )

        # Otherwise, execute model
        if "TensorFlow" in venv:
            ACTIVATE_ENV = str(config.TENSORFLOW_ENV / "bin/python")
        elif "PyTorch" in venv:
            ACTIVATE_ENV = str(config.TORCH_ENV / "bin/python")

        command = [
            ACTIVATE_ENV,
            str(config.MODEL_PIPELINE_DIR / "main.py"),
            str(model_path),
            str(venv),
            str(data_path),
            str(cache_dir),
            str(threshold),  # Ensure threshold is passed as a string
            str(adjust_onset),
            str(channel_store),
        ]

        working_dir = str(config.APP_ROOT)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(working_dir)

        try:
            start_time = time.time()
            subprocess.run(
                command, env=env, text=True, cwd=str(config.MODEL_PIPELINE_DIR)
            )  # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model_probabilities_store = [str(predictions_csv_path)]
            print(f"Model testing executed in {time.time()-start_time:.2f} seconds")

        except Exception as e:
            return (
                f"⚠️ Error running model: {e}",
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        if sensitivity_analysis:
            command = [
                ACTIVATE_ENV,
                str(config.MODEL_PIPELINE_DIR / "run_smoothgrad.py"),
                str(model_path),
                str(venv),
                str(cache_dir),
                str(predictions_csv_path),
                str(threshold),  # Ensure threshold is passed as a string
            ]

            try:
                # Start timing for the second subprocess
                start_time = time.time()
                subprocess.run(
                    command, env=env, text=True, cwd=str(config.MODEL_PIPELINE_DIR)
                )  # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Smoothgrad executed in {time.time()-start_time:.2f} seconds")

            except Exception as e:
                return (
                    f"⚠️ Error running smoothgrad: {e}",
                    0,
                    {"display": "block"},
                    model_probabilities_store,
                    dash.no_update,
                )

            sensitivity_analysis_store = [str(smoothgrad_path)]
            if not smoothgrad_path.exists():
                return (
                    "⚠️ Error running smoothgrad.",
                    0,
                    {"display": "block"},
                    model_probabilities_store,
                    dash.no_update,
                )
            return (
                True,
                0,
                {"display": "block"},
                model_probabilities_store,
                sensitivity_analysis_store,
            )

        if not predictions_csv_path.exists():
            return (
                "⚠️ Error running model.",
                0,
                {"display": "none"},
                dash.no_update,
                dash.no_update,
            )
        return True, 0, {"display": "block"}, model_probabilities_store, dash.no_update


@callback(
    Output("prediction-output-summary-div", "children"),
    Output("prediction-output-distribution-div", "children"),
    Output("prediction-output-table-div", "children"),
    Input("store-display-div", "style"),
    Input("adjusted-threshold", "value"),
    State("model-probabilities-store", "data"),
    prevent_initial_call=True,
)
def update_prediction_table(style, threshold, prediction_csv_path):
    if style["display"] == "none":
        return None, None, None
    if not prediction_csv_path or threshold is None:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        df = pd.read_csv(prediction_csv_path[0])
        df_filtered = df[df["probas"] > threshold]
        df_filtered["probas"] = df_filtered["probas"].round(2)

        if df_filtered.empty:
            msg = html.P("No events found in this recording.")
            return msg, msg, msg

        table = dbc.Table.from_dataframe(
            df_filtered.copy(),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            dark=False,
        )

        return (
            au.build_table_prediction_statistics(df_filtered, len(df)),
            au.build_prediction_distribution_statistics(df, threshold),
            table,
        )
    except Exception as e:
        error_msg = html.P(f"⚠️ Error loading predictions: {e}")
        return error_msg, error_msg, error_msg


@callback(
    Output("model-spike-name", "value"),
    Input("model-dropdown", "value"),
    Input("adjusted-threshold", "value"),
    prevent_initial_call=True,
)
def update_spike_name(model_path, threshold_value):
    if model_path is None or threshold_value is None:
        return dash.no_update
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return f"{model_name}_{threshold_value}"


def register_store_display_prediction():
    @callback(
        Output("annotation-store", "data", allow_duplicate=True),
        Output("sidebar-tabs", "active_tab"),
        Output("store-display-div", "style", allow_duplicate=True),
        Output("history-store", "data"),
        Input("store-display-button", "n_clicks"),
        State("annotation-store", "data"),
        State("model-probabilities-store", "data"),
        State("adjusted-threshold", "value"),
        State("model-spike-name", "value"),
        State("history-store", "data"),
        prevent_initial_call=True,
    )
    def store_display_prediction(
        n_clicks,
        annotation_data,
        prediction_csv_path,
        threshold,
        spike_name,
        history_data,
    ):
        if not n_clicks or n_clicks == 0 or prediction_csv_path is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if not annotation_data:
            annotation_data = []

        df = pd.read_csv(prediction_csv_path[0])
        prediction_df = df[df["probas"] > threshold]
        new_annotations = prediction_df[["onset", "duration"]].copy()
        new_annotations["description"] = spike_name  # Set spike name as description
        new_annotations_dict = new_annotations.to_dict(orient="records")
        annotation_data.extend(new_annotations_dict)

        action = f"Tested model with <{spike_name}> as the predicted event name.\n"
        history_data = hu.fill_history_data(history_data, "models", action)

        # Return updated annotations and switch tab
        return annotation_data, "selection-tab", {"display": "none"}, history_data
