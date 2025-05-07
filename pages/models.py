#predict.py
import dash
from dash import html, dcc,  dash_table, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from layout import INPUT_STYLES, BOX_STYLES, FLEXDIRECTION
import pandas as pd
import os
import subprocess
import config
from callbacks.utils import predict_utils as pu
from callbacks.utils import annotation_utils as au
import callbacks.utils.model_utils as mu
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

dash.register_page(__name__, name = "Model performance", path="/model/performance")

layout = html.Div([
    dcc.Location(id="url", refresh=True),

    dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H4([
                    "Your Models",
                    html.I(
                        className="bi bi-info-circle-fill",
                        id="models-help-icon",
                        style={
                            "fontSize": "0.8em",
                            "cursor": "pointer",
                            "verticalAlign": "middle",
                            "marginLeft": "15px"
                        }
                    )
                ], style={"margin": "0px"}),

                dbc.Tooltip(
                    "Here you can see your pretrained models and, if applicable, compute performance using already stored predictions.",
                    target="models-help-icon",
                    placement="right"
                )
            ], style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "gap": "20px",
                "margin": "30px"
            }),

            mu.render_pretrained_models_table()
        ]),
        className="mb-3",
        style={"width": "100%"}
    ),

    html.Div(id="model-playground", children=[
        html.Div([
            dbc.Tabs([
                dbc.Tab(label="Model Prediction", children=[
                    dbc.RadioItems(
                        id="model-prediction-radio",
                        value="Yes",
                        inline=False,
                        persistence=True,
                        persistence_type="local",
                        style={"margin": "10px 0", "fontSize": "14px"}
                    )
                ], style={"padding": "10px"}),

                dbc.Tab(label="Ground Truth", children=[
                    dbc.Checklist(
                        id="ground-truth-checkboxes",
                        inline=False,
                        persistence=True,
                        persistence_type="local",
                        style={"margin": "10px 0", "fontSize": "14px"}
                    )
                ], style={"padding": "10px"}),

                dbc.Tab(label="Performance Settings", children=[
                    html.Label("Tolerance (ms):", style={"fontWeight": "bold", "fontSize": "14px"}),
                    dbc.Input(id="performance-tolerance", type="number", value=200, step=10, min=0, max=1000, style=INPUT_STYLES["number"]),

                    html.Label("Threshold:", style={"fontWeight": "bold", "fontSize": "14px"}),
                    dbc.Input(id="performance-threshold", type="number", value=0.5, step=0.01, min=0, max=1, style=INPUT_STYLES["number"])
                ], style={"padding": "10px"})
            ], style=FLEXDIRECTION['row-tabs']),

            dbc.Button(
                "Compute performances",
                id="compute-performances-button",
                color="success",
                disabled=True,
                n_clicks=0
            )
        ], style={**BOX_STYLES["classic"], "width": "40%"}),

        html.Div(id="performance-results-div", children=[
            html.Div([
                html.Div([
                    html.H4("Confusion Matrix"),
                    html.Div(id="confusion-matrix-table")
                ], style={"marginBottom": "20px"}),

                html.Div([
                    html.H4("Performance Metrics"),
                    html.Div(id="performance-metrics-table")
                ], style={"marginTop": "20px"})
            ])
        ], style={**BOX_STYLES["classic"], "display": "none", "width": "50%"})
    ], style={**FLEXDIRECTION['row-flex'], "display": "flex"})
])

@callback(
	Output("model-prediction-radio", "options"),
	Output("model-prediction-radio", "value"),
	Output("ground-truth-checkboxes", "options"),
	Output("ground-truth-checkboxes", "value"),
	Input("annotation-store", "data"),
	prevent_initial_call = False
)
def display_model_names_checklist(annotations_store):
	description_counts = au.get_annotation_descriptions(annotations_store)
	options = [{'label': f"{name} ({count})", 'value': f"{name}"} for name, count in description_counts.items()]
	value = [f"{name}" for name in description_counts.keys()]
	pred_options =  options + [{'label': html.Span("run prediction", style={'color': 'blue', 'fontWeight': 'bold'}), 'value': f"run prediction"}]
	return pred_options, "run prediction", options, dash.no_update  # Set all annotations as default selected

@callback(
	Output("compute-performances-button", "disabled"),
	[
		Input("model-prediction-radio", "value"),
		Input("ground-truth-checkboxes", "value"),
	]
)
def toggle_compute_button(model_prediction, ground_truth):
	# Ensure all required inputs exist
	if not model_prediction or not ground_truth:
		return True  # Keep button disabled if any input is missing

	# Ensure at least one ground truth is selected and it's different from model prediction
	if isinstance(ground_truth, list) and model_prediction not in ground_truth:
		return False  # Enable button

	return True  # Otherwise, keep it disabled

def compute_matches(model_onsets, gt_onsets, delta):
	true_positive = 0
	false_positive = 0
	false_negative = 0
	
	matched_gt = set()  # To track which ground truth values have been matched

	# Count true positives: each model prediction must match one unique ground truth
	for m in model_onsets:
		# Check if model prediction m matches any ground truth g within the delta
		matched = False
		for g in gt_onsets:
			if abs(m - g) <= delta and g not in matched_gt:
				true_positive += 1
				matched_gt.add(g)  # Mark this ground truth as matched
				matched = True
				break
		if not matched:
			false_positive += 1  # If no match, it's a false positive

	# Count false negatives: ground truth not matched by any model prediction
	for g in gt_onsets:
		if g not in matched_gt:
			false_negative += 1  # This ground truth has no matching model prediction

	return true_positive, false_positive, false_negative

@callback(
	Output("url", "pathname", allow_duplicate=True),
	# Output("sidebar-tabs", "active_tab", allow_duplicate=True),
	Output("performance-results-div", "style"),  # Output component
	Output("confusion-matrix-table", "children"),
	Output("performance-metrics-table", "children"),  # Output component
	Input("compute-performances-button", "n_clicks"),  # Trigger when button is clicked
	State("model-prediction-radio", "value"),  # Selected model prediction
	State("ground-truth-checkboxes", "value"),  # Selected ground truth(s)
	State("performance-tolerance", "value"),  # User-defined delta threshold
	State("performance-threshold", "value"),  # User-defined delta threshold
	State("annotation-store", "data"),
	prevent_initial_call=True  # Don't trigger on page load
)
def compute_performance(n_clicks, model_prediction, ground_truth, tolerance, threshold, annotations):
	if not model_prediction or not ground_truth or tolerance is None:
		return dash.no_update, "Error: Missing inputs. Please select model predictions, ground truth, and delta.", dash.no_update, dash.no_update
	
	if model_prediction == "run prediction":
		return "/viz/raw-signal", dash.no_update, dash.no_update, dash.no_update
	
	# Convert annotations to DataFrame
	annotations_df = pd.DataFrame(annotations).set_index("onset")  # Ensure onset is the index

	# Filter annotations for selected model prediction and ground truth
	model_onsets = au.get_annotations(model_prediction, annotations_df)  # Example: [0.5, 1.2, 2.3, ...]
	gt_onsets = au.get_annotations(ground_truth, annotations_df)  # Example: [0.6, 1.1, 2.5, ...]

	delta = tolerance/1000

	# # Compute matches (True Positive, False Positive, False Negative)
	# true_positive = sum(any(abs(m - g) <= delta for g in gt_onsets) for m in model_onsets)
	# false_positive = len(model_onsets) - true_positive
	# false_negative = sum(not any(abs(m - g) <= delta for m in model_onsets) for g in gt_onsets)
	true_positive, false_positive, false_negative = compute_matches(model_onsets, gt_onsets, delta)

	# Compute Precision, Recall, F1-score
	precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
	recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
	f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

	# Confusion Matrix Data
	conf_matrix_data = {
		"Metric": ["True Positive", "False Positive", "False Negative", "True Negative"],
		"Count": [true_positive, false_positive, false_negative, "nan"]
	}
	conf_matrix_data = [
		{"": "Actual Negative", "Predicted Negative": "nan", "Predicted Positive": false_positive},
		{"": "Actual Positive", "Predicted Negative": false_negative, "Predicted Positive": true_positive}
	]

	# Performance Metrics Data
	perf_metrics_data = {
		"Metric": ["Precision", "Recall", "F1 Score"],
		"Value": [f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]
	}

	confusion_matrix_data = dbc.Table(
		[
			html.Thead(html.Tr([html.Th(""), html.Th("Predicted Negative"), html.Th("Predicted Positive")])),
			html.Tbody([
				html.Tr([html.Td(row[""]), html.Td(row["Predicted Negative"]), html.Td(row["Predicted Positive"])])
				for row in conf_matrix_data
			])
		],
		bordered=True,
		hover=True,
		responsive=True,
		striped=True,
	)

	# Convert performance metrics data to dbc.Table
	performance_metrics_data = dbc.Table(
		[
			html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
			html.Tbody([
				html.Tr([html.Td(perf_metrics_data["Metric"][i]), html.Td(perf_metrics_data["Value"][i])])
				for i in range(len(perf_metrics_data["Metric"]))
			])
		],
		bordered=True,
		hover=True,
		responsive=True,
		striped=True,
	)

	return dash.no_update, {**BOX_STYLES["classic"], "width": "60%", "display": "flex"}, confusion_matrix_data, performance_metrics_data
