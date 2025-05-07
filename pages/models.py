import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from layout import INPUT_STYLES, BOX_STYLES, FLEXDIRECTION
import pandas as pd
from callbacks.utils import performance_utils as pu
from callbacks.utils import annotation_utils as au
import callbacks.utils.model_utils as mu
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
						persistence_type="memory",
						style={"margin": "10px 0", "fontSize": "14px"}
					)
				], style={"padding": "10px"}),

				dbc.Tab(label="Ground Truth", children=[
					dbc.Checklist(
						id="ground-truth-checkboxes",
						inline=False,
						persistence=True,
						persistence_type="memory",
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

			html.Hr(),

			html.Div([
				html.Div(
					children=[
						dbc.RadioItems(
							id="panel-selector-id",
							options=[
								{"label": "Panel 1", "value": 0},
								{"label": "Panel 2", "value": 1},
								{"label": "Panel 3", "value": 2}
							],
							value=0,
							inline=True,
							labelStyle={"marginRight": "10px"},
							style={"fontSize": "0.9rem"}
						)
					]
				),
				dbc.Button(
					"Compute performances",
					id="compute-performances-button",
					color="success",
					disabled=True,
					n_clicks=0
				)
			], style={"display": "flex", "alignItems": "center", "marginBottom": "20px"})

		], style={**BOX_STYLES["classic"], "width": "40%"}),

		html.Div(
			id="performance-results-div",
			children=[
				html.Div([
					html.Div(id=f"performance-results-title-panel-{i}", children=[html.Small(f"Panel {i}")], style= {"marginBottom": "10px"}),
					dbc.Tabs([
						dbc.Tab(label="Performance Metrics", tab_id=f"metrics-panel-{i}", children=[
							html.Div(id=f"performance-metrics-table-panel-{i}")
						]),
						dbc.Tab(label="Confusion Matrix", tab_id=f"confusion-panel-{i}", children=[
							html.Div(id=f"confusion-matrix-table-panel-{i}")
						]),
						dbc.Tab(label="Distance Statistics", tab_id=f"stats-panel-{i}", children=[
							html.I(
								className="bi bi-info-circle-fill",
								id="distance-help-icon",
								style={
									"fontSize": "0.8em",
									"cursor": "pointer",
									"verticalAlign": "middle",
									"marginLeft": "15px"
								}
							),
							dbc.Tooltip(
								"This panel shows distances between model predictions and ground truth."
								"• TP: Time between matched prediction and true event. Smaller values mean more accurate predictions."
								"• FP: Time to nearest true event (unmatched prediction). Useful for spotting near-misses or totally spurious predictions."
								"• FN: Time to nearest prediction (missed true event). A small distance means the model nearly detected the event.",
								target="distance-help-icon",
								placement="top",
								style={"maxWidth": "600px"}
							),
							html.Div(id=f"distance-statistics-table-panel-{i}")				 
						])
					],
					id=f"performance-tabs-panel-{i}",
					active_tab=f"metrics-panel-{i}",
					style={**FLEXDIRECTION['row-tabs']}
					)
				],
				id=f"performance-panel-{i}",
				style={"marginBottom": "30px"})  # space between panels
				for i in range(3)
			],
			style={**BOX_STYLES["classic"], "width": "60%"}
		)
	], style={**FLEXDIRECTION['row-flex'], "display": "flex"})
])

@callback(
	Output("model-prediction-radio", "options"),
	Output("ground-truth-checkboxes", "options"),
	Input("annotation-store", "data"),
	prevent_initial_call = False
)
def display_model_names_checklist(annotations_store):
	description_counts = au.get_annotation_descriptions(annotations_store)
	options = [{'label': f"{name} ({count})", 'value': f"{name}"} for name, count in description_counts.items()]
	value = [f"{name}" for name in description_counts.keys()]
	return options, options #dash.no_update  # Set all annotations as default selected

@callback(
	Output("compute-performances-button", "disabled"),
	[
		Input("model-prediction-radio", "value"),
		Input("ground-truth-checkboxes", "value"),
	]
)
def toggle_compute_button(model_prediction, ground_truth):
	if not model_prediction or not ground_truth:
		return True
	if isinstance(ground_truth, list) and model_prediction not in ground_truth:
		return False  # Enable button
	return True  # Otherwise, keep it disabled

@callback(
	Output("panel-selector-id", "value"),
	*[
		Output(component_id, "children")
		for i in range(3)
		for component_id in [
			f"performance-results-title-panel-{i}",
			f"confusion-matrix-table-panel-{i}",
			f"performance-metrics-table-panel-{i}",
			f"distance-statistics-table-panel-{i}"
		]
	],
	Input("compute-performances-button", "n_clicks"),  # Trigger when button is clicked
	State("model-prediction-radio", "value"),  # Selected model prediction
	State("ground-truth-checkboxes", "value"),  # Selected ground truth(s)
	State("performance-tolerance", "value"),  # User-defined delta threshold
	State("performance-threshold", "value"),  # User-defined delta threshold
	State("panel-selector-id", "value"),
	State("annotation-store", "data"),
	prevent_initial_call=True  # Don't trigger on page load
)
def compute_performance(n_clicks, model_prediction, ground_truth, tolerance, threshold, panel_index, annotations):
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

	true_positive, false_positive, false_negative, tp_dists, fp_dists, fn_dists = pu.compute_matches(model_onsets, gt_onsets, delta)

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

	conf_matrix = dbc.Table(
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
	perf_metrics = dbc.Table(
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
	  
	# Distance Statistics Data
	tp_stats = pu.get_distance_stats(tp_dists)
	fp_stats = pu.get_distance_stats(fp_dists)
	fn_stats = pu.get_distance_stats(fn_dists)

	distance_stats_data = [
		{"Type": "TP", "Mean": tp_stats["mean"], "Std": tp_stats["std"], "Min": tp_stats["min"], "Max": tp_stats["max"]},
		{"Type": "FP", "Mean": fp_stats["mean"], "Std": fp_stats["std"], "Min": fp_stats["min"], "Max": fp_stats["max"]},
		{"Type": "FN", "Mean": fn_stats["mean"], "Std": fn_stats["std"], "Min": fn_stats["min"], "Max": fn_stats["max"]},
	]

	distance_stats = dbc.Table(
		[
			html.Thead(html.Tr([html.Th("Type"), html.Th("Mean"), html.Th("Std"), html.Th("Min"), html.Th("Max")])),
			html.Tbody([
				html.Tr([html.Td(row["Type"]), html.Td(row["Mean"]), html.Td(row["Std"]), html.Td(row["Min"]), html.Td(row["Max"])])
				for row in distance_stats_data
			])
		],
		bordered=True,
		hover=True,
		responsive=True,
		striped=True,
	)

	title =  html.Div([
		html.Small([
			html.I(className="bi bi-flag-fill me-2 text-primary"),  # icon for ground truth
			f"Ground Truth: ",
			html.Span(ground_truth, className="fw-bold text-dark"),
			html.I(className="bi bi-stars mx-3 text-success"),   # icon for prediction
			f"Prediction: ",
			html.Span(model_prediction, className="fw-bold text-dark"),
			html.I(className="bi bi-sliders2-vertical mx-3 text-warning"),  # icon for parameters
			f"Tolerance: ",
			html.Span(f"{tolerance} ms", className="fw-bold text-dark"),
			html.Span(" | ", className="mx-2"),
			f"Threshold: ",
			html.Span(f"{threshold}", className="fw-bold text-dark"),
	], className="text-muted")])
	
	if panel_index < 2:
		output_list = [panel_index+1]
	else:
		output_list= [0]
	for i in range(3):
		if i == panel_index:
			output_list.extend([title, conf_matrix, perf_metrics, distance_stats])
		else:
			output_list.extend([dash.no_update] * 4)

	return tuple(output_list)