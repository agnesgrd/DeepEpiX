#predict.py
import dash
from dash import html, dcc,  dash_table, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from layout import input_styles, box_styles
import pandas as pd
import os
import subprocess
import config
from callbacks.utils import predict_utils as pu
from callbacks.utils import annotation_utils as au
import callbacks.utils.graph_utils as gu
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

dash.register_page(__name__, name = "Model performance", path="/model/performance")

layout = html.Div([

	dcc.Location(id="url", refresh=True), 

	html.Div([
		html.Div(
			id="your-models-container",
			children=[
				html.Div(children=[
					html.H3([
						"Your Models",
						html.I(
							className="bi bi-info-circle-fill", id="models-help-icon", style={
											"fontSize": "0.8em",
											"cursor": "pointer",
											"verticalAlign": "middle",
											"marginLeft": "15px"
										})
									], style={"margin": "0px"}),
					# Section to display saved montages
										# Tooltip for the info icon
					dbc.Tooltip(
						"Here you can see your pretrained models and, if applicable, compute performance using already stored predictions.",
						target="models-help-icon",
						placement="right"
					),
                ], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "gap": "20px", "margin": "30px"}),

				dash.dash_table.DataTable(
					id="pretrained-models-table",
					columns=[{"name": col, "id": col} for col in pd.read_csv("static/pretrained-models-info.csv")],
					data=pd.read_csv("static/pretrained-models-info.csv").to_dict('records'),  # To be populated with saved montages
					page_size=10,
					style_table={
						"overflowX": "auto", 
						"width": "50%",  # Reduce table width
						"margin": "0 auto"  # Center the table on the page
					},
					style_cell={
						"textAlign": "center",  # Center text in all cells
						"padding": "5px",  # Add padding for better readability
						"fontFamily": "Arial, sans-serif",  # Consistent font
						"fontSize": "14px",  # Adjust font size
						"border": "1px solid #ddd"  # Add light borders
					},
					style_header={
						"fontWeight": "bold",  # Bold headers
						"textAlign": "center",  # Center text in headers
						"backgroundColor": "#f4f4f4",  # Light background for headers
						"borderBottom": "2px solid #ccc"  # Slightly thicker bottom border
					}
				)
			],
			style=box_styles["classic"]
		),
	]),

	html.Div(id="model-playground", children = [
		
		html.Div([
			html.H4("Model Performances"),
			html.Div([
				dbc.Row([
					dbc.Col(
						html.Div([
							dcc.Dropdown(
								id="performance-pretrained-models-dropdown",
								options=pu.get_model_options("all"),
								placeholder="Select ...",
								style = {"marginBottom": "8px"}
							),
						]),
						width=4
					),
					dbc.Col(
						dbc.Button(
							"Compute performances",
							id="compute-performances-button",
							color="success",
							disabled=True,
							n_clicks=0
						),
					)
				])
			], style={"padding": "10px"}),



			html.Div([
				dbc.Tabs([
					dbc.Tab(
						label="Model Prediction",
						children=[
							html.Div([
								dbc.RadioItems(
									id="model-prediction-radio",
									value="Yes",  # Default selection
									inline=False,  # Display buttons in a row
									persistence=True,
									persistence_type="local",
									style={"margin": "20px 0", "fontSize": "14px", "padding":"10px"}
								)
							], style = {"padding": "10px"})
						]
					),
					
					dbc.Tab(
						label="Ground Truth",
						children=[
							html.Div([
								dbc.Checklist(
									id="ground-truth-checkboxes",
									inline=False,
									style={"margin": "10px 0", "fontSize": "14px"},
									persistence=True,
									persistence_type="local"
								)
							], style = {"padding": "10px"}),
						]
					),
					
					dbc.Tab(
						label="Performance Settings",
						children=[
							html.Div([
								html.Label(
									"Tolerance (ms):",
									style={"fontWeight": "bold", "fontSize": "14px"}
								),
								dbc.Input(id="performance-tolerance", type="number", value=200, step=10, min=0, max=1000, style=input_styles["number"]),

								html.Label(
									"Threshold:",
									style={"fontWeight": "bold", "fontSize": "14px"}
								),
								dbc.Input(id="performance-threshold", type="number", value=0.5, step=0.01, min=0, max=1, style=input_styles["number"]),
							], style = {"padding": "10px"}),
						]
					),
										
				], style={
					"display": "flex",
					"flexDirection": "row",  # Ensure tabs are displayed in a row (horizontal)
					"alignItems": "center",  # Center the tabs vertically within the parent container
					"width": "50%",  # Full width of the container
					"borderBottom": "1px solid #ddd"  # Optional, adds a bottom border to separate from content
				})
			]),
			
			html.Hr(),

			# Placeholder divs for the tables, will be defined in the callback
			html.Div(
				id="performance-results-div",
				children=[
					# Tab 1: Confusion Matrix Table
					html.Div([
						html.H4("Confusion Matrix"),  # Title for the Confusion Matrix table
						html.Div(id="confusion-matrix-table"),  # Placeholder for Confusion Matrix table
					], style={"marginBottom": "20px"}),  # Add margin between sections
					
					# Tab 2: Performance Metrics Table
					html.Div([
						html.H4("Performance Metrics"),  # Title for the Performance Metrics table
						html.Div(id="performance-metrics-table"),  # Placeholder for Performance Metrics table
					], style={"marginTop": "20px"})  # Add margin between sections

				],
				style={"display": "none", "marginTop": "20px"}  # Initially hidden, margin for spacing
			),
	

		], style={
			"padding": "15px",
			"border": "1px solid #ddd",
			"borderRadius": "8px",
			"boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
			"width": "60%"}  # Adjust the width as needed,
		),

		html.Div([
			html.H4("Model Parameters"),
			html.Div([
				dbc.Row([
					dbc.Col(
						dcc.Dropdown(
							id="params-pretrained-models-dropdown",
							options=pu.get_model_options("all"),
							placeholder="Select ...",
						),
						width=4
					),
					dbc.Col(
						dbc.Button(
							"More info",
							id="more-info-button",
							color="success",
							disabled=True,
							n_clicks=0
						),
					)
				])
			], style={"padding": "10px"}),

			html.Pre(id="model-info", style={"display": "none"})
		], style={
			"padding": "15px",
			"border": "1px solid #ddd",
			"borderRadius": "8px",
			"boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
			"width": "40%"}  # Adjust the width as needed,
		),
        
	], style = {
			"display": "flex",
            "flexDirection": "row",  # Side-by-side layout
            "alignItems": "flex-start",  # Align to top
            "gap": "20px",  # Add spacing between elements
            "width": "100%"  # Ensure full width

        }),

	# # Explanation of what the user needs to do
	# html.Div([
	# 	html.H2("Edit Models"),
	# 	html.P("Here you can create your models before training them."),
	# 	html.Div([
	# 		dbc.Row([
	# 			dbc.Col(
	# 				dbc.Input(
	# 					id="new-model-name",
	# 					type="text",
	# 					placeholder="Model name...",
	# 					style=input_styles["path"]
	# 				),
	# 				width=4
	# 			),
	# 			dbc.Col(
	# 				dbc.Button(
	# 				"Create",
	# 				id="create-model-button",
	# 				color="success",
	# 				disabled=True,
	# 				n_clicks=0
	# 				)
	# 			)
	# 		])
	# 	], style={"padding": "10px"}),
	# ], style=box_styles["classic"])

])

# Function to run the model analysis in the correct environment
def run_model_info(model_name):
	model_path = os.path.join(config.MODEL_DIR, model_name)
	
	if model_name.endswith((".h5", ".keras")):
		# Activate TensorFlow venv and run script
		command = [f"{c.TENSORFLOW_ENV}/bin/python",f"model_pipeline/tensorflow_model_info.py" ,f"{model_path}"]
	elif model_name.endswith(".pth"):
		# Activate PyTorch venv and run script
		command = [f"{c.TORCH_ENV}/bin/python",f"model_pipeline/pytorch_model_info.py" ,f"{model_path}"]
		# return "Functionality not available for Pytorch models."
	else:
		return "Unknown model format."

	try: 
		# Using Popen to display output in real time
		process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

		# Wait for process to complete
		process.wait()
		full_output = str(process.stdout.read())

		# Capture any error messages
		stderr_output = process.stderr.read()
		if stderr_output:
			print("Error:", stderr_output)
		# return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
	
	except Exception as e:
		return f"Error running model: {e}"
	
	return full_output
	
@callback(
	Output("more-info-button", "disabled"),
	Input("params-pretrained-models-dropdown", "value"),
	prevent_initial_call=True
)
def display_more_info_button(selected_rows):
	if selected_rows:
		return False
	dash.no_update

# Callback to Display Model Info
@callback(
	Output("model-info", "children"),
	Output("model-info", "style"),
	Input("more-info-button", "n_clicks"),
	State("params-pretrained-models-dropdown", "value"),
	prevent_initial_call=True
)
def display_model_info(n_clicks, selected_model):
	if not selected_model or n_clicks == 0:
		return dash.no_update, dash.no_update
	model_name = selected_model
	return run_model_info(model_name), {'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word', 'fontFamily': 'monospace', 'fontSize': '13px', 'backgroundColor': '#f4f4f4', 'padding': '10px'}

@callback(
	Output("model-prediction-radio", "options"),
	Output("model-prediction-radio", "value"),
	Output("ground-truth-checkboxes", "options"),
	Output("ground-truth-checkboxes", "value"),
	Input("annotations-store", "data"),
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
		Input("performance-pretrained-models-dropdown", "value"),
	]
)
def toggle_compute_button(model_prediction, ground_truth, selected_model):
	# Ensure all required inputs exist
	if not model_prediction or not ground_truth or not selected_model:
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
	State("annotations-store", "data"),
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

	return dash.no_update, {"display": "block"}, confusion_matrix_data, performance_metrics_data
