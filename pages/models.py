#predict.py
import dash
from dash import html, dcc,  dash_table, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from layout import input_styles, box_styles
import pandas as pd
import os
import subprocess
import static.constants as c
from callbacks.utils import predict_utils as pu
from callbacks.utils import annotation_utils as au
import callbacks.utils.graph_utils as gu
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

dash.register_page(__name__, name = "Model performance", path="/model/performance")

layout = html.Div([

	html.Div([
		html.Div(
			id="your-models-container",
			children=[
				html.Div(children=[
					html.H1([
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

	html.Div([
		html.H2("Model Parameters"),
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
	], style=box_styles["classic"]),
	
	html.Div([
		html.H2("Model Performances"),
		dbc.Checklist(
			id="performance-annotation-checkboxes",
			inline=False,
			style={"margin": "10px 0", "fontSize": "12px"},
			persistence=True,
			persistence_type="local"
		),
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
			dbc.Row([
				dbc.Col(
					html.Div([
						html.Label(
							"Model prediction:",
							style={"fontWeight": "bold", "fontSize": "14px"}
						),
						dbc.RadioItems(
							id="model-prediction-radio",
							value="Yes",  # Default selection
							inline=False,  # Display buttons in a row
							persistence=True,
							persistence_type="local",
							style={"margin": "10px 0", "fontSize": "12px"}
						)
					], style = box_styles["classic"]),
					width=2
				),
				dbc.Col(
					html.Div([
						html.Label(
							"Ground truth:",
							style={"fontWeight": "bold", "fontSize": "14px"}
						),
						dbc.Checklist(
							id="ground-truth-checkboxes",
							inline=False,
							style={"margin": "10px 0", "fontSize": "12px"},
							persistence=True,
							persistence_type="local"
						)
					], style = box_styles["classic"]),
					width=2
				)
			]),

		], style={"padding": "10px"}),

		html.Div([
			dbc.Row([
				dbc.Col(
					html.Div([
						html.Label(
							"Tolerance (ms):",
							style={"fontWeight": "bold", "fontSize": "14px"}
						),
						dbc.Input(id="performance-tolerance", type="number", value=200, step=10, min=0, max=1000, style=input_styles["small-number"]),
					], style = box_styles["classic"]),
					width=1
				),
				dbc.Col(
					html.Div([
						html.Label(
							"Threshold (if test model):",
							style={"fontWeight": "bold", "fontSize": "14px"}
						),
						dbc.Input(id="performance-threshold", type="number", value=0.5, step=0.01, min=0, max=1, style=input_styles["small-number"]),
					], style = box_styles["classic"]),
					width=2
				)
			])
		], style={"padding": "10px"}),

		# Div to toggle visibility of tables
		html.Div(
			id="performance-results-div",
			children=[
				html.Div([
					# Use a row div to create a 2-column layout
					dbc.Row(
						[
							dbc.Col(
								# First column for Confusion Matrix
								html.Div([
									html.H4("Confusion Matrix"),
									dash_table.DataTable(
										id="confusion-matrix-table",
										columns=[
											{"name": "", "id": ""},
											{"name": "Predicted Negative", "id": "Predicted Negative"},
											{"name": "Predicted Positive", "id": "Predicted Positive"}
										],
										style_table={"overflowX": "auto"},
										style_cell={"textAlign": "center", "padding": "8px"},
										style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
										style_cell_conditional=[
											# Style for the header of the empty column to make it bold
											{
												'if': {'column_id': ''},
												'fontWeight': 'bold',
												'backgroundColor': '#f0f0f0',  # Set a background color if needed
												'color': 'black'
											}],
										data=[]  # Empty data initially, will be updated
									)
								], style = {"font-size": "14px"}),
							width = 3),
							dbc.Col(
								# Second column for Performance Metrics
								html.Div([
									html.H4("Performance Metrics"),
									dash_table.DataTable(
										id="performance-metrics-table",
										columns=[
											{"name": "Metric", "id": "Metric"},
											{"name": "Value", "id": "Value"}
										],
										style_table={"overflowX": "auto"},
										style_cell={"textAlign": "center", "padding": "8px"},
										style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
										style_cell_conditional=[
											# Style for the header of the empty column to make it bold
											{
												'if': {'column_id': 'Metric'},
												'fontWeight': 'bold',
												'backgroundColor': '#f0f0f0',  # Set a background color if needed
												'color': 'black'
											}],
										data=[]  # Empty data initially, will be updated
									)
								], style = {"font-size": "14px"}),
							width = 2),
						]  # Display as a flex row
					)
				])
			],
			style={"display": "none"})  # Initially hidden			

	], style=box_styles["classic"]),

	# Explanation of what the user needs to do
	html.Div([
		html.H2("Edit Models"),
		html.P("Here you can create your models before training them."),
		html.Div([
			dbc.Row([
				dbc.Col(
					dbc.Input(
						id="new-model-name",
						type="text",
						placeholder="Model name...",
						style=input_styles["path"]
					),
					width=4
				),
				dbc.Col(
					dbc.Button(
					"Create",
					id="create-model-button",
					color="success",
					disabled=True,
					n_clicks=0
					)
				)
			])
		], style={"padding": "10px"}),
	], style=box_styles["classic"])

])

# Function to run the model analysis in the correct environment
def run_model_info(model_name):
	model_path = os.path.join(c.MODEL_DIR, model_name)
	
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
	pred_options =  options + [{'label': html.Span("test model", style={'color': 'blue', 'fontWeight': 'bold'}), 'value': f"test model"}]
	return pred_options, "test model", options, dash.no_update  # Set all annotations as default selected

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
	Output("performance-results-div", "style"),  # Output component
	Output("confusion-matrix-table", "data"),
	Output("performance-metrics-table", "data"),  # Output component
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
		return "Error: Missing inputs. Please select model predictions, ground truth, and delta."
	
	# Convert annotations to DataFrame
	annotations_df = pd.DataFrame(annotations).set_index("onset")  # Ensure onset is the index

	print("annotations_df", annotations_df)

	print("model_prediction", model_prediction)
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

	# # Compute performance metrics
	# precision = precision_score(y_true, y_pred, zero_division=0)
	# recall = recall_score(y_true, y_pred, zero_division=0)
	# f1 = f1_score(y_true, y_pred, zero_division=0)

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

	# Display results
	results = f"""
	**Performance Metrics:**
	- **Precision:** {precision:.3f}
	- **Recall:** {recall:.3f}
	- **F1 Score:** {f1:.3f}
	"""

	# Create Dash tables
	confusion_matrix_df = pd.DataFrame(conf_matrix_data)
	performance_metrics_df = pd.DataFrame(perf_metrics_data)

	print(confusion_matrix_df)

	confusion_matrix = dash_table.DataTable(
				id='confusion-matrix-table',
				columns=[{"name": col, "id": col} for col in confusion_matrix_df.columns],
				data=confusion_matrix_df.to_dict('records'),
				style_table={"overflowX": "auto"},
				style_cell={"textAlign": "center", "padding": "8px"},
				style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
			)
	
	performance_metrics = dash_table.DataTable(
				id='performance-metrics-table',
				columns=[{"name": col, "id": col} for col in performance_metrics_df.columns],
				data=performance_metrics_df.to_dict('records'),
				style_table={"overflowX": "auto"},
				style_cell={"textAlign": "center", "padding": "8px"},
				style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
			)
	
	# Confusion Matrix Data
	conf_matrix_data = [
		{"": "Actual Negative", "Predicted Negative": "nan", "Predicted Positive": false_positive},
		{"": "Actual Positive", "Predicted Negative": false_negative, "Predicted Positive": true_positive}
	]

	# Performance Metrics Data
	perf_metrics_data = [
		{"Metric": "Precision", "Value": f"{precision:.3f}"},
		{"Metric": "Recall", "Value": f"{recall:.3f}"},
		{"Metric": "F1 Score", "Value": f"{f1:.3f}"}
	]

	return {"display": "block"}, conf_matrix_data, perf_metrics_data
