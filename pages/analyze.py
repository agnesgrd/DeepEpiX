# analyze.py: Analyze Page
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import mne  # Import the MNE package
import plotly.graph_objs as go  # Import Plotly for graphing
from layout import input_styles
import numpy as np

dash.register_page(__name__)

layout = html.Div([

    html.Div([
        html.H2("Your montage"),
        # Section to display saved montages
        html.Div(
            id="your-montage-container",
            children=[
                html.P("Here you can see which montage you have already created.", 
                    style={"marginBottom": "10px"}),
                dash.dash_table.DataTable(
                    id="saved-montages-table",
                    columns=[
                        {"name": "Montage Name", "id": "montage_name"},
                        {"name": "Channels", "id": "channels"},
                        {"name": "Actions", "id": "actions", "presentation": "markdown"},
                    ],
                    data=[],  # To be populated with saved montages
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
            ],  # This closes the `children` list
            style={
                "padding": "15px", 
                "backgroundColor": "#fff", 
                "border": "1px solid #ddd",
                "borderRadius": "8px", 
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", 
                "marginBottom": "20px"
            }
        )
    ]),
    
    # Explanation of what the user needs to do
    html.Div([
        html.H2("Edit Montage"),
        html.P("Here you can create your montage, give a name and pick it for visualization."),
        html.Div([
            dbc.Input(
                id="new-montage-name",
                type="text",
                placeholder="Montage name...",
                style=input_styles["path"]
            )
        ], style={"padding": "10px"}),

        dbc.Button(
            "Create",
            id="create-button",
            color="success",
            disabled=True,
            n_clicks=0
        )
    ], style={"padding": "15px", "backgroundColor": "#fff", "border": "1px solid #ddd","borderRadius": "8px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "marginBottom": "20px"}),


    # Section for frequency parameters, initially hidden
    html.Div(
        id="frequency-inputs",
        children=[
            html.H3("Frequency Parameters for Signal Processing", style={"margin-bottom": "15px"}),

            # Inputs for frequency parameters
            html.Div([
                html.Label("Resampling Frequency (Hz): "),
                dbc.Input(id="resample-freq", type="number", value=150, step=50, min=50, style=input_styles["number"]),
            ], style={"padding": "10px"}),

            html.Div([
                html.Label("High-pass Frequency (Hz): "),
                dbc.Input(id="high-pass-freq", type="number", value=0.5, step=0.1, min=0.1, style=input_styles["number"]),
            ], style={"padding": "10px"}),

            html.Div([
                html.Label("Low-pass Frequency (Hz): "),
                dbc.Input(id="low-pass-freq", type="number", value=50, step=10, min=10, style=input_styles["number"]),
            ], style={"padding": "10px"}),

            # Button and status display with loading spinner
            html.Div([
                dbc.Button(
                    "Preprocess & Display",
                    id="preprocess-display-button",
                    color="success",
                    disabled=True,
                    n_clicks=0
                ),
                # Loading spinner wraps only the elements that require loading
                dcc.Loading(
                    id="loading",
                    type="default", 
                    children=[
                        html.Div(id="preprocess-status", style={"margin-top": "10px"})
                    ]
                ),
                # Location for URL refresh
                dcc.Location(id="url", refresh=True),
            ], style={"padding": "10px", "margin-top": "20px"})
        ],
        style={
            "padding": "15px",
            "backgroundColor": "#fff",
            "border": "1px solid #ddd",  # Grey border
            "borderRadius": "8px",  # Rounded corners
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            "marginBottom": "20px",
            "display": "none"
        }
    )
])

# Callback to handle delete actions
@dash.callback(
    [
        Output("saved-montages-table", "data"),
        Output("montage-store", "data")
    ],
    Input({"type": "delete-button", "index": ALL}, "n_clicks"),
    [State("saved-montages-table", "data"), State("montage-store", "data")],
    prevent_initial_call=True
)
def delete_row(delete_clicks, table_data, montage_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        return table_data, montage_store

    # Identify which button was clicked
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    triggered_index = json.loads(triggered_id)["index"]

    # Get the montage name from the clicked row
    montage_name_to_delete = table_data[triggered_index]["montage_name"]

    # Remove the montage from the table and the montage store
    updated_table_data = [row for i, row in enumerate(table_data) if i != triggered_index]
    updated_montage_store = {
        name: channels for name, channels in montage_store.items() if name != montage_name_to_delete
    }

    return updated_table_data, updated_montage_store