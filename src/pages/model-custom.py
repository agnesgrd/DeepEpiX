import pandas as pd

import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from layout.config_layout import INPUT_STYLES, FLEXDIRECTION
from callbacks.utils import annotation_utils as au
from callbacks.utils import performance_utils as pu
import callbacks.utils.model_utils as mu

dash.register_page(__name__, name="Model Custom", path="/model/custom")

layout = html.Div(
    [
        dcc.Input(id="file-name", type="text", placeholder="Enter new file name"),
        html.Button("Create File", id="create-btn"),
        html.Div(id="output"),
    ]
)


@callback(
    Output("output", "children"),
    Input("create-btn", "n_clicks"),
    Input("file-name", "value"),
)
def create_file(n_clicks, file_name):
    if n_clicks is None or not file_name:
        return ""

    # Ensure the file has .py extension
    if not file_name.endswith(".py"):
        file_name += ".py"

    # Standard template content
    template = '''"""
New Dash Page
Author: Your Name
"""

from dash import html

layout = html.Div([
    html.H1("New Page"),
    html.P("This is a standard template.")
])
'''
    try:
        with open(file_name, "w") as f:
            f.write(template)
        return f"File '{file_name}' created successfully!"
    except Exception as e:
        return f"Error: {e}"
