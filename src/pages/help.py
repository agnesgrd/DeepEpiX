import pandas as pd

import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, name = "Help", path="/settings/help")

layout = html.Div([
    html.Iframe(
        src="http://localhost:8000",  # Replace with your MkDocs site URL
        style={"width": "100%", "heigth": "600px", "border": "none"}
    )
])

