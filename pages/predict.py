# predict.py
import dash
from dash import html

dash.register_page(__name__)

layout = html.Div([
    html.H1("PREDICT: Test or Fine-Tune Model"),
    html.Div("This is the body of the Model Page. You can add components here later.")
])