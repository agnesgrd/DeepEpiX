# view.py
import dash
from dash import html

dash.register_page(__name__)

layout = html.Div([
    html.H1("VIEW: Visualize and Annotate MEG Signal"),
    html.Div("This is the body of the Viewer Page. You can add components here later.")
])