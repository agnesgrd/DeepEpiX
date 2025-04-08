# save.py
import dash
from dash import html

dash.register_page(__name__, name="Model fine-tuning", href="model-fine-tuning")

layout = html.Div([
    html.H1("Not a page yet"),
    html.Div("This is the body of the Page. You can add components here later.")
])