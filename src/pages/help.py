import dash
from dash import html

dash.register_page(__name__, name = "Help", path="/settings/help")

layout = html.Div([
    html.Iframe(
        src="http://localhost:8000",  # Replace with your MkDocs site URL
        style={"width": "100%", "height": "1600px", "border": "none"}
    )
])

