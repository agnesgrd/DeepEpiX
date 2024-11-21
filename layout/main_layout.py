import dash
from dash import html, dcc

# Main layout for the app (homepage)
layout = html.Div([
    html.H1("Welcome to DeepEpiX"),
    html.Div([
        html.P("Analyze MEG data for epileptic spike detection."),
        dcc.Link("Go to Spike Detection", href='/spike-detection')
    ])
])