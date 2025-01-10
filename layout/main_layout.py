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

# Common styles

input_styles = {
    "path":{
        "width": "100%",
        "padding": "10px",
        "fontSize": "16px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
    },
    "number":{
        "width": "10%",
        "padding": "10px",
        "fontSize": "16px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "margin": "10px"
    },
    "small-number":{
        "width": "100%",
        "padding": "10px",
        "fontSize": "12px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "margin": "10px"
    }
}

box_style = {
    "padding": "15px",
    "backgroundColor": "#fff",
    "border": "1px solid #ddd",  # Grey border
    "borderRadius": "8px",  # Rounded corners for the channel section
    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
    "marginBottom": "20px"  # Space between the sections
    }