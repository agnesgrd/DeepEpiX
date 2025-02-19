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
    "number-in-box":{
        "width": "50%",
        "padding": "10px",
        "fontSize": "16px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "margin": "10px"
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
        "padding": "8px",
        "fontSize": "12px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "margin": "10px 0"
    }
}

button_styles = {
    "big": {
        "fontSize": "12px",
        "padding": "8px",
        "borderRadius": "5px",
        "width": "100%",
        "margin": "10px 0"
    }
}

box_styles = {
    "panel-tabs": {
        "padding": "15px 25px",  # More spacious padding
        "text-decoration": "none", 
        "font-size": "18px", 
        "color": "white", 
        "border-radius": "12px",  # Rounded corners for a modern feel
        "margin": "10px",  # Increased margin for better separation
        "display": "inline-block",
        "box-shadow": "0px 4px 12px rgba(0, 0, 0, 0.15)",  # Deeper shadow for depth
        "transition": "all 0.3s ease",  # Smooth transition for hover
        "background-color": "#6c757d",  # Soft gray background
        "cursor": "pointer",  # Pointer cursor to indicate interactivity
    },
    "classic": {
        "padding": "15px",
        "backgroundColor": "#fff",
        "border": "1px solid #ddd",  # Grey border
        "borderRadius": "8px",  # Rounded corners for the channel section
        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
        "marginBottom": "20px"  # Space between the sections
    }
}

label_styles = {
    "classic": {
        "fontWeight": "bold",
        "fontSize": "14px",
        "marginBottom": "5px",
    }
}