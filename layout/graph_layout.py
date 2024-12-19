from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

# Helper function to create the sidebar with checkboxes
def create_sidebar():
    return html.Div([
        html.Label("Select Channels:"),
        dbc.Button(
            "Check All",
            id="check-all-btn",
            color="success",
            outline="True",
            size="sm",
            n_clicks=0
        ),
        dbc.Button(
            "Clear All",
            id="clear-all-btn",
            color="danger",
            outline="True",
            size="sm",
            n_clicks=0
        ),
        dcc.Checklist(
            id="channel-region-checkboxes",
            options=[
                {
                    'label': f"{region_code} ({len(channels)})", 
                    'value': region_code
                }
                for region_code, channels in c.GROUP_CHANNELS_BY_REGION.items()
            ],
            value=["MRF", "MLF"],  # Default selected regions (Right, Left, Z regions)
            inline=False,
            style={"margin": "10px 0"},
            persistence=True,
            persistence_type="local"
        )
    ], style={
        "padding": "10px",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "center"
    })

# Helper function to create the graph container
def create_graph_container():
    return html.Div([
        dcc.Graph(id="meg-signal-graph")
    ], style={"flexGrow": 1, "padding": "10px"})

# Helper function to create the time slider container
def create_time_slider():
    return html.Div([
        html.Label("Select Time Range (s):"),
        dcc.RangeSlider(
            id="time-slider",
            min=0,
            max=180,  # Default max, update dynamically
            step=1,
            marks={i: str(i) for i in range(0, 181, 10)},  # Add visible marks
            value=[0, 10],  # Default range
            tooltip={"placement": "bottom", "always_visible": True},
            persistence=True,
            persistence_type="local"
        ),
        html.Div([
            html.Button("←", id="time-left", n_clicks=0, style={"font-size": "16px", "padding": "5px 10px"}),
            html.Button("→", id="time-right", n_clicks=0, style={"font-size": "16px", "padding": "5px 10px", "margin-left": "10px"})
        ], style={"display": "flex", "flex-direction": "row", "align-items": "center", "margin-top": "10px"})
    ], style={
        "margin-top": "20px",
        "width": "100%",
        "textAlign": "center"
    })

# Main function that assembles everything into the layout
def get_graph_layout():
    return html.Div([
        # Sidebar with Channel Slider
        create_sidebar(),

        # Graph and Time Slider Container
        html.Div([
            # Graph
            create_graph_container(),

            # Time Slider
            create_time_slider()
        ], style={
            "flexGrow": 1,
            "display": "flex",
            "flexDirection": "column",  # Stack graph and time slider vertically
            "padding": "10px"
        })
    ], style={
        "display": "flex",  # Main container layout
        "height": "500px",
        "gap": "10px"
    })
