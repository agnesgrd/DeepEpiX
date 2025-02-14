from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles

def create_selection():
    return html.Div([

        # Montage Selection
        html.Div([
            html.Label(
                "Select Montage:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            dcc.RadioItems(
                id="montage-radio",
                options = [],
                inline=False,
                style={"margin": "10px 0", "fontSize": "12px"},
                persistence=True,
                persistence_type="local"
            ),
        ], style = box_styles["classic"]),

        # Channel Selection
        html.Div([
            html.Label(
                "Select Channels:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            dbc.Button(
                "Check All",
                id="check-all-btn",
                color="success",
                outline=True,
                size="sm",
                n_clicks=0,
                style={
                    "fontSize": "12px",
                    "padding": "6px 12px",
                    "borderRadius": "5px",
                    "width": "100%",
                    "marginBottom": "10px"
                }
            ),
            dbc.Button(
                "Clear All",
                id="clear-all-btn",
                color="danger",
                outline=True,
                size="sm",
                n_clicks=0,
                style={
                    "fontSize": "12px",
                    "padding": "6px 12px",
                    "borderRadius": "5px",
                    "width": "100%",
                    "marginBottom": "15px"
                }
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
                value=["MRF", "MLF"],  # Default selected regions
                inline=False,
                style={
                    "margin": "10px 0",
                    "fontSize": "12px",
                    "borderRadius": "5px",
                    "padding": "8px",
                    "border": "1px solid #ddd",
                },
                persistence=True,
                persistence_type="local"
            ),
        ], style=box_styles["classic"]),

        # Annotation Selection
        html.Div([
            html.Label(
                "Select Annotations:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            dcc.Checklist(
                id="annotation-checkboxes",
                inline=False,
                style={"margin": "10px 0", "fontSize": "12px"},
                persistence=True,
                persistence_type="session"
            ),
        ], style = box_styles["classic"]),

        # Offset Selection
        html.Div([
            html.Label(
                "Select Amplitude:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            html.Div(
                children=[
                    dbc.Button("-", id="offset-decrement", color="primary", size="sm", n_clicks=0),
                    html.Span(
                        id="offset-display", 
                        children="5",  # Default offset value
                        style={
                            "margin": "0 10px",  # Space between buttons
                            "fontWeight": "bold",
                            "fontSize": "12px"
                        }
                    ),
                    dbc.Button("+", id="offset-increment", color="primary", size="sm", n_clicks=0),
                ],
                style={
                    "display": "flex", 
                    "alignItems": "center", 
                    "gap": "12px"  # Space between elements
                }
            ),
        ], style=box_styles["classic"]),

        # Montage Selection
        html.Div([
            html.Label(
                "Select Colors:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            dcc.RadioItems(
                id="colors-radio",
                options = [{'label':'unified', 'value': 'unified'}, {'label':'rainbow', 'value': 'rainbow'}],
                value='rainbow',
                inline=False,
                style={"margin": "10px 0", "fontSize": "12px"},
                # persistence=True,
                # persistence_type="local"
            ),
        ], style = box_styles["classic"]),


    ])
