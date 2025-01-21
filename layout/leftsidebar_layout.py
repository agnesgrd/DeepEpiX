from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles

# Helper function to create the sidebar with checkboxes
def create_leftsidebar():
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
                persistence=False,
                persistence_type="local"
            ),
        ], style = box_styles["classic"]),
    ], style={
        # "padding": "20px",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "flex-start",  # Align content at the top
        "gap": "20px",  # Space between elements
        "width": "250px",  # Sidebar width is now fixed
        "boxSizing": "border-box",
        "fontSize": "12px",
        # "backgroundColor": "#f9f9f9",  # Light background color for the sidebar
        "borderRadius": "10px",  # Rounded corners for the sidebar itself
        # "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",  # Subtle shadow for the whole sidebar
        "overflowY": "auto",  # Enable scrolling if content exceeds height
    })