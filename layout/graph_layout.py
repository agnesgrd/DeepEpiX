from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c

# Helper function to create the sidebar with checkboxes
def create_sidebar():
    return html.Div([
        # Channel Selection
        html.Label("Select Channels:"),
        dbc.Button(
            "Check All",
            id="check-all-btn",
            color="success",
            outline=True,
            size="sm",
            n_clicks=0
        ),
        dbc.Button(
            "Clear All",
            id="clear-all-btn",
            color="danger",
            outline=True,
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
            value=["MRF", "MLF"],  # Default selected regions
            inline=False,
            style={"margin": "10px 0"},
            persistence=True,
            persistence_type="local"
        ),
        # Annotation Selection
        html.Div([
            html.Label("Select Annotations:"),
            dcc.Checklist(
                id="annotation-checkboxes",
                # options=[{'label': name, 'value': name} for name in annotation_names],
                # value=annotation_names,  # Default to showing all annotations
                inline=True,
                style={"margin": "10px 0"},
                persistence=True,
                persistence_type="local"
            ),
        ], style={"marginTop": "20px"})  # Add spacing between sections
    ], style={
        "padding": "0px",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "center",
        "gap": "0px",  # Space between elements
        "width": "100px",  # Adjust width as needed for better layout
        "boxSizing": "border-box"
    })

# Helper function to create the graph container with relevant styles
def create_graph_container():
    return html.Div(
        dcc.Graph(
            id="meg-signal-graph",
            config={"responsive": True}
        ),
        style={
            "flex": 1,
            "overflowY": "auto",   # If necessary
            "whiteSpace": "nowrap",
            "padding": "0px"
        }
    )

# Main function that assembles everything into the layout
def get_graph_layout():
    return html.Div([
        # Sidebar with Channel and Annotation Sliders
        create_sidebar(),

        # Graph Container
        create_graph_container(),
        
    ], style={
        "display": "flex",  # Main container layout
        "height": "85vh",  # Full height of the viewport
        "width": "70%",  # Full width of the viewport
        "gap": "0px",  # Space between elements
        "boxSizing": "border-box",  # Ensure padding is included in dimensions
        "flexDirection": "row",  # Layout horizontally to balance the sidebar and graph
        "padding": "10px"
    })