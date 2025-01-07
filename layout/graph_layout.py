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
            n_clicks=0,
            style={"fontSize": "12px"}
        ),
        dbc.Button(
            "Clear All",
            id="clear-all-btn",
            color="danger",
            outline=True,
            size="sm",
            n_clicks=0,
            style={"fontSize": "12px"}
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
        "boxSizing": "border-box",
        "fontSize": "12px"
    })



def create_graph_container():
    return html.Div(
        [
            # MEG Signal Graph (base graph)
            dcc.Graph(
                id="meg-signal-graph",
                figure={
                    'data': [],
                    'layout': {
                        'xaxis': {
                            'range': [0, 10],  # Initial time range
                            'title': 'Time (s)',
                            'rangeslider': {
                                'visible': True, # Enable the range slider
                                'thickness': 0.02
                            },
                        },
                        'yaxis': {
                            'title': 'Channels',
                        },
                        'title': 'MEG Signal Visualization',
                    },
                },
                config={"responsive": True},
                style={
                    "width": "100%", 
                    "height": "100%", 
                    "position": "relative"  # Position relative to allow overlaying
                },
            ),

            # Annotation Graph (overlay graph)
            dcc.Graph(
                id="annotation-graph",
                figure={
                    'data': [],
                    'layout': {
                        'xaxis': {
                            'range': [0, 180],  # Initial time range
                            'title': '',  # Hide title for overlay
                            'showgrid': False,  # Hide grid for cleaner look
                            'zeroline': False,  # Remove zero line
                            # 'ticks': 'outside',  # Remove ticks
                            # 'showticklabels': True, # Hide tick labels
                        },
                        'yaxis': {
                            'visible': False,  # Hide y-axis completely
                            'range': [-5,5]
                        },
                        'title': '',  # No title for overlay graph
                        'margin': {
                            'l': 0, 'r': 0, 't': 0, 'b': 0  # Minimize margins
                        },
                        'paper_bgcolor': 'rgba(0,0,0,0)',  # Transparent background
                        'plot_bgcolor': 'rgba(0,0,0,0)'  # Transparent plot area
                    },
                },
                config={"staticPlot": True},  # Disable interaction
                style={
                    "position": "absolute",  # Overlay on top of MEG graph
                    "top": "93%",  # Adjust position to be higher (10% from the top)
                    "left": "0",  # Align to the left
                    "width": "88%",  # Full width
                    "height": "3%",  # Set height to 10% of the container
                    "marginLeft": "7%",  # Align to the right
                    "marginRight": "0",  # No margin on the right
                    "pointerEvents": "none"  # Ensure it doesn't block interactions with the MEG graph
                },
            )
        ],
        style={
            "position": "relative",  # Allow absolute positioning within
            "width": "100%", 
            "height": "100vh", 
            "overflow": "hidden",  # Prevent overflow
            "display": "block",  # Default block layout
            "padding": "0px",
            "margin": "0px",
            "boxSizing": "border-box"  # Ensure padding is included in height calculation
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