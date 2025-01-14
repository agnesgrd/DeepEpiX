from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles
# Helper function to create the sidebar with checkboxes
def create_leftsidebar():
    return html.Div([
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
        "padding": "20px",
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
                            'range': [0, 10],
                            'title': 'Time (s)',
                            'rangeslider': {'visible': True, 'thickness': 0.02},
                        },
                        'yaxis': {'title': 'Channels'},
                        'title': 'MEG Signal Visualization',
                        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                    },
                },
                config={"responsive": True},
                style={"width": "100%", "height": "100%"}
            ),
            # Annotation Graph (overlay graph)
            dcc.Graph(
                id="annotation-graph",
                figure={
                    'data': [],
                    'layout': {
                        'xaxis': {
                            'range': [0, 180],
                            'title': '',  # Hide title for overlay
                            'showgrid': False,  # Remove grid lines for cleaner overlay
                            'zeroline': False,
                        },
                        'yaxis': {
                            'title': 'Events',  # Hide y-axis completely
                            'titlefont': {
                                'color': 'red'  # Transparent text
                            },
                            'showgrid': True,
                            'tickvals': [0],
                            'ticktext': ['MRF67-2805'],
                            'ticklabelposition': 'outside right',
                            'side': 'right',
                            'tickfont': {
                                'color': 'rgba(0, 0, 0, 0)'  # Transparent text
                            },
                            'range': [0, 5],  # Adjust based on the annotation data
                        },
                        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                        'paper_bgcolor': 'rgba(0,0,0,0)',  # Transparent background
                        'plot_bgcolor': 'rgba(0,0,0,0)'  # Transparent plot area
                    },
                },
                config={"staticPlot": True},  # Disable interaction
                style={
                    "width": "100%",  # Ensure full width for both graphs
                    "height": "20vh",  # Set height to 20% of the screen height
                    "pointerEvents": "none",  # Allow interactions with the MEG graph
                }
            )
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100vh",  # Ensure full screen height usage
        }
    )
def create_rightsidebar():
    return html.Div([
        # Plot topomap on a unique timepoint
        html.Div([
            # Label and input field for timepoint entry
            html.Label(
                "Please enter a timepoint to plot topomap:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            dbc.Input(
                id="topomap-timepoint",  # Unique ID for each input
                type="number",
                placeholder="",
                step=0.01,
                min=0,
                max=180,
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Button(
                "Plot Topomap",
                id="plot-topomap-button",  # Unique ID for each button
                color="info",
                outline=True,
                size="sm",
                n_clicks=0,
                style=button_styles["plot-topomap"]
            ),
            # Modal (popup) for displaying the topomap image
            dbc.Modal(
                [
                    dbc.ModalHeader("Topomap", close_button=False),
                    dbc.ModalBody(
                        html.Img(
                            id="topomap-img",
                            src="https://via.placeholder.com/150",  # Placeholder image URL
                            alt="topomap-img",
                            style={
                                "width": "100%",
                                "height": "auto",
                                "borderRadius": "10px",
                                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"  # Light shadow for the image
                            }
                        ),
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-topomap-modal", color="secondary")
                    ),
                ],
                id="topomap-modal",
                is_open=False,  # Initially hidden
            ),
        ], style=box_styles["classic"]),
    ], style={
        "display": "flex",
        "flexDirection": "column",  # Stack the three sections vertically
        "gap": "20px",  # Space between sections
        "width": "10%",
        "maxWidth": "450px",  # You can set a max width for the sidebar
        "margin": "0 10px"
    })
def get_graph_layout():
    return html.Div([
        create_leftsidebar(),
        create_graph_container(),
        create_rightsidebar(),
    ], style={
        "display": "flex",  # Horizontal layout
        "flexDirection": "row",
        "height": "85vh",  # Use the full height of the viewport
        "width": "95vw",  # Use the full width of the viewport
        "overflow": "hidden",  # Prevent overflow in case of resizing
        "boxSizing": "border-box"
    })