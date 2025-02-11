from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles


def create_graph_container():
    return html.Div(
        [
            # Page selector (positioned on top-left)
            html.Div(
                id="page-buttons-container",
                style={
                    "position": "absolute",
                    "top": "15px",
                    "left": "15px",
                    "background-color": "rgba(0,0,0,0)",
                    # "padding": "5px",
                    "border-radius": "5px",
                    "box-shadow": "2px 2px 5px rgba(0,0,0,0.2)",
                    "z-index": "1000",
                    "opacity": 0.8  # Slight transparency
                },
                children=[
                    dcc.RadioItems(
                        id="page-selector",
                        options=[],  # Initially empty
                        value=0  # Default to the first page
                    )
                ]
            ),

            # Page selector (positioned on top-left)
            html.Div(
                id="next-spike-buttons-container",
                style={
                    "position": "absolute",
                    "top": "15px",
                    "left": "300px",
                    "background-color": "rgba(0,0,0,0)",
                    #"padding": "5px",
                    "border-radius": "5px",
                    "box-shadow": "2px 2px 5px rgba(0,0,0,0.2)",
                    "z-index": "1000",
                    "opacity": 0.8  # Slight transparency
                },
                children=[
                    dbc.Button("Previous", id="prev-spike", color="primary", outline=True, n_clicks=0),
                    dbc.Button("Next", id="next-spike", color="primary", outline=True, n_clicks=0)
                ]
            ),

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
                            'showspikes': True
                        },
                        'yaxis': {'title': 'Channels', 'fixedrange': True},
                        'title': 'MEG Signal Visualization',
                        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                        'hovermode': 'closest'
                        
                    },
                },
                config={
                    "responsive": True,
                    'doubleClick': 'reset',  # Reset zoom on double-click
                    },
                style={"width": "100%", "height": "80vh"}
            ),
            # Annotation Graph (overlay graph)
            dcc.Graph(
                id="annotation-graph",
                figure={
                    'data': [],
                    'layout': {
                        'xaxis': {
                            'title': '',  # Hide title for overlay
                            'showgrid': False,  # Remove grid lines for cleaner overlay
                            'zeroline': False
                        },
                        'yaxis': {
                            'title': 'Events',  # Hide y-axis completely
                            'titlefont': {
                                'color': 'rgba(0,0,0,0)'  # Transparent text
                            },
                            'showgrid': False,
                            'tickvals': [0],
                            'ticktext': ['MRF67-2805'],
                            'ticklabelposition': 'outside right',
                            'side': 'right',
                            'tickfont': {
                                'color': 'rgba(0, 0, 0, 0)'  # Transparent text
                            },
                            'range': [0, 1],  # Adjust based on the annotation data
                        },
                        'margin': {'l': 10, 'r': 0, 't': 0, 'b': 20},
                        'paper_bgcolor': 'rgba(0,0,0,0)',  # Transparent background
                        'plot_bgcolor': 'rgba(0,0,0,0)',  # Transparent plot area
                    },
                },
                config={"staticPlot": True},  # Disable interaction
                style={
                    "width": "100%",  # Ensure full width for both graphs
                    "height": "8vh",  # Set height to 20% of the screen height
                    "pointerEvents": "none",  # Allow interactions with the MEG graph
                }
            )
        ],
        style={
            "position": "relative",  # Ensures absolute positioning inside this container
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100vh",  # Full screen height
        }
    )

        # html.Div(id="page-buttons-container", 
        #     style=box_styles["classic"], 
        #     children=[dcc.RadioItems(
        #         id="page-selector",
        #         options=[],  # Initially empty
        #         value=0  # Default to the first page
        #     )]),