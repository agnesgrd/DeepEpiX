from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles

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