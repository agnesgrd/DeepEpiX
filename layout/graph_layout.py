from dash import html, dcc
import dash_bootstrap_components as dbc

def create_graph_container(
    update_button_id="update-button",
    update_container_id="update-container",
    page_buttons_container_id="page-buttons-container",
    page_selector_id="page-selector",
    next_spike_buttons_container_id="next-spike-buttons-container",
    prev_spike_id="prev-spike",
    next_spike_id="next-spike",
    annotation_dropdown_id="annotation-dropdown",
    loading_id="loading-graph",
    signal_graph_id="meg-signal-graph",
    annotation_graph_id="annotation-graph"
):
    return html.Div(
        [
            # Update Button
            html.Div(
                id=update_container_id,
                style={
                    "position": "absolute",
                    "top": "5px",
                    "left": "15px",
                    "background-color": "rgba(0,0,0,0)",
                    "border-radius": "5px",
                    "box-shadow": "2px 2px 5px rgba(0,0,0,0.2)",
                    "z-index": "1000",
                    "opacity": 0.8
                },
                children=[
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise")],
                        id=update_button_id,
                        n_clicks=0,
                        outline=True,
                        color="primary"
                        # className="btn btn-primary"
                    ),
                    dbc.Tooltip("Refresh the graph after modifying the display parameters", target=update_button_id, placement="top"),
                ]
            ),

            # Page Buttons
            html.Div(
                id=page_buttons_container_id,
                style={
                    "position": "absolute",
                    "top": "5px",
                    "left": "300px",
                    "background-color": "rgba(0,0,0,0)",
                    "border-radius": "5px",
                    "box-shadow": "2px 2px 5px rgba(0,0,0,0.2)",
                    "z-index": "1000",
                    "opacity": 0.8
                },
                children=[
                    dcc.RadioItems(
                        id=page_selector_id,
                        options=[],
                        value=0
                    )
                ]
            ),
            dbc.Tooltip("The graph is divided in multiple page.", target=page_buttons_container_id, placement="top"),

            # Prev / Next Spike Buttons
            html.Div(
                id=next_spike_buttons_container_id,
                style={
                    "position": "absolute",
                    "top": "5px",
                    "left": "600px",
                    "background-color": "rgba(0,0,0,0)",
                    "border-radius": "5px",
                    "box-shadow": "2px 2px 5px rgba(0,0,0,0.2)",
                    "z-index": "1000",
                    "opacity": 0.8, 
                    "display":"flex"
                },
                children=[
                    dbc.Button(
                        html.I(className="bi bi-arrow-left-circle"),
                        id=prev_spike_id,
                        color="link",  # optional, you can also keep 'primary'
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "border": "none",
                            "boxShadow": "none",
                            "fontSize": "1.5rem",
                            "padding": "0.25rem",
                            "color": "#007bff",  # You can adjust the color if needed
                            "cursor": "pointer"
                        }
                    ),
                    dbc.Tooltip("Move graph to previous spike", target=prev_spike_id, placement="top"),
                    dcc.Dropdown(
                        id=annotation_dropdown_id,
                        persistence=True,
                        persistence_type="local",
                        clearable=False,
                        style={
                            "width": "110px",
                            "fontSize": "10px"
                        }
                    ),
                    dbc.Button(
                        html.I(className="bi bi-arrow-right-circle"),
                        id=next_spike_id,
                        color="link",  # optional, you can also keep 'primary'
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "border": "none",
                            "boxShadow": "none",
                            "fontSize": "1.5rem",
                            "padding": "0.25rem",
                            "color": "#007bff",  # You can adjust the color if needed
                            "cursor": "pointer"
                        }
                    ),
                    dbc.Tooltip("Move graph to next spike", target=next_spike_id, placement="top"),
                ]
            ),

            # Signal Graph with loading
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[
                    dcc.Graph(
                        id=signal_graph_id,
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
                            'doubleClick': 'reset'
                        },
                            style={
                            "width": "100%",
                            "height": "80vh",
                            "borderRadius": "10px",
                            "overflow": "hidden",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                        }
                    ),
                ],
                overlay_style={
                    "backgroundColor": "rgba(255, 255, 255, 0.5)",
                    "pointerEvents": "none",  # So the graph stays interactive
                    "visibility": "visible"  # This is the key: don't hide the child
                }

            ),

            # Annotation Graph
            dcc.Graph(
                id=annotation_graph_id,
                figure={
                    'data': [],
                    'layout': {
                        'xaxis': {
                            'title': '',
                            'showgrid': False,
                            'zeroline': False
                        },
                        'yaxis': {
                            'title': 'Events',
                            'titlefont': {'color': 'rgba(0,0,0,0)'},
                            'showgrid': False,
                            'tickvals': [0],
                            'ticktext': ['MRF67-2805'],
                            'ticklabelposition': 'outside right',
                            'side': 'right',
                            'tickfont': {'color': 'rgba(0, 0, 0, 0)'},
                            'range': [0, 1],
                        },
                        'margin': {'l': 10, 'r': 0, 't': 0, 'b': 20},
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                    },
                },
                config={"staticPlot": True},
                style={
                    "width": "100%",
                    "height": "8vh",
                    "pointerEvents": "none",
                    "borderRadius": "10px",
                    "overflow": "hidden",
                    "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                }
            )
        ],
        style={
            "position": "relative",
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100vh"
        }
    )