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
                    "top": "15px",
                    "left": "30px",
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
                        color="warning",
                        style={
                            "minWidth": "20px",
                            "height": "30px",
                            "lineHeight": "1",
                            "fontSize": "1.5rem",
                            "border": "none"
                        }
                    ),
                    dbc.Tooltip("Refresh the graph after modifying the display parameters", target=update_button_id, placement="top"),
                ]
            ),

            # Page Buttons
            html.Div(
                id=page_buttons_container_id,
                style={
                    "position": "absolute",
                    "top": "20px",
                    "left": "100px",
                    "background-color": "rgba(0,0,0,0)",
                    "border-radius": "5px",
                    "box-shadow": "2px 2px 5px rgba(0,0,0,0.2)",
                    "z-index": "1000",
                    "opacity": 0.8
                },
                children=[
                    dbc.RadioItems(
                        id=page_selector_id,
                        options=[],
                        value=0,
                        style={
                            # "minWidth": "24px",
                            # "height": "24px",
                            "lineHeight": "1",
                            "fontSize": "1.8rem"
                        }
                    )
                ]
            ),
            dbc.Tooltip("The graph is divided in multiple page.", target=page_buttons_container_id, placement="top"),

            # Prev / Next Spike Buttons
            html.Div(
                id=next_spike_buttons_container_id,
                style={
                    "position": "absolute",
                    "top": "10px",
                    "left": "300px",
                    # "background-color": "rgba(0,0,0,0)",
                    # "border-radius": "5px",
                    # "box-shadow": "2px 2px 5px rgba(0,0,0,0.2)",
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
                        outline=True,
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
                            "width": "100px",
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

            # Prev / Next Spike Buttons
            html.Div(
                id="cursor",
                style={
                    "position": "absolute",
                    "top": "10px",
                    "left": "50%",
                    "z-index": "1000",
                    "opacity": 0.8, 
                    "display":"flex"
                },
                children=
                html.Span(
                    html.I(className="bi bi-caret-down-fill"),
                ),
            ),

            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[
                    dcc.Graph(
                        id=signal_graph_id,
                        figure={
                            'data': [],
                            'layout': {
                                'template': 'plotly_dark',
                                'xaxis': {
                                    'range': [0, 10], 
                                    'rangeslider': {
                                        'visible': True,
                                        'thickness': 0.02,
                                        'bgcolor': 'rgba(128, 128, 128, 0.5)',  # Medium grey with some transparency
                                        'bordercolor': 'rgba(64, 64, 64, 1)',   # Darker grey for border
                                    },
                                    'showspikes': True
                                },
                                'yaxis': {
                                    'title': None,
                                    'showticklabels': False,
                                    'autorange': True
                                },
                                'font': {'color': 'white'},
                                'paper_bgcolor': 'rgba(0,0,0,1)',
                                'plot_bgcolor': 'rgba(0,0,0,1)',
                                'margin': {'l': 0, 'r': 0, 't': 30, 'b': 0},
                                'hovermode': 'closest',
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
                            "backgroundColor": "#000",
                            "boxShadow": "none"
                        }
                    ),
                ],
                overlay_style={
                    "pointerEvents": "none",
                    "visibility": "visible"
                }
            ),

            dcc.Graph(
                id=annotation_graph_id,
                figure={
                    'data': [],
                    'layout': {
                        'template': 'plotly_dark',
                        'xaxis': {
                            'title': '',
                            'showgrid': False,
                            'zeroline': False,
                            'color': 'white'
                        },
                        'yaxis': {
                            'title': '',
                            'showgrid': False,
                            'tickvals': [0],
                            'ticktext': ['MRF67-2805'],
                            'ticklabelposition': 'outside right',
                            'side': 'right',
                            'tickfont': {'color': 'rgba(0, 0, 0, 0)'},
                            'range': [0, 1],
                            'color': 'white'
                        },
                        'paper_bgcolor': 'rgba(0,0,0,1)',
                        'plot_bgcolor': 'rgba(0,0,0,1)',
                        'font': {'color': 'white'},
                        'margin': {'l': 10, 'r': 0, 't': 0, 'b': 20},
                    },
                },
                config={"staticPlot": True},
                style={
                    "width": "100%",
                    "height": "8vh",
                    "pointerEvents": "none",
                    "borderRadius": "10px",
                    "overflow": "hidden",
                    "backgroundColor": "#000",
                    "boxShadow": "none"
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