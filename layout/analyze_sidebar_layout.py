from dash import html, dcc
import dash_bootstrap_components as dbc
import config
from layout import input_styles, box_styles, button_styles


def create_analyze():
    return html.Div([
    
    # Plot topomap on a interval timepoint
        html.Div([
            html.Div([
                html.H6([html.I(className=f"bi bi-crosshair"), " Topomap"], style={"fontWeight": "bold", "marginBottom": "10px"}),
                dbc.Button(
                    "Activate",
                    id="plot-topomap-button",  # Unique ID for each button
                    color="info",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    disabled=False,
                    style=button_styles["big"]
                ),
                dbc.Tooltip("Turn on the topomap with this button, then click the graph at the moment you're interested in.", target="plot-topomap-button", placement="left"),
                # Loading component to show the loading spinner while the long callback is processing
                dcc.Loading(
                    id="topomap-loading",
                    type="dot",  # You can use "circle", "dot", etc. for different spinner styles
                    children=html.Div(id="topomap-result", style={"marginTop": "0px"})
                ),
            ]),
            html.Div(id="topomap-picture"),
            # Modal (popup) for displaying the topomap video
            dbc.Modal(
                [
                    dbc.ModalHeader("Topomap", close_button=True),
                    dbc.ModalBody(
                        children=[
                            html.Div(id="topomap-modal-content"),  # Content dynamically populated
                        ]
                    )
                ],
                id="topomap-range-modal",
                is_open=False,  # Initially hidden
                size="lg",
                style={
                    "maxWidth": "90vw"
                }
            ),
        ], style=box_styles["classic"]),

        # Add a spike
        html.Div([
            html.H6([html.I(className=f"bi bi-pencil"), " Event modification"], style={"fontWeight": "bold", "marginBottom": "10px"}),
            dbc.Input(
                id="event-name",  # Unique ID for each input
                type="text",
                placeholder="Enter a name ...",
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Input(
                id="event-onset",  # Unique ID for each input
                type="number",
                placeholder="Onset (s) ...",
                step=0.01,
                min=0,
                max=180,
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Tooltip("Click on the graph to mark the desired time point or enter it manually.", target="event-onset", placement="left"),
            dbc.Input(
                id="event-duration",  # Unique ID for each input
                type="number",
                placeholder="Duration (s) ...",
                step=0.01,
                min=0,
                max=180,
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Button(
                "Add new event",
                id="add-event-button",  # Unique ID for each button
                color="success",
                outline=True,
                size="sm",
                n_clicks=0,
                style=button_styles["big"],
                disabled=True
            ),
            dbc.Button(
                "Delete selected event(s)",
                id="delete-event-button",  # Unique ID for each button
                color="danger",
                outline=True,
                size="sm",
                n_clicks=0,
                style=button_styles["big"],
                disabled=True
            )
        ], style=box_styles["classic"]),

        # History Section
        html.Div([
            html.H6([html.I(className=f"bi bi-activity"), " Annotations History"], style={"fontWeight": "bold", "marginBottom": "10px"}),  # Title for the history section
            html.Div(
                id="history-log",  # Dynamic log area
                style={
                    "height": "200px",  # Adjust the height as needed
                    "overflowY": "auto",  # Scrollable if content exceeds height
                }
            ),
            dbc.Button(
                "Clean",
                id="clean-history-button",
                color="danger",
                outline=True,
                size="sm",
                n_clicks=0,
                style=button_styles["big"]
            ),
        ], style=box_styles["classic"]),
    ])     