from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles


def create_analyze():
    return html.Div([

        # # Plot topomap on a unique timepoint
        # html.Div([
        #     # Label and input field for timepoint entry
        #     # html.Label(
        #     #     "Timestep (s) :",
        #     #     style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
        #     # ),
        #     dbc.Input(
        #         id="topomap-timepoint",  # Unique ID for each input
        #         type="number",
        #         placeholder="Timestep (s) ...",
        #         step=0.01,
        #         min=0,
        #         max=180,
        #         size="sm",
        #         persistence=True,
        #         persistence_type="local",
        #         style={**input_styles["small-number"]}
        #     ),
        #     dbc.Button(
        #         "Plot Topomap",
        #         id="plot-topomap-button",  # Unique ID for each button
        #         color="info",
        #         outline=True,
        #         size="sm",
        #         n_clicks=0,
        #         style=button_styles["big"]
        #     ),
        #     # Modal (popup) for displaying the topomap image
        #     dbc.Modal(
        #         [
        #             dbc.ModalHeader("Topomap", close_button=True),
        #             dbc.ModalBody(
        #                 html.Img(
        #                     id="topomap-img",
        #                     src="https://via.placeholder.com/150",  # Placeholder image URL
        #                     alt="topomap-img",
        #                     style={
        #                         "width": "auto",
        #                         "height": "20%",
        #                         "borderRadius": "10px",
        #                         "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)"  # Light shadow for the image
        #                     }
        #                 ),
        #             ),
        #             # dbc.ModalFooter(
        #             #     dbc.Button("Close", id="close-topomap-modal", color="secondary")
        #             # ),
        #         ],
        #         id="topomap-modal",
        #         is_open=False,  # Initially hidden
        #     ),
        # ], style=box_styles["classic"]),
    
    # Plot topomap on a interval timepoint
        html.Div([
            # Label and input field for timepoint entry
            # html.Label(
            #     "Time Range (s) :",
            #     style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            # ),
            dbc.Input(
                id="topomap-min-range",  # Unique ID for each input
                type="number",
                placeholder="Minimum range (s) ...",
                step=0.01,
                min=0,
                max=180,
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Input(
                id="topomap-max-range",  # Unique ID for each input
                type="number",
                placeholder="Maximum range (s) ...",
                step=0.01,
                min=0,
                max=180,
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            html.Div([
                dbc.Button(
                    "Plot Topomap",
                    id="plot-topomap-button-range",  # Unique ID for each button
                    color="info",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    disabled=True,
                    style=button_styles["big"]
                ),
                # Loading component to show the loading spinner while the long callback is processing
                dcc.Loading(
                    id="topomap-loading",
                    type="dot",  # You can use "circle", "dot", etc. for different spinner styles
                    children=html.Div(id="topomap-result", style={"marginTop": "0px"})
                ),
            ]),
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
            # Label and input field for timepoint entry
            # html.Label(
            #     "Spike:",
            #     style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            # ),
            dbc.Input(
                id="spike-name",  # Unique ID for each input
                type="text",
                placeholder="Spike name ...",
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Input(
                id="spike-timestep",  # Unique ID for each input
                type="number",
                placeholder="Timestep (s) ...",
                step=0.01,
                min=0,
                max=180,
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Button(
                "Add new spike",
                id="add-spike-button",  # Unique ID for each button
                color="success",
                outline=True,
                size="sm",
                n_clicks=0,
                style=button_styles["big"],
                disabled=True
            ),
            dbc.Button(
                "Delete selected spike",
                id="delete-spike-button",  # Unique ID for each button
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
            html.H6("History", style={"fontWeight": "bold", "marginBottom": "10px"}),  # Title for the history section
            html.Div(
                id="history-log",  # Dynamic log area
                style={
                    "height": "150px",  # Adjust the height as needed
                    "overflowY": "auto",  # Scrollable if content exceeds height
                    "border": "1px solid #ccc",  # Light border for clarity
                    "borderRadius": "5px",
                    "padding": "5px",
                    "backgroundColor": "#f9f9f9",  # Light background
                    "fontSize": "12px",
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