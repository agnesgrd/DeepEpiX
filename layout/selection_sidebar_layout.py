from dash import html, dcc
import dash_bootstrap_components as dbc
import config
from layout import input_styles, box_styles, button_styles

def create_selection(
    montage_radio_id,
    check_all_button_id,
    clear_all_button_id,
    channel_region_checkboxes_id,
    check_all_annotations_btn_id,
    clear_all_annotations_btn_id,
    delete_annotations_btn_id,
    annotation_checkboxes_id,
    delete_confirmation_modal_id,
    cancel_delete_btn_id,
    confirm_delete_btn_id,
    offset_decrement_id,
    offset_display_id,
    offset_increment_id,
    colors_radio_id
):
    return html.Div([

        # Montage Selection
        html.Div([
            html.Label(
                "Select Montage:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            dcc.RadioItems(
                id=montage_radio_id,
                options=[],
                inline=False,
                style={"margin": "10px 0", "fontSize": "12px"},
                persistence=True,
                persistence_type="local"
            ),
        ], style=box_styles["classic"]),

        # Channel Selection
        html.Div([
            html.Label(
                "Select Channels:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            # Button container with flexbox to align buttons side by side
            html.Div([
                dbc.Button(
                    "Check All",
                    id=check_all_button_id,
                    color="success",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style={
                        "fontSize": "12px",
                        "padding": "6px 12px",
                        "borderRadius": "5px",
                        "width": "48%",  # Adjusted width to fit side by side
                    }
                ),
                dbc.Button(
                    "Clear All",
                    id=clear_all_button_id,
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style={
                        "fontSize": "12px",
                        "padding": "6px 12px",
                        "borderRadius": "5px",
                        "width": "48%",  # Adjusted width to fit side by side
                    }
                ),
            ], style={"display": "flex", "justifyContent": "space-between", "gap": "4%"}),  # Align buttons side by side

            dcc.Checklist(
                id=channel_region_checkboxes_id,
                options=[
                    {
                        'label': f"{region_code} ({len(channels)})",
                        'value': region_code
                    }
                    for region_code, channels in config.GROUP_CHANNELS_BY_REGION.items()
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
            # Button container with flexbox to align buttons side by side
            html.Div([
                dbc.Button(
                    "Check All",
                    id=check_all_annotations_btn_id,
                    color="success",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style={
                        "fontSize": "12px",
                        "padding": "6px 12px",
                        "borderRadius": "5px",
                        "width": "48%",  # Adjusted width
                    }
                ),
                dbc.Button(
                    "Clear All",
                    id=clear_all_annotations_btn_id,
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    style={
                        "fontSize": "12px",
                        "padding": "6px 12px",
                        "borderRadius": "5px",
                        "width": "48%",  # Adjusted width
                    }
                ),
            ], style={"display": "flex", "justifyContent": "space-between", "gap": "4%", "marginBottom": "6px"}),  # Align buttons side by side

            dbc.Button(
                "Delete Selected",
                id=delete_annotations_btn_id,
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
                id=annotation_checkboxes_id,
                inline=False,
                style={"margin": "10px 0", "fontSize": "12px"},
                persistence=True,
                persistence_type="local"
            ),

            # Confirmation Modal
            dbc.Modal(
                [
                    dbc.ModalHeader("Confirm Deletion"),
                    dbc.ModalBody(id="delete-modal-body", children="Are you sure you want to delete the selected annotations?"),
                    dbc.ModalFooter([
                        dbc.Button("Cancel", id=cancel_delete_btn_id, color="secondary", n_clicks=0),
                        dbc.Button("Delete", id=confirm_delete_btn_id, color="danger", n_clicks=0)
                    ])
                ],
                id=delete_confirmation_modal_id,
                is_open=False,
            )

        ], style=box_styles["classic"]),

        # Offset Selection
        html.Div([
            html.Label(
                "Select Amplitude:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            html.Div(
                children=[
                    dbc.Button("-", id=offset_decrement_id, color="primary", size="sm", n_clicks=0),
                    html.Span(
                        id=offset_display_id,
                        children="5",  # Default offset value
                        style={
                            "margin": "0 10px",  # Space between buttons
                            "fontWeight": "bold",
                            "fontSize": "12px"
                        }
                    ),
                    dbc.Button("+", id=offset_increment_id, color="primary", size="sm", n_clicks=0),
                ],
                style={
                    "display": "flex", 
                    "alignItems": "center", 
                    "gap": "12px"  # Space between elements
                }
            ),
        ], style=box_styles["classic"]),

        # Montage Selection
        html.Div([
            html.Label(
                "Select Colors:",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            dcc.RadioItems(
                id=colors_radio_id,
                options=[{'label': 'blue', 'value': 'blue'}, {'label': 'rainbow', 'value': 'rainbow'}],
                value='rainbow',
                inline=False,
                style={"margin": "10px 0", "fontSize": "12px"},
                persistence=True,
                persistence_type="local"
            ),
        ], style=box_styles["classic"]),

    ])
