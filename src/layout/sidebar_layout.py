from dash import html
import dash_bootstrap_components as dbc

from layout.selection_sidebar_layout import create_selection
from layout.analyze_sidebar_layout import create_analyze
from layout.predict_sidebar_layout import create_predict
from layout.save_sidebar_layout import create_save


def create_sidebar():
    return html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        create_selection(
                            check_all_annotations_btn_id="check-all-annotations-btn",
                            clear_all_annotations_btn_id="clear-all-annotations-btn",
                            delete_annotations_btn_id="delete-annotations-btn",
                            annotation_checkboxes_id="annotation-checkboxes",
                            delete_confirmation_modal_id="delete-confirmation-modal",
                            delete_modal_body_id="delete-modal-body",
                            cancel_delete_btn_id="cancel-delete-btn",
                            confirm_delete_btn_id="confirm-delete-btn",
                            create_intersection_btn_id="create-intersection-btn",
                            create_intersection_modal_id="create-intersection-modal",
                            create_intersection_modal_body_id="create-intersection-modal-body",
                            intersection_tolerance_id="intersection-tolerance",
                            cancel_intersection_btn_id="cancel-intersection-btn",
                            confirm_intersection_btn_id="confirm-intersection-btn",
                            offset_decrement_id="offset-decrement",
                            offset_display_id="offset-display",
                            offset_increment_id="offset-increment",
                            colors_radio_id="colors-radio",
                            montage_radio_id="montage-radio",
                            check_all_button_id="check-all-channels-btn",
                            clear_all_button_id="clear-all-channels-btn",
                            channel_region_checkboxes_id="channel-region-checkboxes",
                        ),
                        labelClassName="bi bi-hand-index-thumb",
                        tab_id="selection-tab",
                    ),
                    dbc.Tab(
                        create_analyze(),
                        labelClassName="bi bi-activity",
                        tab_id="analyzing-tab",
                    ),
                    dbc.Tab(
                        create_predict(),
                        labelClassName="bi bi-stars",
                        tab_id="prediction-tab",
                    ),
                    dbc.Tab(
                        create_save(),
                        labelClassName="bi bi-floppy",
                        tab_id="saving-tab",
                    ),
                ],
                id="sidebar-tabs",
                persistence=True,
                persistence_type="session",
                className="custom-sidebar",
            ),
        ],
        style={
            "padding": "10px 0",
            "height": "90vh",
            "display": "flex",
            "flexDirection": "column",
            "overflowY": "auto",
            "justifyContent": "flex-start",
            "gap": "20px",
            "minWidth": "250px",
            "boxSizing": "border-box",
            "fontSize": "12px",
            "borderRadius": "10px",
            "overflowX": "auto",
        },
    )
