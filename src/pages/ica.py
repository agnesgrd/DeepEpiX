# view.py
import dash
from dash import html
from dash_extensions import Keyboard

import dash_bootstrap_components as dbc

from callbacks.selection_callbacks import (
    register_cancel_or_confirm_annotation_suppression,
    register_annotation_checkboxes_options,
    register_annotation_dropdown_options,
    register_clear_check_all_annotation_checkboxes,
    register_offset_display,
    register_page_buttons_display,
    register_popup_annotation_suppression
)

from callbacks.annotation_callbacks import (
    register_move_to_next_annotation,
    register_update_annotation_graph,
    register_update_annotations_on_graph,
)

from callbacks.history_callbacks import (
    register_update_ica_history
)

from callbacks.ica_callbacks import (
    register_compute_ica,
    register_fill_ica_results
)

from callbacks.graph_callbacks import (
    register_update_graph_ica
)

# Layout imports
import layout.graph_layout as gl
from layout.ica_sidebar_layout import create_sidebar

dash.register_page(__name__, name="ICA", path='/viz/ica')


layout = html.Div([

    Keyboard(
        id="keyboard-ica",
        captureKeys = ["ArrowRight", "ArrowLeft", "+", "-"]
    ),

    html.Div([
        # Sidebar container
        html.Div([

            # Collapsible sidebar
            dbc.Collapse(
                create_sidebar(),
                id="sidebar-collapse-ica",
                is_open=True,
                dimension="width",
                className="sidebar-collapse"
            ),

            # Button stack on the left
            html.Div([
                dbc.Button(html.I(id="sidebar-toggle-icon-ica", className="bi bi-x-lg"), id="toggle-sidebar", color="danger", size="sm", className="mb-2 shadow-sm"),

                dbc.Button(html.I(className="bi bi-noise-reduction"), id="nav-compute-ica", color="warning", size="sm", className="mb-2", title="Compute ICA"),
                dbc.Button(html.I(className="bi bi-hand-index-thumb"), id="nav-select-ica", color="primary", size="sm", className="mb-2", title="Select")
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "marginTop": "10px"
            }),
        ], style={
            "display": "flex",
            "flexDirection": "row",  # button -> collapse (left to right)
            "alignItems": "flex-start",
            "zIndex": 1000
        }),
        gl.create_graph_container(
            update_button_id="update-button-ica",
            update_container_id="update-container-ica",
            page_buttons_container_id="page-buttons-container-ica",
            page_selector_id="page-selector-ica",
            next_spike_buttons_container_id="next-spike-buttons-container-ica",
            prev_spike_id="prev-spike-ica",
            next_spike_id="next-spike-ica",
            annotation_dropdown_id="annotation-dropdown-ica",
            loading_id="loading-graph-ica",
            signal_graph_id="graph-ica",
            annotation_graph_id="annotation-graph-ica"
        )
    ], style={
        "display": "flex",  # Horizontal layout
        "flexDirection": "row",
        "height": "85vh",  # Use the full height of the viewport
        "width": "95vw",  # Use the full width of the viewport
        "overflow": "hidden",  # Prevent overflow in case of resizing
        "boxSizing": "border-box",
        "gap": "20px",
    }),

    html.Div(id="python-error-ica")

])


# Callback to update the ICA graph
register_compute_ica()
register_update_graph_ica(
    ica_result_radio_id="ica-result-radio"
)

# --- Same as RAW VIZ
register_cancel_or_confirm_annotation_suppression(
    confirm_btn_id="confirm-delete-btn-ica", 
    cancel_btn_id="cancel-delete-btn-ica", 
    checkboxes_id="annotation-checkboxes-ica", 
    modal_id="delete-confirmation-modal-ica")

register_annotation_checkboxes_options(
    checkboxes_id="annotation-checkboxes-ica",
)
register_annotation_dropdown_options(
    dropdown_id="annotation-dropdown-ica",
    checkboxes_id="annotation-checkboxes-ica"
)
register_clear_check_all_annotation_checkboxes(
    check_all_btn_id="check-all-annotations-btn-ica",
    clear_all_btn_id="clear-all-annotations-btn-ica",
    checkboxes_id="annotation-checkboxes-ica"
)
register_offset_display(
    offset_decrement_id="offset-decrement-ica", 
    offset_increment_id="offset-increment-ica", 
    offset_display_id="offset-display-ica",
    keyboard_id="keyboard-ica"
)
register_page_buttons_display(
    page_buttons_container_id="page-buttons-container-ica",
    page_selector_id="page-selector-ica"
)
register_popup_annotation_suppression(
    btn_id="delete-annotations-btn-ica",
    checkboxes_id="annotation-checkboxes-ica",
    modal_id="delete-confirmation-modal-ica",
    modal_body_id="delete-modal-body-ica")

register_move_to_next_annotation(
    prev_spike_id="prev-spike-ica",
    next_spike_id="next-spike-ica",
    graph_id="graph-ica",
    dropdown_id="annotation-dropdown-ica",
    checkboxes_id="annotation-checkboxes-ica",
    page_selector_id="page-selector-ica"
)

register_update_annotation_graph(
    update_button_id="update-button-ica",
    page_selector_id="page-selector-ica",
    checkboxes_id="annotation-checkboxes-ica",
    annotation_graph_id="annotation-graph-ica"
)

register_update_annotations_on_graph(
    graph_id="graph-ica",
    checkboxes_id="annotation-checkboxes-ica",
    page_selector_id="page-selector-ica"
)

register_update_ica_history()

register_fill_ica_results(
    ica_result_radio_id="ica-result-radio"
)
