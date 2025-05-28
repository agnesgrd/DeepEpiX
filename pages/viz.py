import dash
from dash import html, dcc
from dash_extensions import Keyboard

# Layout imports
import layout.graph_layout as gl
from layout.sidebar_layout import create_sidebar

# Callback imports

# --- Selection ---
from callbacks.selection_callbacks import (
    register_update_channels_checklist_options,
    register_cancel_or_confirm_annotation_suppression,
    register_annotation_checkboxes_options,
    register_annotation_dropdown_options,
    register_callbacks_montage_names,
    register_callbacks_sensivity_analysis,
    register_hide_channel_selection_when_montage,
    register_clear_check_all_annotation_checkboxes,
    register_manage_channels_checklist,
    register_offset_display,
    register_page_buttons_display,
    register_popup_annotation_suppression
)

# --- Graph ---
from callbacks.graph_callbacks import register_update_graph_raw_signal

# --- Annotation ---
from callbacks.annotation_callbacks import (
    register_move_to_next_annotation,
    register_update_annotation_graph,
    register_update_annotations_on_graph,
    register_move_with_keyboard
)

# --- Topomap ---
from callbacks.topomap_callbacks import (
    register_activate_deactivate_topomap_button,
    register_display_topomap_on_click,
)

# --- Spikes ---
from callbacks.spike_callbacks import (
    register_add_event_to_annotation,
    register_add_event_onset_duration_on_click,
    register_delete_selected_spike,
    register_enable_add_event_button,
    register_enable_delete_event_button,
)

# --- History ---
from callbacks.history_callbacks import (
    register_clean_annotation_history,
    register_update_annotation_history,
)

# --- Save ---
from callbacks.save_callbacks import (
    register_display_annotations_to_save_checkboxes,
    register_display_bad_channels_to_save_checkboxes,
    register_save_modifications,
)

# --- Predict ---
from callbacks.predict_callbacks import (
    register_execute_predict_script,
    register_store_display_prediction,
    register_update_selected_model,
)

dash.register_page(__name__, name="Data Viz & Analyze", path='/viz/raw-signal')

layout = html.Div([
    
    Keyboard(
        id="keyboard",
        captureKeys = ["ArrowRight", "ArrowLeft", "+", "-"]
    ),

    dcc.Location(id='url', refresh=False),

    html.Div(
        [
        create_sidebar(),
        gl.create_graph_container(
            update_button_id="update-button",
            update_container_id="update-container",
            page_buttons_container_id="page-buttons-container",
            page_selector_id="page-selector",
            next_spike_buttons_container_id="next-spike-buttons-container",
            prev_spike_id="prev-spike",
            next_spike_id="next-spike",
            loading_id="loading-graph",
            signal_graph_id="meg-signal-graph",
            annotation_graph_id="annotation-graph"
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
    
    html.Div(id="python-error")

])

# --- Page Navigation ---
register_page_buttons_display(
    page_buttons_container_id="page-buttons-container",
    page_selector_id="page-selector"
)

# --- Channels checklist ---
register_update_channels_checklist_options(
    checkboxes_id="channel-region-checkboxes"
)

register_manage_channels_checklist(
    checkboxes_id="channel-region-checkboxes"
)

register_hide_channel_selection_when_montage()

# --- Annotation Management ---
register_annotation_checkboxes_options(
    checkboxes_id="annotation-checkboxes",
)

register_annotation_dropdown_options(
    dropdown_id="annotation-dropdown",
    checkboxes_id="annotation-checkboxes"
)

register_clear_check_all_annotation_checkboxes(
    check_all_btn_id="check-all-annotations-btn",
    clear_all_btn_id="clear-all-annotations-btn",
    checkboxes_id="annotation-checkboxes"
)

register_update_annotations_on_graph(
    graph_id="meg-signal-graph",
    checkboxes_id="annotation-checkboxes",
    page_selector_id="page-selector"
)

register_update_annotation_graph(
    update_button_id="update-button",
    page_selector_id="page-selector",
    checkboxes_id="annotation-checkboxes",
    annotation_graph_id="annotation-graph"
)

register_display_annotations_to_save_checkboxes()

register_clear_check_all_annotation_checkboxes(
    check_all_btn_id="check-all-annotations-to-save-btn",
    clear_all_btn_id="clear-all-annotations-to-save-btn",
    checkboxes_id="annotations-to-save-checkboxes"
)

register_popup_annotation_suppression(
        btn_id="delete-annotations-btn",
        checkboxes_id="annotation-checkboxes",
        modal_id="delete-confirmation-modal",
        modal_body_id="delete-modal-body")

register_cancel_or_confirm_annotation_suppression(
    confirm_btn_id="confirm-delete-btn", 
    cancel_btn_id="cancel-delete-btn", 
    checkboxes_id="annotation-checkboxes", 
    modal_id="delete-confirmation-modal")

# --- Graph & Channel Handling ---
register_update_graph_raw_signal()
register_offset_display(
    offset_decrement_id="offset-decrement", 
    offset_increment_id="offset-increment", 
    offset_display_id="offset-display",
    keyboard_id="keyboard"
)

# --- Topomap Interactions ---
register_display_topomap_on_click()
register_activate_deactivate_topomap_button()

# --- Spike Handling ---
register_add_event_onset_duration_on_click()
register_add_event_to_annotation()
register_delete_selected_spike()
register_enable_add_event_button()
register_enable_delete_event_button()


register_move_with_keyboard(
    keyboard_id="keyboard",
    graph_id="meg-signal-graph",
    page_selector_id="page-selector"
)

register_move_to_next_annotation(
    prev_spike_id="prev-spike",
    next_spike_id="next-spike",
    graph_id="meg-signal-graph",
    dropdown_id="annotation-dropdown",
    checkboxes_id="annotation-checkboxes",
    page_selector_id="page-selector"
)

# --- History ---
register_update_annotation_history()
register_clean_annotation_history()

# --- Predict ---
register_execute_predict_script()
register_store_display_prediction()
register_update_selected_model()

# --- Save ---
register_display_bad_channels_to_save_checkboxes()
register_save_modifications()

# --- Analysis ---
register_callbacks_sensivity_analysis()

# --- Montage ---
register_callbacks_montage_names(
    radio_id="montage-radio"
)