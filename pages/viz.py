# view.py
import dash
from dash import html
from dash_extensions import Keyboard

# Layout imports
import layout.graph_layout as gl
from layout.sidebar_layout import create_sidebar

# Callback imports
from callbacks.selection_callbacks import (
    register_page_buttons_display,
    register_update_page_button_styles,
    register_manage_channels_checklist,
    register_manage_annotations_checklist,
    register_offset_display,
    register_popup_annotation_suppression,
    register_cancel_or_confirm_annotation_suppression,
    register_callbacks_montage_names,
    register_callbacks_annotation_names,
    register_callbacks_annotation_names_dropdown,
    register_hide_channel_selection_when_montage,
    register_callbacks_sensivity_analysis
)


from callbacks.graph_callbacks import (
    register_update_graph_time_channel
)

from callbacks.annotation_callbacks import (
    register_update_annotations,
    register_update_annotation_graph,
    register_move_to_next_annotation
)

from callbacks.topomap_callbacks import (
    register_display_topomap_on_click,
    register_activate_deactivate_topomap_button
)
from callbacks.spike_callbacks import (
    register_add_spike_timestep_on_click,
    register_add_spike_to_annotation,
    register_delete_selected_spike,
    register_enable_add_spike_button,
    register_enable_delete_spike_button
)
from callbacks.history_callbacks import (
    register_update_history,
    register_clean_history
)

from callbacks.utils import history_utils

from callbacks.save_callbacks import (
    register_enter_default_saving_folder_path,
    register_callbacks_annotations_to_save_names,
    register_save_new_markerfile,
    register_manage_annotations_to_save_checklist
)

from callbacks.predict_callbacks import (
    register_execute_predict_script,
    register_store_display_prediction,
    register_update_selected_model
)

from callbacks.anom_detect_callbacks import (
    register_update_selected_model_anom_detect,
    register_execute_predict_script_anom_detect,
    register_display_anom_detect
)

dash.register_page(__name__, name="Data Viz & Analyze", path='/viz/raw-signal')

layout = html.Div([
    

    Keyboard(
        captureKeys=["ArrowLeft", "ArrowRight"],  # Captures ArrowLeft and ArrowRight keys
        id="keyboard"
    ),

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
        "boxSizing": "border-box"
    }),
    
    html.Div(id="python-error", style={"padding": "10px", "font-style": "italic", "color": "#555"})

])


register_page_buttons_display(
    chunk_limits_store_id="chunk-limits-store",
    page_buttons_container_id="page-buttons-container",
    page_selector_id="page-selector"
)

register_update_page_button_styles(
    page_selector_id="page-selector",
    chunk_limits_store_id="chunk-limits-store"
)

register_callbacks_annotation_names(
    annotation_checkboxes_id="annotation-checkboxes",
)

register_callbacks_annotation_names_dropdown(
    annotation_dropdown_id="annotation-dropdown",
    annotation_checkboxes_id="annotation-checkboxes"
)

register_callbacks_montage_names(
    montage_radio_id="montage-radio"
)

register_update_graph_time_channel()

register_update_annotations(
    graph_id="meg-signal-graph",
    annotation_checkboxes_id="annotation-checkboxes",
    page_selector_id="page-selector",
    chunk_limits_store_id="chunk-limits-store"
)

register_manage_channels_checklist()

register_update_annotation_graph(
    update_button_id="update-button",
    page_selector_id="page-selector",
    annotation_checkboxes_id="annotation-checkboxes",
    annotation_graph_id="annotation-graph",
    chunk_limits_store_id="chunk-limits-store"
)

register_hide_channel_selection_when_montage()

register_add_spike_timestep_on_click()

register_add_spike_to_annotation()

register_delete_selected_spike()

register_update_history()

register_clean_history()

register_enable_add_spike_button()

register_enable_delete_spike_button()

register_offset_display()

register_display_topomap_on_click()

register_activate_deactivate_topomap_button()

register_move_to_next_annotation(
    prev_spike_id="prev-spike",
    next_spike_id="next-spike",
    graph_id="meg-signal-graph",
    annotation_dropdown_id="annotation-dropdown",
    annotation_checkboxes_id="annotation-checkboxes",
    page_selector_id="page-selector",
    chunk_limits_store_id="chunk-limits-store"
)

register_callbacks_sensivity_analysis()

register_manage_annotations_checklist(
    check_all_annotations_btn_id="check-all-annotations-btn",
    clear_all_annotations_btn_id="clear-all-annotations-btn",
    annotation_checkboxes_id="annotation-checkboxes"
)


register_popup_annotation_suppression()

register_cancel_or_confirm_annotation_suppression()

register_enter_default_saving_folder_path()

register_save_new_markerfile()

register_manage_annotations_to_save_checklist()

register_callbacks_annotations_to_save_names()

register_execute_predict_script()

register_store_display_prediction()

register_update_selected_model()

register_update_selected_model_anom_detect()

register_execute_predict_script_anom_detect()

register_display_anom_detect()