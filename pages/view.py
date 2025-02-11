# view.py
import dash
from dash import html
from dash_extensions import Keyboard
from dash.dependencies import Input, Output, State


# Layout imports
import layout.graph_layout as gl
from layout.sidebar_layout import create_sidebar

# Callback imports
from callbacks.selection_callbacks import (
    register_page_buttons_display,
    register_update_page_button_styles
)

from callbacks.graph_callbacks import (
    register_update_graph_time_channel,
    register_update_annotations,
    register_callbacks_montage_names,
    register_callbacks_annotation_names,
    register_manage_channels_checklist,
    register_move_time_slider,
    register_update_annotation_graph,
    register_hide_channel_selection_when_montage,
    register_offset_display,
    register_move_to_spike
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

dash.register_page(__name__)


layout = html.Div([
    

    Keyboard(
        captureKeys=["ArrowLeft", "ArrowRight"],  # Captures ArrowLeft and ArrowRight keys
        id="keyboard"
    ),

    html.Div(
        [
        create_sidebar(),
        gl.create_graph_container()
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


register_page_buttons_display()

register_update_page_button_styles()

register_callbacks_annotation_names()

register_callbacks_montage_names()

register_update_graph_time_channel()

register_update_annotations()

register_move_time_slider()

register_manage_channels_checklist()

register_update_annotation_graph()

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

register_move_to_spike()

