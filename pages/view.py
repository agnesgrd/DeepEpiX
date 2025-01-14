# view.py
import dash
from dash import html
from dash_extensions import Keyboard
import layout.graph_layout as gl
from callbacks.graph_callbacks import register_update_graph_time_channel, register_update_annotations, register_callbacks_annotation_names, register_manage_channels_checklist, register_move_time_slider, register_update_annotation_graph
from callbacks.topomap_callbacks import register_display_topomap, register_close_topomap
from callbacks.topomap_callbacks import register_display_topomap_video, register_close_topomap_video

dash.register_page(__name__)


layout = html.Div([

    Keyboard(
        captureKeys=["ArrowLeft", "ArrowRight"],  # Captures ArrowLeft and ArrowRight keys
        id="keyboard"
    ),

    gl.get_graph_layout(),
    
    html.Div(id="python-error", style={"padding": "10px", "font-style": "italic", "color": "#555"})

])

register_callbacks_annotation_names()

register_update_graph_time_channel()

register_update_annotations()

register_move_time_slider()

register_manage_channels_checklist()

register_update_annotation_graph()

register_display_topomap()

register_close_topomap()

register_display_topomap_video()

register_close_topomap_video()



