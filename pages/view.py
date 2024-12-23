# view.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go  # Import Plotly for graphing
from plotly.subplots import make_subplots
from pages.home import get_preprocessed_dataframe
import pandas as pd
import static.constants as c
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB
import numpy as np
import traceback
import layout.graph_layout as gl
from callbacks.folder_path_callbacks import register_callbacks_folder_path
from callbacks.graph_callbacks import register_update_graph_time_channel, register_update_annotations, register_callbacks_annotation_names, register_manage_channels_checklist
from callbacks.utils import annotation_utils as au


dash.register_page(__name__)


layout = html.Div([

    gl.get_graph_layout(),
    
    html.Div(id="python-error", style={"padding": "10px", "font-style": "italic", "color": "#555"})

])

register_callbacks_annotation_names()

register_update_graph_time_channel()

register_update_annotations()

# register_move_time_slider()

register_manage_channels_checklist()

