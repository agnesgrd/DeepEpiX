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
from callbacks.graph_callbacks import register_update_graph_time_channel, register_update_annotations, register_move_time_slider, register_callbacks_annotation_names
from callbacks.utils import annotation_utils as au


dash.register_page(__name__)


layout = html.Div([
    html.H1("VIEW: Visualize and Annotate MEG Signal"),

    # Display folder path
    html.Div(id="display-folder-path", style={"padding": "10px", "font-style": "italic", "color": "#555"}),

    # Sidebar or control panel for selecting annotation types
        html.Div([
            html.Label("Select Annotations:"),
            dcc.Checklist(
                id="annotation-checkboxes",
                # options=[{'label': name, 'value': name} for name in annotation_names],  # List of types
                # value=annotation_names,  # Default to showing all annotations
                inline=True,  # Display items inline
                style={"margin": "10px 0"},
                persistence=True,
                persistence_type="local"
            ),
        ], style={
            "padding": "10px",
            "width": "100%",
            "textAlign": "center"
        }),

    gl.get_graph_layout(),
    
    html.Div(id="python-error", style={"padding": "10px", "font-style": "italic", "color": "#555"})

])

register_callbacks_folder_path()

register_callbacks_annotation_names()

register_update_graph_time_channel()

register_update_annotations()

register_move_time_slider()

