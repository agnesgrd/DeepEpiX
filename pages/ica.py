# view.py
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import os
import mne
import plotly.graph_objects as go
import numpy as np
import static.constants as c
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import graph_utils as gu
import plotly.express as px
import traceback
import pandas as pd
from callbacks.selection_callbacks import register_manage_annotations_checklist, register_callbacks_annotation_names, register_page_buttons_display, register_update_page_button_styles, register_callbacks_montage_names
from callbacks.annotation_callbacks import register_update_annotations, register_update_annotation_graph, register_move_to_next_annotation

# Layout imports
import layout.graph_layout as gl
from layout.ica_sidebar_layout import create_sidebar

dash.register_page(__name__, name="ICA", path='/viz/ica')


layout = html.Div([

    html.Div(
        [
        create_sidebar(),
        gl.create_graph_container(
            update_button_id="update-button-ica",
            update_container_id="update-container-ica",
            page_buttons_container_id="page-buttons-container-ica",
            page_selector_id="page-selector-ica",
            next_spike_buttons_container_id="next-spike-buttons-container-ica",
            prev_spike_id="prev-spike-ica",
            next_spike_id="next-spike-ica",
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
        "boxSizing": "border-box"
    })

])

# # Callback to generate page buttons based on the chunk limits in the store
# @callback(
#     Output("page-buttons-container-ica", "children"),
#     Input("chunk-limits-store", "data"),
#     prevent_initial_call=False
# )
# def update_buttons(chunk_limits):
#     if not chunk_limits:
#         return dash.no_update  # Default to the first page
    
#     return html.Div(
#         # RadioItems for the page buttons
#         dcc.RadioItems(
#             id="page-selector-ica",
#             options=[
#                 {"label": html.Span(
#                 str(i + 1),
#                 style={
#                     "textDecoration": "underline" if i == 0 else "none",  # Underline selected number
#                 })
#                 , "value": i} for i in range(len(chunk_limits))
#             ],
#             value=0,  # Default selected page
#             className="btn-group",  # Group styling
#             inputClassName="btn-check",  # Bootstrap class for hidden radio inputs
#             labelClassName="btn btn-outline-primary",  # Default button style
#             inputStyle={"display": "none"},  # Hide the input completely
#         ),
#     )

register_page_buttons_display(
    chunk_limits_store_id="chunk-limits-store",
    page_buttons_container_id="page-buttons-container-ica",
    page_selector_id="page-selector-ica"
)

register_update_page_button_styles(
    page_selector_id="page-selector-ica",
    chunk_limits_store_id="chunk-limits-store"
)

@callback(
    Output("ica-results", "children"),
    Input("compute-ica-button", "n_clicks"),  # Trigger the callback with the button
    State("folder-store", "data"),
    State("chunk-limits-store", "data"),
    State("n-components", "value"),  # Number of ICA components
    State("ica-method", "value"),  # ICA method selected ('fastica', 'infomax', etc.)
    State("max-iter", "value"),  # Max iterations for ICA fitting
    State("decim", "value"),  # Temporal decimation
    prevent_initial_call=True
)
def compute_ica(n_clicks, folder_path, chunk_limits, n_components, ica_method, max_iter, decim):
    """Update ICA signal visualization."""
    
    if n_clicks == 0:
        return dash.no_update
    
    if None in (folder_path, chunk_limits, n_components, ica_method, max_iter, decim):
        return dash.no_update
    
    # raw = mne.io.read_raw_ctf(folder_path, preload=True)  # Example for FIF format
    # raw = raw.pick_types(meg=True)  # Pick only MEG channels

    # # High-pass filtering to remove low-frequency drifts (1 Hz cutoff recommended)
    # raw = raw.filter(l_freq=1.0, h_freq=None)  # Apply 1 Hz high-pass filter

    # ica = mne.preprocessing.ICA(n_components=n_components, method=ica_method, max_iter=max_iter, random_state=97)

    # ica.fit(raw, decim=decim)
        
    for chunk_idx in chunk_limits:
        start_time, end_time = chunk_idx
        ica_df = pu.get_ica_sources_for_chunk(folder_path, start_time, end_time, n_components, ica_method, max_iter, decim)

    return html.Div([f"n_components = {n_components}, method: {ica_method}, max_iter: {max_iter}, decim: {1} \n, "])

# Callback to update the ICA graph
@callback(
    Output("graph-ica", "figure"),  # Trigger the callback to update the graph
    Input("update-button-ica", "n_clicks"),  # Button trigger
    Input("page-selector-ica", "value"),  # Page selection (if needed)
    State("folder-store", "data"),  # Folder path (for loading data)
    State("chunk-limits-store", "data"),  # Limits for chunking data
    State("n-components", "value"),  # Number of ICA components
    State("ica-method", "value"),  # ICA method selected ('fastica', 'infomax', etc.)
    State("max-iter", "value"),  # Max iterations for ICA fitting
    State("decim", "value"),  # Temporal decimation
    State("graph-ica", "figure"),  # MEG signal graph figure (to update)
    prevent_initial_call=True
)
def update_ica_graph(n_clicks, page_selection, folder_path, chunk_limits, n_components, ica_method, max_iter, decim, graph):
    """Update ICA signal visualization."""

    print(dash.callback_context.triggered_id)
    # Check if the graph is already populated
    if graph and 'data' in graph and graph['data']:  # if there's already data in the figure
        return graph
    
    if None in (page_selection, folder_path, chunk_limits, n_components, ica_method, max_iter, decim):
        return dash.no_update
    
    time_range = chunk_limits[int(page_selection)]

    # Get the current x-axis center
    xaxis_range = graph["layout"]["xaxis"].get("range", [])
    if xaxis_range[1] < time_range[0] or xaxis_range[0] > time_range[1]:
        xaxis_range = [time_range[0], time_range[0]+10]

    raw_df = pu.get_ica_sources_for_chunk(folder_path, time_range[0], time_range[1], n_components, ica_method, max_iter, decim)

    shifted_times = gu.get_shifted_time_axis(time_range, raw_df)
    
    selected_channels = np.arange(n_components)

    offset_selection = 5

    # Offset channel traces along the y-axis
    channel_offset = gu.calculate_channel_offset(len(selected_channels))*(11-offset_selection)*9 #/10 #/ 12
    channel_offset = 10
    y_axis_ticks = gu.get_y_axis_ticks(selected_channels, channel_offset)
    shifted_filtered_raw_df = raw_df + np.tile(y_axis_ticks, (len(raw_df), 1))

    shifted_filtered_raw_df["Time"] = shifted_times  # Add time as a column for Plotly Express

    fig = px.line(
        shifted_filtered_raw_df,
        x="Time",
        y=shifted_filtered_raw_df.columns[:-1],  # Exclude the Time column from y
        labels={"value": "Value", "variable": "Channel", "Time": "Time (s)"}
    )

    # Update layout with x-axis range and other customizations
    fig.update_layout(
        autosize=True,
        xaxis=dict(
            title='Time (s)',
            range=xaxis_range,
            minallowed=time_range[0],
            maxallowed=time_range[1],
            fixedrange=False,
            rangeslider=dict(visible=True, thickness=0.02),
            showspikes=True,
            spikemode="across+marker",
            spikethickness = 1,
        ),
        yaxis=dict(
            title='Channels',
            autorange=True,
            showgrid=True,
            tickvals=y_axis_ticks,
            ticktext=[f'{selected_channels[i]}' for i in range(len(selected_channels))],
            ticklabelposition="outside right",
            side="right",
            # automargin=True,
            spikethickness = 0
        ),
        title={
            'text': folder_path if folder_path else 'Select a folder path in Home Page',
            'x': 0.5,
            'font': {'size': 12},
            'automargin': True,
            'yref': 'paper',
        },
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode =  'select',
        selectdirection = 'h',
        hovermode = 'closest'
    )
    # Update the line width after creation


    fig.update_traces(line=dict(width=1))

    try:
        return fig
    
    except Exception as e:
        traceback.print_exec()
        return dash.no_update
    


register_update_annotation_graph(
    update_button_id="update-button-ica",
    page_selector_id="page-selector-ica",
    annotation_checkboxes_id="annotation-checkboxes-ica",
    annotations_store_id="annotations-store",
    annotation_graph_id="annotation-graph-ica",
    chunk_limits_store_id="chunk-limits-store"
)

register_manage_annotations_checklist(
    check_all_annotations_btn_id="check-all-annotations-btn-ica",
    clear_all_annotations_btn_id="clear-all-annotations-btn-ica",
    annotation_checkboxes_id="annotation-checkboxes-ica"
)

register_callbacks_annotation_names(
    annotation_checkboxes_id="annotation-checkboxes-ica",
    annotations_store_id="annotations-store"
)

register_update_annotations(
    graph_id="graph-ica",
    annotation_checkboxes_id="annotation-checkboxes-ica",
    page_selector_id="page-selector-ica",
    annotations_store_id="annotations-store",
    chunk_limits_store_id="chunk-limits-store"
)

register_move_to_next_annotation(
    prev_spike_id="prev-spike-ica",
    next_spike_id="next-spike-ica",
    graph_id="graph-ica",
    annotation_checkboxes_id="annotation-checkboxes-ica",
    annotations_store_id="annotations-store",
    page_selector_id="page-selector-ica",
    chunk_limits_store_id="chunk-limits-store"
)

register_callbacks_montage_names(
    montage_radio_id="montage-radio-ica"
)
