import dash
from dash import Patch, Input, Output, State, callback
import callbacks.utils.dataframe_utils as du
from callbacks.utils import graph_utils as gu
import plotly.graph_objects as go
import pandas as pd
import itertools
from layout import COLOR_PALETTE
from dash.exceptions import PreventUpdate

def register_update_annotations_on_graph(
    graph_id,
    checkboxes_id,
    page_selector_id
):
    @callback(
        Output(graph_id, "figure", allow_duplicate=True),
        Input(checkboxes_id, "value"),  # Annotations to show
        Input(graph_id, "figure"),  # Current figure to update
        State(page_selector_id, "value"),
        State("annotation-store", "data"),
        State("chunk-limits-store", "data"),
        prevent_initial_call=True,
        # suppress_callback_exceptions=True
    )
    def _update_annotations_on_graph(annotations_to_show, fig_dict, page_selection, annotations, chunk_limits):
        """Update annotations visibility based on the checklist selection."""

        if not annotations_to_show or len(annotations_to_show) == 0 or not fig_dict['data']: # is None or 'layout' not in fig_dict or 'yaxis' not in fig_dict['layout']:
            fig_patch = Patch()
            fig_patch["layout"]["shapes"] = []
            fig_patch["layout"]["annotations"] = []
            return fig_patch
        
        if dash.callback_context.triggered[0]['prop_id'].split('.')[0] == graph_id and fig_dict['layout'].get('shapes', []):      
            return dash.no_update
    
        return gu.update_annotations_on_graph(fig_dict, annotations_to_show, page_selection, annotations, chunk_limits)
        
def register_update_annotation_graph(
    update_button_id,
    page_selector_id,
    checkboxes_id,
    annotation_graph_id
):
    @callback(
        Output(annotation_graph_id, "figure"),
        Input(update_button_id, "n_clicks"),
        Input(page_selector_id, "value"),
        State(checkboxes_id, "options"),
        State(checkboxes_id, "value"),
        State("annotation-store", "data"),
        State(annotation_graph_id, "figure"),
        State("chunk-limits-store", "data"),
        prevent_initial_call=True
    )
    def update_annotation_graph(n_clicks, page_selection, annotation_options, annotations_to_show, annotations_data, annotation_fig, chunk_limits):

        if not n_clicks or not annotations_data or not isinstance(annotations_data, list) or not annotations_to_show or not isinstance(annotations_to_show, list):
            return dash.no_update
        
        time_range = chunk_limits[int(page_selection)]

        annotations_df = pd.DataFrame(annotations_data).set_index("onset")

        # Create the annotation graph
        tick_vals = []
        tick_labels = []
        shapes = []

        for _, row in annotations_df.iterrows():
            if row["description"] in annotations_to_show:
                if row["duration"] == 0:
                    tick_vals.append(row.name)  # Use the onset time as the tick position
                    tick_labels.append(round(row.name, 2))

                if row["duration"] > 0:
                    shapes.append({
                        'type': 'rect',
                        'x0': row.name,
                        'x1': row.name + row["duration"],  # Duration defines the rectangle width
                        'y0': 0,  # Starting y-position of rectangle
                        'y1': 1,  # Ending y-position of rectangle
                        'line': {'color': 'rgba(0,0,255,0.6)', 'width': 2},  # Rectangle line color
                        'fillcolor': 'rgba(0,0,255,0.2)',  # Rectangle fill color
                    })

        # Update the figure with the new shapes and annotations
        fig_patch = go.Figure(annotation_fig)

        # Add styling improvements
        fig_patch.update_layout(
            xaxis=dict(
                range=time_range,
                showgrid=True,
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_labels,
                tickfont=dict(size=10),  # Smaller font size
                showticklabels=True,
                gridcolor="red",  # Subtle grid color
                tickangle=0  # Angle for better readability
            ),
            yaxis=dict(
                range=[0, 1],  # More generous y-axis range
                showgrid=False
            ),
            shapes=shapes,
            margin=dict(l=10, r=0, t=10, b=100)  # Adjust margins for better spacing
        )

        return fig_patch
    
def register_move_to_next_annotation(
    prev_spike_id,
    next_spike_id,
    graph_id,
    dropdown_id,
    checkboxes_id,
    page_selector_id
):
    @callback(
        Output(graph_id, "figure", allow_duplicate=True),
        Output(page_selector_id, "value"),
        Input(prev_spike_id, "n_clicks"),
        Input(next_spike_id, "n_clicks"),
        State(graph_id, "figure"),
        State(dropdown_id, "value"),
        State(checkboxes_id, "value"),
        State("annotation-store", "data"),
        State(page_selector_id, "value"),
        State("chunk-limits-store", "data"),
        prevent_initial_call=True
    )
    def _move_to_next_annotation(prev_spike, next_spike, graph, annotations_to_show, selected_annotations, annotations_data, page_selection, chunk_limits):

        if not annotations_data or not annotations_to_show:
            return dash.no_update, dash.no_update  # No annotations available, return the same graph
        if annotations_to_show == "__ALL__":
            annotations_to_show = selected_annotations
        if not annotations_to_show or len(annotations_data) == 0:
            return dash.no_update, dash.no_update

        # Extract x-coordinates (onset times) of spikes from annotations
        spike_x_positions = [
            ann["onset"] for ann in annotations_data if ann["description"] in annotations_to_show
        ]
        spike_x_positions = sorted(spike_x_positions)  # Ensure sorted order

        if not spike_x_positions:
            return dash.no_update, dash.no_update  # No spikes to navigate

        # Get the current x-axis center
        xaxis_range = graph["layout"]["xaxis"].get("range", [])
        if xaxis_range:
            current_x_center = sum(xaxis_range) / 2  # Midpoint of current view
        else:
            current_x_center = spike_x_positions[0]  # Default to first spike

        # Determine next or previous spike
        if dash.ctx.triggered_id == next_spike_id:
            next_spike_x = next((x for x in spike_x_positions if x > current_x_center), spike_x_positions[-1])
        elif dash.ctx.triggered_id == prev_spike_id:
            next_spike_x = next((x for x in reversed(spike_x_positions) if x < current_x_center), spike_x_positions[0])
        else:
            return dash.no_update, dash.no_update  # No valid button click
        
        # Find which page contains the target x
        for i, (start, end) in enumerate(chunk_limits):
            if start <= next_spike_x <= end:
                time_range_limits = chunk_limits[int(i)]
                new_page_selection = i
                
        # Extract time range limits
        time_range_min, time_range_max = time_range_limits

        # Compute x-axis range offset
        x_range_offset = (xaxis_range[1] - xaxis_range[0]) / 2 if xaxis_range else 10

        # Default centered range
        x_min = next_spike_x - x_range_offset
        x_max = next_spike_x + x_range_offset

        # Ensure the adjusted range is within valid limits
        x_min = max(x_min, time_range_min)
        x_max = min(x_max, time_range_max)

        # Update the graph layout
        graph["layout"]["xaxis"]["range"] = [x_min, x_max]

        if new_page_selection == page_selection:
            return graph, dash.no_update
        else:
            return graph, new_page_selection