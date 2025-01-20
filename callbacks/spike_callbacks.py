import dash
from dash import Input, Output, State, html
import traceback
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash import Patch
import callbacks.utils.graph_utils as gu
import pandas as pd



def register_middle_time_on_selection():   
    @dash.callback(
        Output("topomap-timepoint", "value"),
        Output("spike-timestep", "value"),
        [Input("meg-signal-graph", "selectedData")]  # Capture selection data from the graph
    )
    def update_range_on_selection(selected_data):
        print(selected_data)
        if selected_data:
            # Get the selected range (from selectedData)
            x_range = selected_data['range']['x']  # Extract the x values (time points)
            middle = round((x_range[1] + x_range[0])/2, 3)  # Get the middletime value from the selection
            return middle, middle  # Update the min and max range for the topomap
        else:
            return dash.no_update, dash.no_update  # Default range if no selection has been made
        
def register_plot_potential_spike():
    @dash.callback(
        Output("meg-signal-graph", "figure", allow_duplicate=True),
        Input("spike-timestep", "value"),
        State("meg-signal-graph", "figure"),  # Current figure to update
        prevent_initial_call=True
    )
    def plot_potential_spike(timestep, fig_dict):
        """Update annotations visibility based on the checklist selection."""
        # Default time range in case the figure doesn't contain valid x-axis range data
        time_range = [0, 180]

        # Create a Patch for the figure
        fig_patch = Patch()

        # Check if fig_dict is None (i.e., if it is the initial empty figure)
        if fig_dict is None or 'layout' not in fig_dict or 'yaxis' not in fig_dict['layout']:
            # Set default y_min and y_max if the figure layout is not available
            y_min, y_max = 0, 1  # Set default range for the y-axis
        else:
            # Get the current y-axis range from the figure
            y_min, y_max = fig_dict['layout']['yaxis'].get('range', [0, 1])

        # Prepare the shapes and annotations for the selected annotations
        new_shapes = [
            dict(
                type="line",
                x0=timestep,
                x1=timestep,
                y0=y_min,
                y1=y_max,
                xref="x",
                yref="y",
                line=dict(color="red", width=4, dash="dot"),
                opacity=0.25
            )
        ]
        # Add the label in the margin
        new_annotations = [
            dict(
                x=timestep,
                y=0.98,  # Slightly above the graph in the margin
                xref="x",
                yref="paper",  # Use paper coordinates for the y-axis (margins)
                text='new spike ?',  # Annotation text
                showarrow=False,  # No arrow needed
                font=dict(size=10, color="red"),  # Customize font
                align="center",
            )
        ]

        # Update the figure with the new shapes and annotations
        fig_patch["layout"]["shapes"] = new_shapes
        fig_patch["layout"]["annotations"] = new_annotations

        print("finished adding annotations on main graph")

        return fig_patch
    
def register_add_spike_to_annotation():
    @dash.callback(
        Output("annotations-store", "data", allow_duplicate=True),
        Output("spike-saving-status", "children"),
        Input("add-spike-button", "n_clicks"),
        State("spike-name", "value"),
        State("spike-timestep", "value"),
        State("annotations-store", "data"),
        prevent_initial_call=True
    )
    def add_spike_to_annotation(n_clicks, spike_name, spike_timestep, annotations_data):
        # Validate input
        if n_clicks is None or n_clicks == 0:
            return dash.no_update, ""
        if not spike_name or spike_timestep is None:
            return dash.no_update, "Error: Missing name or timestep value."

        # Convert annotations data to DataFrame if not empty
        if not annotations_data:

            # Initialize empty DataFrame if no data exists
            annotations_data = []

        # Create a new row for the spike annotation
        annotations_data.append({"onset": spike_timestep, "duration": 0, "description": spike_name})

        # Convert the updated DataFrame back to a records-based format
        return annotations_data, "Success: New spike added!"

