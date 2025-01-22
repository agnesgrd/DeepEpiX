import dash
from dash import Input, Output, State, html
import traceback
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash import Patch
import callbacks.utils.graph_utils as gu
import pandas as pd
from callbacks.utils import history_utils as hu

def register_enable_delete_spike_button():
    @dash.callback(
        Output("delete-spike-button", "disabled"),
        Input("meg-signal-graph", "selectedData"),
        prevent_initial_call=True
    )
    def enable_delete_spike_button(selected_data):
        if selected_data['range']['x'] is not None:
            return False  # Enable the button if both inputs are provided
        return True  # Disable the button if either input is missing
    
def register_enable_add_spike_button():
    @dash.callback(
        Output("add-spike-button", "disabled"),
        Input("spike-name", "value"),
        Input("spike-timestep", "value")
    )
    def enable_add_spike_button(name, timestep):
        if name is not None and timestep is not None:
            return False  # Enable the button if both inputs are provided
        return True  # Disable the button if either input is missing

def register_middle_time_on_selection():   
    @dash.callback(
        #Output("topomap-timepoint", "value"),
        Output("spike-timestep", "value"),
        [Input("meg-signal-graph", "selectedData")]  # Capture selection data from the graph
    )
    def update_range_on_selection(selected_data):
        if selected_data:
        # Get the selected range (from selectedData)
            try:
                x_range = selected_data['range']['x']  # Extract the x values (time points)
                middle = round((x_range[1] + x_range[0])/2, 3)  # Get the middletime value from the selection
                return middle  # Update the min and max range for the topomap
            except (KeyError, TypeError, IndexError):
                return dash.no_update
        else:
            return dash.no_update  # Default range if no selection has been made
        
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

        return fig_patch
    
def register_add_spike_to_annotation():
    @dash.callback(
        Output("annotations-store", "data", allow_duplicate=True),
        Output("history-store", "data", allow_duplicate=True),
        Input("add-spike-button", "n_clicks"),
        State("spike-name", "value"),
        State("spike-timestep", "value"),
        State("annotations-store", "data"),
        State("history-store", "data"),
        prevent_initial_call=True
    )
    def add_spike_to_annotation(n_clicks, spike_name, spike_timestep, annotations_data, history_data):
        # Validate input
        if n_clicks is None or n_clicks == 0:
            return dash.no_update, dash.no_update
        if not spike_name or spike_timestep is None:
            return dash.no_update, dash.no_update

        # Convert annotations data to DataFrame if not empty
        if not annotations_data:

            # Initialize empty DataFrame if no data exists
            annotations_data = []

        # Create a new row for the spike annotation
        annotations_data.append({"onset": spike_timestep, "duration": 0, "description": spike_name})

        action = f"Added a spike <{spike_name}> at {spike_timestep} (s).\n"
        history_data = hu.fill_history_data(history_data, action)

        # Convert the updated DataFrame back to a records-based format
        return annotations_data, history_data
    
def register_delete_selected_spike():
    @dash.callback(
        Output("annotations-store", "data", allow_duplicate=True),
        Output("history-store", "data", allow_duplicate=True),
        Input("delete-spike-button", "n_clicks"),
        State("meg-signal-graph", "selectedData"),
        State("annotations-store", "data"),
        State("annotation-checkboxes", "value"),
        State("history-store", "data"),
        prevent_initial_call=True
    )
    def delete_selected_spike(n_clicks, selected_data, annotations_data, visible_annotations, history_data):
        # Validate input
        if n_clicks is None or n_clicks == 0:
            return dash.no_update, dash.no_update
        if selected_data is None:
            return dash.no_update, dash.no_update
        
        # Get the selected range (from selectedData)
        try:
            x_range = selected_data['range']['x']  # Extract the x values (time points)
            x_min, x_max = x_range[0], x_range[1]
        except (KeyError, TypeError, IndexError):
            return dash.no_update, dash.no_update

        # Convert annotations data to DataFrame
        if annotations_data:
            annotations_df = pd.DataFrame.from_records(annotations_data)
        else:
            return dash.no_update, dash.no_update

        # Validate the 'onset' column
        if "onset" not in annotations_df.columns:
            return dash.no_update, dash.no_update
        
        # Identify rows to be deleted (spikes that fall within x_range and match visible annotations)
        spikes_to_delete = annotations_df[
            (annotations_df['onset'] >= x_min) & 
            (annotations_df['onset'] <= x_max) & 
            (annotations_df['description'].isin(visible_annotations))
        ]

        # Log actions for each deleted spike
        for _, spike_deleted in spikes_to_delete.iterrows():
            spike_description = spike_deleted.get('description', 'Unknown description')  # Safely retrieve 'description'
            spike_timestep = spike_deleted.get('onset', 'Unknown timestep')  # Safely retrieve 'onset'
            action = f"Deleted a spike <{spike_description}> at {spike_timestep} (s).\n" # Log or store the action as needed
            history_data = hu.fill_history_data(history_data, action)

        # Filter out the identified rows from the original DataFrame
        annotations_df = annotations_df.drop(spikes_to_delete.index)

        # Convert the updated DataFrame back to a records-based format
        updated_annotations = annotations_df.to_dict(orient="records")

        return updated_annotations, history_data


    


