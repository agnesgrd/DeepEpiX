import dash
from dash import Input, Output, State
import pandas as pd
from callbacks.utils import history_utils as hu

def register_enable_delete_spike_button():
    @dash.callback(
        Output("delete-spike-button", "disabled"),
        Input("meg-signal-graph", "selectedData"),
        prevent_initial_call=True
    )
    def enable_delete_spike_button(selected_data):
        if selected_data is None:
            return dash.no_update
        if 'range' not in selected_data.keys():
            return dash.no_update
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

def register_add_spike_timestep_on_click():   
    @dash.callback(
        #Output("topomap-timepoint", "value"),
        Output("spike-timestep", "value"),
        Input('meg-signal-graph', 'clickData'),  # Capture selection data from the graph
        prevent_initial_call = True
    )
    def update_spike_timestep_on_click(click_info):
        if click_info:
        # Get the selected range (from selectedData)
            try:
                t = click_info["points"][0]['x']
                return round(t , 3) 
            except (KeyError, TypeError, IndexError):
                return dash.no_update
        else:
            return dash.no_update  # Default range if no selection has been made
            
def register_add_spike_to_annotation():
    @dash.callback(
        Output("annotations-store", "data", allow_duplicate=True),
        Output("history-store", "data", allow_duplicate=True),
        Output("annotation-checkboxes", "value", allow_duplicate=True),
        Input("add-spike-button", "n_clicks"),
        State("spike-name", "value"),
        State("spike-timestep", "value"),
        State("annotations-store", "data"),
        State("history-store", "data"),
        State("annotation-checkboxes", "value"),
        prevent_initial_call=True
    )
    def add_spike_to_annotation(n_clicks, spike_name, spike_timestep, annotations_data, history_data, checkbox_values):
        # Validate input
        if n_clicks is None or n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update
        if not spike_name or spike_timestep is None:
            return dash.no_update, dash.no_update, dash.no_update

        # Convert annotations data to DataFrame if not empty
        if not annotations_data:

            # Initialize empty DataFrame if no data exists
            annotations_data = []

        # Create a new row for the spike annotation
        annotations_data.append({"onset": spike_timestep, "duration": 0, "description": spike_name})

        action = f"Added a spike <{spike_name}> at {spike_timestep} (s).\n"
        history_data = hu.fill_history_data(history_data, action)

        # Ensure checkbox_values is a list before appending
        if checkbox_values is None:
            checkbox_values = []
        if spike_name not in checkbox_values:
            checkbox_values.append(spike_name)

        # Convert the updated DataFrame back to a records-based format
        return annotations_data, history_data, checkbox_values
    
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


    


