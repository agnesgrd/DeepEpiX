import dash
from dash import Input, Output, State, callback
import pandas as pd
from callbacks.utils import history_utils as hu

def register_enable_delete_event_button():
    @callback(
        Output("delete-event-button", "disabled"),
        Input("meg-signal-graph", "selectedData"),
        prevent_initial_call=True
    )
    def enable_delete_event_button(selected_data):
        if selected_data is None:
            return dash.no_update
        if 'range' not in selected_data.keys():
            return dash.no_update
        if selected_data['range']['x'] is not None:
            return False  # Enable the button if both inputs are provided
        return True  # Disable the button if either input is missing
    
def register_enable_add_event_button():
    @callback(
        Output("add-event-button", "disabled"),
        Input("event-name", "value"),
        Input("event-onset", "value"),
        Input("event-duration", "value"),
        prevent_initial_call=False
    )
    def _enable_add_event_button(name, onset, duration):
        if None in (name, onset, duration):
            return True
        else:
            return False

def register_add_event_onset_duration_on_click():   
    @callback(
        Output("event-onset", "value"),
        Output("event-duration", "value"),
        Input('meg-signal-graph', 'clickData'),
        Input('meg-signal-graph', 'selectedData'),
        prevent_initial_call=True
    )
    def _update_event_onset_duration_on_click(click_info, selected_data):
        trigger_id = dash.ctx.triggered[0]['prop_id'].split('.')[0]

        try:
            if trigger_id == 'meg-signal-graph' and selected_data:
                # Triggered by selection
                start_time = selected_data['range']['x'][0]
                end_time = selected_data['range']['x'][1]
                duration = end_time - start_time
                return round(start_time, 3), round(duration, 3)

            elif trigger_id == 'meg-signal-graph' and click_info:
                # Triggered by click
                t = click_info["points"][0]['x']
                return round(t, 3), 0
            
        except (KeyError, TypeError, IndexError):
            return dash.no_update, dash.no_update

        else:
            # If neither clickData nor selectedData is available, return default
            return dash.no_update, dash.no_update
            
def register_add_event_to_annotation():
    @callback(
        Output("annotation-store", "data", allow_duplicate=True),
        Output("history-store", "data", allow_duplicate=True),
        Output("annotation-checkboxes", "value", allow_duplicate=True),
        Input("add-event-button", "n_clicks"),
        State("event-name", "value"),
        State("event-onset", "value"),
        State("event-duration", "value"),  # New input for duration
        State("annotation-store", "data"),
        State("history-store", "data"),
        State("annotation-checkboxes", "value"),
        prevent_initial_call=True
    )
    def _add_event_to_annotation(n_clicks, spike_name, spike_timestep, spike_duration, annotations_data, history_data, checkbox_values):
        # Validate input
        if n_clicks is None or n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update
        if not spike_name or spike_timestep is None:
            return dash.no_update, dash.no_update, dash.no_update

        # Ensure that spike_duration is not None or NaN (default it to 0 if no duration is provided)
        if spike_duration is None or spike_duration < 0:
            spike_duration = 0

        # Convert annotations data to DataFrame if not empty
        if not annotations_data:
            annotations_data = []

        # Create a new row for the spike annotation with duration
        annotations_data.append({"onset": spike_timestep, "duration": spike_duration, "description": spike_name})

        # Action to log the history (adding an event with the name and time)
        action = f"Added an event <{spike_name}> at {spike_timestep} (s) with duration {spike_duration} (s).\n"
        history_data = hu.fill_history_data(history_data, "annotations", action)

        # Ensure checkbox_values is a list before appending
        if checkbox_values is None:
            checkbox_values = []
        if spike_name not in checkbox_values:
            checkbox_values.append(spike_name)

        # Return updated data for annotations, history, and checkboxes
        return annotations_data, history_data, checkbox_values
    
def register_delete_selected_spike():
    @callback(
        Output("annotation-store", "data", allow_duplicate=True),
        Output("history-store", "data", allow_duplicate=True),
        Input("delete-event-button", "n_clicks"),
        State("meg-signal-graph", "selectedData"),
        State("annotation-store", "data"),
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
            action = f"Deleted an event <{spike_description}> at {spike_timestep} (s).\n" # Log or store the action as needed
            history_data = hu.fill_history_data(history_data, "annotations", action)

        # Filter out the identified rows from the original DataFrame
        annotations_df = annotations_df.drop(spikes_to_delete.index)

        # Convert the updated DataFrame back to a records-based format
        updated_annotations = annotations_df.to_dict(orient="records")

        return updated_annotations, history_data


    


