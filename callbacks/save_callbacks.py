import dash
from dash import Input, Output, State, callback
from callbacks.utils import markerfile_utils as mu
from callbacks.utils import annotation_utils as au



def register_enter_default_saving_folder_path():
    @callback(
        Output("saving-folder-path-dropdown", "value"),
        Input("folder-store", "data"),
        prevent_initial_call = False
    )
    def enter_default_saving_folder(folder_path):
        if folder_path:
            return folder_path
        
def register_callbacks_annotations_to_save_names():
    # Callback to populate the checklist options and default value dynamically
    @callback(
        Output("annotations-to-save-checkboxes", "options"),
        Output("annotations-to-save-checkboxes", "value"),
        Input("annotations-store", "data"),
        prevent_initial_call = False
    )
    def display_annotation_names_to_save_checklist(annotations_store):
        if not annotations_store:
            return dash.no_update, dash.no_update
        
        description_counts = au.get_annotation_descriptions(annotations_store)

        options = [{'label': f"{name} ({count})", 'value': f"{name}"} for name, count in description_counts.items()]
        value = [f"{name}" for name in description_counts.keys()]
        return options, value  # Set all annotations as default selected
        
def register_manage_annotations_to_save_checklist():
    @callback(
        Output("annotations-to-save-checkboxes", "value", allow_duplicate=True),
        [Input("check-all-annotations-to-save-btn", "n_clicks"),
        Input("clear-all-annotations-to-save-btn", "n_clicks")],
        State("annotations-to-save-checkboxes", "options"),
        prevent_initial_call = True
    )
    def manage_annotation_names_to_save_checklist(check_all_clicks, clear_all_clicks, options):
        # Determine which button was clicked
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "check-all-annotations-to-save-btn":
            all_options = [option['value'] for option in options]
            return all_options  # Select all regions
        elif triggered_id == "clear-all-annotations-to-save-btn":
            return []  # Clear all selections

        return dash.no_update
    
def register_save_new_markerfile():
    # Callback function to save the annotation file
    @callback(
        Output("saving-mrk-status", "children"),  # Display a message in the saving status area
        Input("save-annotation-button", "n_clicks"),  # Trigger when the Save button is clicked
        State("saving-folder-path-dropdown", "value"),  # Get the selected folder path from the dropdown
        State("old-mrk-name", "value"),
        State("new-mrk-name", "value"),
        State("annotations-to-save-checkboxes", "value"),
        State("annotations-store", "data")  # Assuming annotations are stored somewhere
    )
    def save_annotation_file(n_clicks, folder_path, old_mrk_name, new_mrk_name, annotations_to_save, annotations):
        if n_clicks > 0:
            # Check if folder path and annotations are valid
            if not folder_path:
                return "Error: No folder path selected."
            if not annotations:
                return "Error: No annotations found."
            if annotations_to_save == []:
                return dash.no_update
            
            # Rename old marker file to OldMarkerFile.mrk
            mu.modify_name_oldmarkerfile(folder_path, old_mrk_name)

            # Save the new marker file
            try:
                mu.save_mrk_file(folder_path, old_mrk_name, new_mrk_name, annotations_to_save, annotations)
                return "File saved successfully!"
            except Exception as e:
                return f"Error saving the file: {str(e)}"
        return ""
        

