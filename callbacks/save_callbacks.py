import dash
from dash import Input, Output, State, callback
from callbacks.utils import markerfile_utils as mu
from callbacks.utils import annotation_utils as au
from datetime import datetime

def register_enter_default_saving_folder_path():
    @callback(
        Output("saving-folder-path-dropdown", "value"),
        Input("folder-store", "data"),
        prevent_initial_call = False
    )
    def _enter_default_saving_folder(folder_path):
        if folder_path:
            return folder_path
        return None
        
def register_display_annotations_to_save_checkboxes():
    # Callback to populate the checklist options and default value dynamically
    @callback(
        Output("annotations-to-save-checkboxes", "options"),
        Output("annotations-to-save-checkboxes", "value"),
        Input("annotation-store", "data"),
        prevent_initial_call = False
    )
    def _display_annotations_to_save_checkboxes(annotations_store):
        if not annotations_store:
            return dash.no_update, dash.no_update
        
        description_counts = au.get_annotation_descriptions(annotations_store)
        options = [{'label': f"{name} ({count})", 'value': f"{name}"} for name, count in description_counts.items()]
        value = [f"{name}" for name in description_counts.keys()]
        return options, value  # Set all annotations as default selected

def register_set_old_markerfile_name():          
    @callback(
        Output("old-mrk-name", "value"),
        Input("sidebar-tabs", "active_tab"),  # Dummy trigger to run on load, or use another valid Input
        prevent_initial_call=False
    )
    def set_old_marker_name(active_tab):
        if active_tab=="saving-tab":
            date_str = datetime.now().strftime('%d.%m.%H.%M')
            return f"OldMarkerFile_{date_str}"
        return dash.no_update

def register_save_new_markerfile():
    @callback(
        Output("saving-mrk-status", "children"),  # Display a message in the saving status area
        Input("save-annotation-button", "n_clicks"),  # Trigger when the Save button is clicked
        State("saving-folder-path-dropdown", "value"),  # Get the selected folder path from the dropdown
        State("old-mrk-name", "value"),
        State("new-mrk-name", "value"),
        State("annotations-to-save-checkboxes", "value"),
        State("annotation-store", "data")  # Assuming annotations are stored somewhere
    )
    def _save_new_markerfile_file(n_clicks, folder_path, old_mrk_name, new_mrk_name, annotations_to_save, annotations):
        """Modify name of old markerfile and create new markerfile."""
        if n_clicks > 0:
            if not folder_path:
                return "⚠️ Error: No folder path selected."
            if not annotations:
                return "⚠️ Error: No annotations found."
            if annotations_to_save == []:
                return dash.no_update
            
            try:
                mu.modify_name_oldmarkerfile(folder_path, old_mrk_name)
                mu.save_mrk_file(folder_path, new_mrk_name, annotations_to_save, annotations)
                return "File saved successfully!"
            except Exception as e:
                return f"⚠️ Error saving the file: {str(e)}"
            
        return dash.no_update