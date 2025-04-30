# Dash & Plotly
import dash
from dash import Input, Output, State, callback

# External Libraries
import mne
import os

# Local Imports
from layout import flexDirection
from callbacks.utils import folder_path_utils as fpu

def register_update_dropdown():
    @callback(
        Output("folder-path-dropdown", "options"),
        Output("folder-path-dropdown", "value"),
        Output("folder-path-warning", "children"),  # Optional: warning display
        Input("open-folder-button", "n_clicks"),
        State("folder-path-dropdown", "options"),
        prevent_initial_call=True
    )
    def update_dropdown(n_clicks, folder_path_list):
        """Update dropdown when a folder is selected via file explorer."""
        if n_clicks > 0:
            folder_path = fpu.browse_folder()

            # Init options if None
            if folder_path_list is None:
                folder_path_list = []

            if folder_path:
                if not fpu.test_ds_folder(folder_path):
                    return dash.no_update, dash.no_update, "Selected folder is not a valid .ds MEG folder."

                # Prevent duplicates
                if not any(option["value"] == folder_path for option in folder_path_list):
                    folder_path_list.append({
                        "label": fpu.get_ds_folder(folder_path),
                        "value": folder_path
                    })

                return folder_path_list, folder_path, ""  # Clear warning if successful

        return dash.no_update, dash.no_update, dash.no_update

def register_handle_valid_folder_path():
    @callback(
        Output("load-button", "disabled"),
        Output("frequency-container", "style"),
        Output("folder-path-warning", "children", allow_duplicate=True),
        Input("folder-path-dropdown", "value"),
        prevent_initial_call=True
    )
    def handle_valid_folder_path(folder_path):
        """Validate folder path and show warning if invalid."""
        if folder_path:
            if not os.path.isdir(folder_path):
                return True, {"display": "none"}, "Selected path is not a valid folder."

            if not folder_path.endswith(".ds"):
                return True, {"display": "none"},"Folder must end with '.ds' to be a valid MEG directory."

            try:
                mne.io.read_raw_ctf(folder_path, preload=False, verbose=False)
                return False, {"display": "none"}, ""  # Valid: enable button and clear warning
            except Exception as e:
                return True, {"display": "none"}, f"Invalid MEG folder: {str(e)}"

        return True, {"display": "none"}, "Please select a folder."

def register_store_folder_path_and_clear_data():
    @callback(
        Output("frequency-container", "style", allow_duplicate=True),
        Output("folder-store", "data"),
        Output("chunk-limits-store", "clear_data"),
        Output("frequency-store", "clear_data"),
        Output("annotations-store", "clear_data"),
        Output('model-probabilities-store', 'clear_data'),
        Output('sensitivity-analysis-store', 'clear_data'),
        Input("load-button", "n_clicks"),
        State("folder-path-dropdown", "value"),
        prevent_initial_call=True
    )
    def store_folder_path_and_clear_data(n_clicks, folder_path):
        """Clear all stores and display frequency section on load."""
        if not folder_path:
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)
        return (
            {**flexDirection["row-flex"], "display": "flex"},
            folder_path, 
            True, True, True, True, True
        )

def register_populate_tab_contents():
    @callback(
        Output("raw-info-container", "children"),
        Output("event-stats-container", "children"),
        Input("tabs", "active_tab"),
        Input("folder-store", "data"),
        prevent_initial_call=False
    )
    def populate_tab_contents(selected_tab, folder_path):
        """Populate tab content based on selected tab and stored folder path."""
        if not folder_path:
            return dash.no_update, dash.no_update

        raw_info_content = dash.no_update
        event_stats_content = dash.no_update

        if selected_tab == "raw-info-tab":
            raw_info_content = fpu.build_table_raw_info(folder_path)

        if selected_tab == "events-tab":
            event_stats_content = fpu.build_table_events_statistics(folder_path)

        return raw_info_content, event_stats_content