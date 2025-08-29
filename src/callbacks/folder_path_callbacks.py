# Dash & Plotly
import dash
from dash import Input, Output, State, callback

# Local Imports
from callbacks.utils import folder_path_utils as fpu
from layout.config_layout import FLEXDIRECTION


def register_handle_valid_folder_path():
    @callback(
        Output("load-button", "disabled"),
        Output("preprocess-display-button", "disabled", allow_duplicate=True),
        Output("frequency-container", "style"),
        Output("folder-path-warning", "children", allow_duplicate=True),
        Input("folder-path-dropdown", "value"),
        prevent_initial_call=True,
    )
    def handle_valid_folder_path(folder_path):
        """Validate folder path and show warning if invalid."""
        if folder_path:
            if not fpu.test_ds_folder(folder_path):
                return (
                    True,
                    True,
                    {"display": "none"},
                    "Path must end with '.ds' or '.fif' or contain 3 files for 4D neuroimaging to be a valid raw MEG object.",
                )

            try:
                fpu.read_raw(folder_path, preload=False, verbose=False)
                return (
                    False,
                    True,
                    {"display": "none"},
                    "",
                )  # Valid: enable button and clear warning
            except Exception as e:
                return True, True, {"display": "none"}, f"Invalid MEG path: {str(e)}"

        return True, True, {"display": "none"}, "Please select a path."


def register_store_folder_path_and_clear_data():
    @callback(
        Output("frequency-container", "style", allow_duplicate=True),
        Output("preprocess-display-button", "disabled"),
        Output("folder-store", "data"),
        Output("chunk-limits-store", "clear_data"),
        Output("frequency-store", "clear_data"),
        Output("annotation-store", "clear_data"),
        Output("channel-store", "clear_data"),
        Output("model-probabilities-store", "clear_data"),
        Output("sensitivity-analysis-store", "clear_data"),
        Output("ica-store", "clear_data"),
        Input("load-button", "n_clicks"),
        State("folder-path-dropdown", "value"),
        prevent_initial_call=True,
    )
    def store_folder_path_and_clear_data(n_clicks, folder_path):
        """Clear all stores and display frequency section on load."""
        if not folder_path:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        return (
            {"display": "flex", **FLEXDIRECTION["row-flex"]},
            False,
            folder_path,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        )


def register_populate_tab_contents():
    @callback(
        Output("raw-info-container", "children"),
        Output("event-stats-container", "children"),
        Input("tabs", "active_tab"),
        Input("folder-store", "data"),
        prevent_initial_call=True,
    )
    def populate_tab_contents(selected_tab, folder_path):
        """Populate tab content based on selected tab and stored folder path."""
        if not folder_path or not selected_tab:
            return dash.no_update, dash.no_update

        raw_info_content = dash.no_update
        event_stats_content = dash.no_update

        if selected_tab == "raw-info-tab":
            raw_info_content = fpu.build_table_raw_info(folder_path)

        if selected_tab == "events-tab":
            event_stats_content = fpu.build_table_events_statistics(folder_path)

        return raw_info_content, event_stats_content
