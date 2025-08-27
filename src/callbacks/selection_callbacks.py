import os
import dash
from dash import html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Local import
import callbacks.utils.annotation_utils as au

# ─────────────────────────────
# 📄 Sidebar Callbacks
# ─────────────────────────────

def register_toggle_sidebar(collapse_id, icon_id, toggle_id):
    @callback(
        Output(collapse_id, "is_open"),
        Output(icon_id, "className"),
        Input(toggle_id, "n_clicks"),
        State(collapse_id, "is_open"),
        prevent_initial_call=True
    )
    def _toggle_sidebar(n_clicks, is_open):
        if is_open:
            return False, "bi bi-layout-sidebar-inset"  # icon for closed state
        else:
            return True, "bi bi-x-lg"  # icon for open state

def register_navigate_tabs_raw(collapse_id, sidebar_tabs_id, icon_id):
    @callback(
        Output(collapse_id, "is_open", allow_duplicate=True),    
        Output(sidebar_tabs_id, "active_tab", allow_duplicate=True),
        Output(icon_id, "className", allow_duplicate=True),
        Input("nav-select", "n_clicks"),
        Input("nav-analyze", "n_clicks"),
        Input("nav-predict", "n_clicks"),
        Input("nav-save", "n_clicks"),
        prevent_initial_call=True
    )
    def navigate_tabs(n1, n2, n3, n4):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        btn_id = ctx.triggered_id

        if btn_id == "nav-select":
            return True, "selection-tab", "bi bi-x-lg"
        elif btn_id == "nav-analyze":
            return True, "analyzing-tab", "bi bi-x-lg"
        elif btn_id == "nav-predict":
            return True, "prediction-tab", "bi bi-x-lg"
        elif btn_id == "nav-save":
            return True, "saving-tab", "bi bi-x-lg"

def register_navigate_tabs_ica(collapse_id, sidebar_tabs_id, icon_id):
    @callback(
        Output(collapse_id, "is_open", allow_duplicate=True),    
        Output(sidebar_tabs_id, "active_tab", allow_duplicate=True),
        Output(icon_id, "className", allow_duplicate=True),
        Input("nav-compute-ica", "n_clicks"),
        Input("nav-select-ica", "n_clicks"),
        prevent_initial_call=True
    )
    def navigate_tabs(n1, n2):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        btn_id = ctx.triggered_id

        if btn_id == "nav-compute-ica":
            return True, "compute-tab-ica", "bi bi-x-lg"
        elif btn_id == "nav-select-ica":
            return True, "select-tab-ica", "bi bi-x-lg"

# ─────────────────────────────
# 📄 Page Navigation Callbacks
# ─────────────────────────────

# def register_page_buttons_display(page_buttons_container_id, page_selector_id):
#     @callback(
#         Output(page_buttons_container_id, "children"),
#         Input("chunk-limits-store", "data"),
#         prevent_initial_call=False
#     )
#     def _display_page_buttons(chunk_limits):
#         """Display number of page depending on chunks length."""
#         if not chunk_limits:
#             return dash.no_update

#         return html.Div(
#             dbc.RadioItems(
#                 id=page_selector_id,
#                 options=[
#                     {
#                         "label": None,
#                         "value": i
#                     } for i in range(len(chunk_limits))
#                 ],
#                 value=0,
#                 inline=True,
#                 style={
#                     "minWidth": "24px",
#                     "height": "24px",
#                     "lineHeight": "1",
#                     "fontSize": "0.8rem"
#                 }
#             )
#         )

def register_page_buttons_display(page_selector_id):
    @callback(
        Output(page_selector_id, "options"),
        Input("chunk-limits-store", "data"),
        prevent_initial_call=False
    )
    def _display_page_buttons(chunk_limits):
        if not chunk_limits:
            return dash.no_update

        return [
            {"label": str(i+1), "value": i}
            for i in range(len(chunk_limits))
        ]

# ─────────────────────────────
# 📦 Channel & Annotation Management
# ─────────────────────────────

def register_update_channels_checklist_options(checkboxes_id):
    @callback(
        Output(checkboxes_id, "options"),
        Output(checkboxes_id, "value"),
        Input("channel-store", "data"),
        prevent_initial_call=False
    )
    def update_checklist_options(channel_data):
        if not channel_data:
            return [], []

        options = [
            {
                "label": f"{region} ({len(channels)})",
                "value": region
            }
            for region, channels in channel_data.items()
        ]

        # Example default: first two regions if available
        default_values = list(channel_data.keys())[:2]
        return options, default_values

def register_manage_channels_checklist(checkboxes_id):
    @callback(
        Output(checkboxes_id, "value", allow_duplicate=True),
        [Input("check-all-channels-btn", "n_clicks"),
         Input("clear-all-channels-btn", "n_clicks")],
        State("channel-store", "data"),
        prevent_initial_call=True
    )
    def manage_checklist(check_all_clicks, clear_all_clicks, channel_store):
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "check-all-channels-btn":
            return list(channel_store.keys())
        elif triggered_id == "clear-all-channels-btn":
            return []

        return dash.no_update


def register_clear_check_all_annotation_checkboxes(check_all_btn_id, clear_all_btn_id, checkboxes_id):
    @callback(
        Output(checkboxes_id, "value", allow_duplicate=True),
        [Input(check_all_btn_id, "n_clicks"),
         Input(clear_all_btn_id, "n_clicks")],
        State(checkboxes_id, "options"),
        prevent_initial_call=True
    )
    def _manage_checklist(check_all_clicks, clear_all_clicks, options):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == check_all_btn_id:
            return [option['value'] for option in options]
        elif triggered_id == clear_all_btn_id:
            return []

        return dash.no_update


# ─────────────────────────────
# 🧭 Offset Display
# ─────────────────────────────

def register_offset_display(offset_decrement_id, offset_increment_id, offset_display_id, keyboard_id):
    @callback(
        Output(offset_display_id, "value"),
        Input(offset_decrement_id, "n_clicks"),
        Input(offset_increment_id, "n_clicks"),
        Input(keyboard_id, "n_keydowns"),
        State(keyboard_id, "keydown"),
        State(offset_display_id, "value"),
        prevent_initial_call=True
    )
    def update_offset(decrement_clicks, increment_clicks, n_keydowns, keydown, current_offset):
        step = 1
        min_value, max_value = 1, 10
        try:
            offset = int(current_offset)
        except (TypeError, ValueError):
            offset = 5  # fallback default if current_offset is not valid

        triggered = dash.ctx.triggered_id
        if triggered == offset_increment_id:
            offset += step
        elif triggered == offset_decrement_id:
            offset -= step
        elif triggered==keyboard_id:
            key = keydown["key"]
            if key in ["+"]:
                offset += step
            elif key == "-":
                offset -= step

        offset = max(min_value, min(max_value, offset))
        return offset


# ─────────────────────────────
# ❌ Annotation Suppression Popup
# ─────────────────────────────

def register_modal_annotation_suppression(btn_id, checkboxes_id, modal_id, modal_body_id):
    @callback(
        Output(modal_id, "is_open"),
        Output(modal_body_id, "children"),
        Input(btn_id, "n_clicks"),
        State(checkboxes_id, "value"),
        State(modal_id, "is_open"),
        prevent_initial_call=True
    )
    def _open_delete_modal(delete_click, selected_annotations, is_open):
        if delete_click and selected_annotations:
            selected_text = ", ".join(selected_annotations)
            return True, f"Are you sure you want to delete: {selected_text}?"
        return is_open, dash.no_update


def register_cancel_or_confirm_annotation_suppression(confirm_btn_id, cancel_btn_id, checkboxes_id, modal_id):
    @callback(
        Output(modal_id, "is_open", allow_duplicate=True),
        Output("annotation-store", "data", allow_duplicate=True),
        Input(confirm_btn_id, "n_clicks"),
        Input(cancel_btn_id, "n_clicks"),
        State(checkboxes_id, "value"),
        State("annotation-store", "data"),
        State(modal_id, "is_open"),
        prevent_initial_call=True
    )
    def handle_delete_confirmation(confirm_click, cancel_click, selected_annotations, annotations_dict, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open, dash.no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == confirm_btn_id:
            annotations_dict = [a for a in annotations_dict if a['description'] not in selected_annotations]
            return False, annotations_dict
        if trigger_id == cancel_btn_id:
            return False, dash.no_update
        return is_open, dash.no_update

# ─────────────────────────────
# ❌ Intersection Creation Popup
# ─────────────────────────────
    
def register_toggle_intersection_modal(btn_id, checkboxes_id, modal_id, modal_body_id):    
    @callback(
        Output(modal_id, "is_open"),
        Output(modal_body_id, "children"),
        Input(btn_id, "n_clicks"),
        State(checkboxes_id, "value"),
        State(modal_id, "is_open"),
        prevent_initial_call = True
    )
    def toggle_intersection_modal(n_click, selected_annotations, is_open):
        if n_click and selected_annotations:
            selected_text = "∩".join(selected_annotations)
            return True, f"Are you sure you want to create new annotations from intersection: {selected_text}?"
        return is_open, dash.no_update
    
def register_create_intersection(confirm_btn_id, cancel_btn_id, checkboxes_id, tolerance_id, modal_id):
    @callback(
        Output(modal_id, "is_open", allow_duplicate=True),
        Output("annotation-store", "data", allow_duplicate=True),
        Input(confirm_btn_id, "n_clicks"),
        Input(cancel_btn_id, "n_clicks"),
        State(checkboxes_id, "value"),
        State(tolerance_id, "value"),
        State("annotation-store", "data"),
        State(modal_id, "is_open"),
        prevent_initial_call=True
    )
    def handle_delete_confirmation(confirm_click, cancel_click, selected_annotations, tolerance, annotations_dict, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open, dash.no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == confirm_btn_id:
            if tolerance is not None and selected_annotations and len(selected_annotations) >= 2:
                # 1. Get the selected annotations
                selected = [a for a in annotations_dict if a["description"] in selected_annotations]
                intersections = au.compute_multiway_intersections(selected, tolerance)
                annotations_dict+=intersections
                return False, annotations_dict

        if trigger_id == cancel_btn_id:
            return False, dash.no_update
        return is_open, dash.no_update


# ─────────────────────────────
# 🏷 Annotation Name Callbacks
# ─────────────────────────────

def register_annotation_checkboxes_options(checkboxes_id):
    @callback(
        Output(checkboxes_id, "options"),
        Input("annotation-store", "data"),
        prevent_initial_call=False
    )
    def _list_annotation_checkboxes_options(annotations_store):
        if not annotations_store:
            return dash.no_update

        desc_counts = au.get_annotation_descriptions(annotations_store)
        return [{'label': f"{name} ({count})", 'value': name} for name, count in desc_counts.items()]


def register_annotation_dropdown_options(dropdown_id, checkboxes_id):
    @callback(
        Output(dropdown_id, "options"),
        Input(checkboxes_id, "options"),
        Input(checkboxes_id, "value"),
        prevent_initial_call=False
    )
    def _update_annotation_dropdown(annotation_options, annotation_value):
        """Depending of which annotations is checked, update the dropdown that move to previous/next event."""
        if not annotation_value or not annotation_options:
            return dash.no_update
        
        valid_option_values = {opt["value"] for opt in annotation_options}
        filtered_values = [val for val in annotation_value if val in valid_option_values]
        return [{"label": "All Selected", "value": "__ALL__"}] + \
               [{"label": name, "value": name} for name in filtered_values]


# ─────────────────────────────
# 🎛 Montage Management
# ─────────────────────────────

def register_callbacks_montage_names(radio_id):
    @callback(
        Output(radio_id, "options"),
        Input("montage-store", "data"),
        prevent_initial_call=False
    )
    def display_montage_names(montage_store):
        if montage_store is None:
            raise PreventUpdate
        
        options = [{'label': k, 'value': k} for k in montage_store.keys()] + \
                  [{'label': 'channel selection', 'value': 'channel selection'}]

        return options


def register_hide_channel_selection_when_montage():
    @callback(
        Output("channel-region-checkboxes", "options", allow_duplicate=True),
        Input("montage-radio", "options"),
        Input("montage-radio", "value"),
        State("channel-region-checkboxes", "options"),
        prevent_initial_call=True
    )
    def toggle_channel_selection(montage_options, montage_value, channel_options):
        disabled = montage_value != 'channel selection'
        return [
            {**opt, "disabled": disabled} for opt in channel_options
        ]


# ─────────────────────────────
# 📊 Sensitivity Analysis
# ─────────────────────────────

def register_callbacks_sensivity_analysis():
    @callback(
        Output("colors-radio", "options"),
        Input("sensitivity-analysis-store", "data"),
        Input("colors-radio", "value"),
        State("colors-radio", "options"),
        prevent_initial_call=False
    )
    def update_sensitivity_options(sa_store, selected, current_options):
        if not sa_store:
            return dash.no_update

        sa_options = [{'label': os.path.basename(k[:-len("_smoothGrad.pkl")]), 'value': k} for k in sa_store or {}]
        new_options = [opt for opt in sa_options if opt not in current_options]
        return current_options + new_options