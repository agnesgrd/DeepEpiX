import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Local import
import callbacks.utils.annotation_utils as au
import config

# ─────────────────────────────
# 📄 Page Navigation Callbacks
# ─────────────────────────────

def register_page_buttons_display(chunk_limits_store_id, page_buttons_container_id, page_selector_id):
    @callback(
        Output(page_buttons_container_id, "children"),
        Input(chunk_limits_store_id, "data"),
        prevent_initial_call=False
    )
    def _display_page_buttons(chunk_limits):
        """Display number of page depending on chunks length."""
        if not chunk_limits:
            return dash.no_update

        return html.Div(
            dbc.RadioItems(
                id=page_selector_id,
                options=[
                    {
                        "label": html.Span(
                            str(i + 1),
                            style={"textDecoration": "underline" if i == 0 else "none"}
                        ),
                        "value": i
                    } for i in range(len(chunk_limits))
                ],
                value=0,
                # className="btn-group",
                # inputClassName="btn-check",
                # labelClassName="btn btn-outline-primary",
                # inputStyle={"display": "none"},
                inline=True,
                style={
                    "minWidth": "24px",
                    "height": "24px",
                    "lineHeight": "1",
                    "fontSize": "0.8rem"
                }
            )
        )


# def register_update_page_button_styles(page_selector_id, chunk_limits_store_id):
#     @callback(
#         Output(page_selector_id, "options"),
#         Input(page_selector_id, "value"),
#         State(chunk_limits_store_id, "data"),
#         prevent_initial_call=True
#     )
#     def _update_button_styles(selected_value, chunk_limits):
#         """Underline page button when selected."""
#         if not chunk_limits:
#             return dash.no_update

#         return [
#             {
#                 "label": html.Span(
#                     str(i + 1),
#                     style={
#                     "textDecoration": "underline" if i == selected_value else "none"}
#                 ),
#                 "value": i
#             } for i in range(len(chunk_limits))
#         ]


# ─────────────────────────────
# 📦 Channel & Annotation Management
# ─────────────────────────────

def register_manage_channels_checklist():
    @callback(
        Output("channel-region-checkboxes", "value"),
        [Input("check-all-channels-btn", "n_clicks"),
         Input("clear-all-channels-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def manage_checklist(check_all_clicks, clear_all_clicks):
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "check-all-channels-btn":
            return list(config.GROUP_CHANNELS_BY_REGION.keys())
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

def register_offset_display():
    @callback(
        Output("offset-display", "children", allow_duplicate=True),
        Input("offset-decrement", "n_clicks"),
        Input("offset-increment", "n_clicks"),
        State("offset-display", "children"),
        prevent_initial_call=True
    )
    def update_offset(decrement_clicks, increment_clicks, current_offset):
        step = 1
        min_value, max_value = 1, 10
        offset = 5

        offset += (increment_clicks - decrement_clicks) * step
        offset = max(min_value, min(max_value, offset))

        return str(offset)


# ─────────────────────────────
# ❌ Annotation Suppression Popup
# ─────────────────────────────

def register_popup_annotation_suppression():
    @callback(
        Output("delete-confirmation-modal", "is_open"),
        Output("delete-modal-body", "children"),
        Input("delete-annotations-btn", "n_clicks"),
        State("annotation-checkboxes", "value"),
        State("delete-confirmation-modal", "is_open"),
        prevent_initial_call=True
    )
    def _open_delete_modal(delete_click, selected_annotations, is_open):
        if delete_click and selected_annotations:
            selected_text = ", ".join(selected_annotations)
            return True, f"Are you sure you want to delete: {selected_text}?"
        return is_open, dash.no_update


def register_cancel_or_confirm_annotation_suppression():
    @callback(
        Output("delete-confirmation-modal", "is_open", allow_duplicate=True),
        Output("annotations-store", "data", allow_duplicate=True),
        Input("confirm-delete-btn", "n_clicks"),
        Input("cancel-delete-btn", "n_clicks"),
        State("annotation-checkboxes", "value"),
        State("annotations-store", "data"),
        State("delete-confirmation-modal", "is_open"),
        prevent_initial_call=True
    )
    def handle_delete_confirmation(confirm_click, cancel_click, selected_annotations, annotations_dict, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open, dash.no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "confirm-delete-btn":
            annotations_dict = [a for a in annotations_dict if a['description'] not in selected_annotations]
            return False, annotations_dict
        if trigger_id == "cancel-delete-btn":
            return False, dash.no_update
        return is_open, dash.no_update


# ─────────────────────────────
# 🏷 Annotation Name Callbacks
# ─────────────────────────────

def register_annotation_checkboxes_options(checkboxes_id):
    @callback(
        Output(checkboxes_id, "options"),
        Input("annotations-store", "data"),
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
        Input(checkboxes_id, "value"),
        prevent_initial_call=False
    )
    def _update_annotation_dropdown(annotations_value):
        """Depending of which annotations is checked, update the dropdown that move to previous/next event."""
        if not annotations_value:
            return dash.no_update

        return [{"label": "All Selected", "value": "__ALL__"}] + \
               [{"label": name, "value": name} for name in annotations_value]


# ─────────────────────────────
# 🎛 Montage Management
# ─────────────────────────────

def register_callbacks_montage_names(radio_id):
    @callback(
        Output(radio_id, "options"),
        Output(radio_id, "value"),
        Input("montage-store", "data"),
        State(radio_id, "value"),
        prevent_initial_call=False
    )
    def display_montage_names(montage_store, current_value):
        if montage_store is None:
            raise PreventUpdate

        options = [{'label': k, 'value': k} for k in montage_store.keys()] + \
                  [{'label': 'channel selection', 'value': 'channel selection'}]

        if current_value not in [o['value'] for o in options]:
            return options, options[0]['value']

        return options, current_value


def register_hide_channel_selection_when_montage():
    @callback(
        Output("channel-region-checkboxes", "options"),
        Input("montage-radio", "options"),
        Input("montage-radio", "value"),
        State("channel-region-checkboxes", "options"),
        prevent_initial_call=False
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
        Input("anomaly-detection-store", "data"),
        Input("colors-radio", "value"),
        State("colors-radio", "options"),
        prevent_initial_call=False
    )
    def update_sensitivity_options(sa_store, ad_store, selected, current_options):
        if not sa_store and not ad_store:
            return dash.no_update

        sa_options = [{'label': k, 'value': k} for k in sa_store or {}]
        ad_options = [{'label': k, 'value': k} for k in ad_store or {}]

        new_options = [opt for opt in sa_options + ad_options if opt not in current_options]

        return current_options + new_options