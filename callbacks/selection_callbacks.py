import dash
from dash import html, dcc, Input, Output, State, callback
import static.constants as c
import callbacks.utils.annotation_utils as au
from dash.exceptions import PreventUpdate

def register_page_buttons_display(chunk_limits_store_id, page_buttons_container_id, page_selector_id):
    @callback(
        Output(page_buttons_container_id, "children"),
        Input(chunk_limits_store_id, "data"),
        prevent_initial_call=False
    )
    def update_buttons(chunk_limits):
        if not chunk_limits:
            return dash.no_update
        
        return html.Div(
            dcc.RadioItems(
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
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                inputStyle={"display": "none"},
            )
        )

def register_update_page_button_styles(page_selector_id, chunk_limits_store_id):
    @callback(
        Output(page_selector_id, "options"),
        Input(page_selector_id, "value"),
        State(chunk_limits_store_id, "data"),
        prevent_initial_call=True
    )
    def update_button_styles(selected_value, chunk_limits):
        if not chunk_limits:
            return dash.no_update
        
        return [
            {
                "label": html.Span(
                    str(i + 1),
                    style={"textDecoration": "underline" if i == selected_value else "none"}
                ),
                "value": i
            } for i in range(len(chunk_limits))
        ]
    
def register_manage_channels_checklist():
    @callback(
        Output("channel-region-checkboxes", "value"),
        [Input("check-all-channels-btn", "n_clicks"),
        Input("clear-all-channels-btn", "n_clicks")],
        prevent_initial_call = True
    )
    def manage_checklist(check_all_clicks, clear_all_clicks):
        # Determine which button was clicked
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "check-all-channels-btn":
            all_regions = list(c.GROUP_CHANNELS_BY_REGION.keys())
            return all_regions  # Select all regions
        elif triggered_id == "clear-all-channels-btn":
            return []  # Clear all selections

        return dash.no_update
    
def register_manage_annotations_checklist(
    check_all_annotations_btn_id,
    clear_all_annotations_btn_id,
    annotation_checkboxes_id
):
    @callback(
        Output(annotation_checkboxes_id, "value", allow_duplicate=True),
        [
            Input(check_all_annotations_btn_id, "n_clicks"),
            Input(clear_all_annotations_btn_id, "n_clicks")
        ],
        State(annotation_checkboxes_id, "options"),
        prevent_initial_call=True
    )
    def manage_checklist(check_all_clicks, clear_all_clicks, options):
        # Determine which button was clicked
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == check_all_annotations_btn_id:
            all_options = [option['value'] for option in options]
            return all_options  # Select all regions
        elif triggered_id == clear_all_annotations_btn_id:
            return []  # Clear all selections

        return dash.no_update
    

    
# def register_check_new_annotation_by_default():
#     @callback(
#         Output("annotation-checkboxes", "value", allow_duplicate=True),
#         Input("annotation-store", "data"),
#         Input("annotation-checkboxes", "options"),
#         prevent_initial_call = True
#     )
#     def check_new_annotation_by_default(value, options):
#         # Determine which button was clicked
#         new_option = options[-1]['value']
#         if new_option not in value:
#             return value.append(new_option)  # Select all regions
#         return dash.no_update
    
def register_offset_display():
    @callback(
        Output("offset-display", "children", allow_duplicate=True),  # Update displayed offset value
        Input("offset-decrement", "n_clicks"),  # `-` button clicks
        Input("offset-increment", "n_clicks"),  # `+` button clicks
        State("offset-display", "children"),    # Current offset value
        prevent_initial_call=True
    )
    def update_offset(decrement_clicks, increment_clicks, current_offset):
        # Step value and range constraints
        step = 1
        min_value = 1
        max_value = 10

        # Convert current offset to integer
        offset = 5

        # Calculate new offset based on button clicks
        offset += (increment_clicks - decrement_clicks) * step

        # Enforce boundaries
        offset = max(min_value, min(max_value, offset))

        return str(offset)  # Return updated offset as a string
    
def register_popup_annotation_suppression():
    @callback(
        Output("delete-confirmation-modal", "is_open"),
        Output("delete-modal-body", "children"),
        Input("delete-annotations-btn", "n_clicks"),
        State("annotation-checkboxes", "value"),
        State("delete-confirmation-modal", "is_open"),
        prevent_initial_call=True
    )
    def open_delete_modal(delete_click, selected_annotations, is_open):
        if delete_click and selected_annotations:
            selected_text = ", ".join(selected_annotations)
            return True, f"Are you sure you want to delete: {selected_text}?"
        return is_open, dash.no_update

def register_cancel_or_confirm_annotation_suppression():
    @callback(
        Output("delete-confirmation-modal", "is_open", allow_duplicate=True),
        Output("annotations-store", "data", allow_duplicate=True),  # Update checklist after deletion
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
            # Remove items with the selected description
            annotations_dict = [annotation for annotation in annotations_dict if annotation['description'] not in selected_annotations]
            return False, annotations_dict  # Close modal and clear selected annotations

        if trigger_id == "cancel-delete-btn":
            return False, dash.no_update  # Close modal without deleting

        return is_open, dash.no_update
    
def register_callbacks_annotation_names(
    annotation_checkboxes_id
    
):
    # Callback to populate the checklist options and default value dynamically
    @callback(
        Output(annotation_checkboxes_id, "options"),
        Input("annotations-store", "data"),
        prevent_initial_call=False
    )
    def display_annotation_names_checklist(annotations_store):
        if not annotations_store:
            return dash.no_update
                
        description_counts = au.get_annotation_descriptions(annotations_store)

        options = [{'label': f"{name} ({count})", 'value': f"{name}"} for name, count in description_counts.items()]
        # value = [f"{name}" for name in (current_value or []) if name in description_counts.keys()]
        return options  # Set all annotations as default selected
    
def register_callbacks_annotation_names_dropdown(
    annotation_dropdown_id,
    annotation_checkboxes_id
    
):
    # Callback to populate the checklist options and default value dynamically
    @callback(
        Output(annotation_dropdown_id, "options"),
        Input(annotation_checkboxes_id, "value"),
        prevent_initial_call=False
    )
    def display_annotation_names_checklist(annotations_value):
        if not annotations_value:
            return dash.no_update

        options = [{"label": "All Selected", "value": "__ALL__"}]+[{'label': f"{name}", 'value': f"{name}"} for name in annotations_value]

        return options
    
def register_callbacks_montage_names(montage_radio_id):
    # Callback to populate the checklist options and default value dynamically
    @callback(
        Output(montage_radio_id, "options"),
        Output(montage_radio_id, "value"),
        Input("montage-store", "data"),
        State(montage_radio_id, "value"),
        prevent_initial_call=False
    )
    def display_montage_names_checklist(montage_store, value):

        if montage_store is None:
            raise PreventUpdate
        
        # Create options for the checklist from the channels in montage_store
        options = [{'label': key, 'value': key} for key in montage_store.keys()]
        options.append({'label': 'channel selection', 'value': 'channel selection'})

        # If montage_store is empty or the current value is not valid, select the first option
        valid_values = [option['value'] for option in options]
        if not montage_store or value not in valid_values:
            # Return the first option as default
            return options, options[0]['value']

        # If value is valid, keep the current selection
        return options, value
    
def register_hide_channel_selection_when_montage():
    # Callback to populate the checklist options and default value dynamically
    @callback(
        Output("channel-region-checkboxes", "options"),
        Input("montage-radio", "options"),
        Input("montage-radio", "value"),
        State("channel-region-checkboxes", "options"),
        prevent_initial_call=False
    )
    def hide_channel_selection_when_montage(montage_option, montage_value, channel_options):
        if montage_value != 'channel selection':
            # Disable all options
            return [{'label': option['label'], 'value': option['value'], 'disabled': True} for option in channel_options]
        else:
            # Enable all options
            return [{'label': option['label'], 'value': option['value'], 'disabled': False} for option in channel_options]

def register_callbacks_sensivity_analysis():
    # Callback to populate the checklist options and default value dynamically
    @callback(
        Output("colors-radio", "options"),
        Input("sensitivity-analysis-store", "data"),
        Input("anomaly-detection-store", "data"),
        Input("colors-radio", "value"),
        State("colors-radio", "options"),
        prevent_initial_call=False
    )
    def display_sensitivity_analysis_checklist(sa_store, ad_store, value, default_options):
        # Create options for the checklist from the channels in montage_store
        if (sa_store is None or sa_store == {}) and (ad_store is None or ad_store == {}):
            return dash.no_update
        new_options = [{'label': key, 'value': key} for key in sa_store.keys() if {'label': key, 'value': key} not in default_options] + [{'label': key, 'value': key} for key in ad_store.keys() if {'label': key, 'value': key} not in default_options]

        updated_options = default_options + new_options

        return updated_options

    