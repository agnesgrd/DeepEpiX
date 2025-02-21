import dash
from dash import html, dcc, Input, Output, State
import static.constants as c
import callbacks.utils.annotation_utils as au

def register_page_buttons_display():
    # Callback to generate page buttons based on the chunk limits in the store
    @dash.callback(
        Output("page-buttons-container", "children"),
        Input("chunk-limits-store", "data"),
        prevent_initial_call=False
    )
    def update_buttons(chunk_limits):
        return html.Div(
            # RadioItems for the page buttons
            dcc.RadioItems(
                id="page-selector",
                options=[
                    {"label": html.Span(
                    str(i + 1),
                    style={
                        "textDecoration": "underline" if i == 0 else "none",  # Underline selected number
                    })
                    , "value": i} for i in range(len(chunk_limits))
                ],
                value=0,  # Default selected page
                className="btn-group",  # Group styling
                inputClassName="btn-check",  # Bootstrap class for hidden radio inputs
                labelClassName="btn btn-outline-primary",  # Default button style
                inputStyle={"display": "none"},  # Hide the input completely
            ),
        )

def register_update_page_button_styles():
    # Callback to handle button click and update styles for all buttons
    @dash.callback(
        Output("page-selector", "options"),
        Input("page-selector", "value"),
        State("chunk-limits-store", "data"),
        prevent_initial_call=True
    )
    def update_button_styles(selected_value, chunk_limits):
        # Update styles dynamically for each button
        if chunk_limits is None or selected_value is None:
            return dash.no_update  # Default to the first page
        return [
            {
                "label": html.Span(
                    str(i + 1),
                    style={
                        "textDecoration": "underline" if i == selected_value else "none",  # Underline selected number
                    },
                ),
                "value": i,
            }
            for i in range(len(chunk_limits))
        ]
    
def register_manage_channels_checklist():
    @dash.callback(
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
    
def register_manage_annotations_checklist():
    @dash.callback(
        Output("annotation-checkboxes", "value", allow_duplicate=True),
        [Input("check-all-annotations-btn", "n_clicks"),
        Input("clear-all-annotations-btn", "n_clicks")],
        State("annotation-checkboxes", "options"),
        prevent_initial_call = True
    )
    def manage_checklist(check_all_clicks, clear_all_clicks, options):
        # Determine which button was clicked
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "check-all-annotations-btn":
            all_options = [option['value'] for option in options]
            return all_options  # Select all regions
        elif triggered_id == "clear-all-annotations-btn":
            return []  # Clear all selections

        return dash.no_update
    
# def register_check_new_annotation_by_default():
#     @dash.callback(
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
    @dash.callback(
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
    @dash.callback(
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
    @dash.callback(
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
    
def register_callbacks_annotation_names():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("annotation-checkboxes", "options"),
        Output("annotation-checkboxes", "value"),
        Input("annotations-store", "data"),
        prevent_initial_call = False
    )
    def display_annotation_names_checklist(annotations_store):
        description_counts = au.get_annotation_descriptions(annotations_store)

        options = [{'label': f"{name} ({count})", 'value': f"{name}"} for name, count in description_counts.items()]
        value = [f"{name}" for name in description_counts.keys()]
        return options, dash.no_update  # Set all annotations as default selected
    
def register_callbacks_montage_names():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("montage-radio", "options"),
        Output("montage-radio", "value"),
        Input("montage-store", "data"),
        State("montage-radio", "value"),
        prevent_initial_call=False
    )
    def display_annotation_names_checklist(montage_store, value):
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
    @dash.callback(
        Output("channel-region-checkboxes", "options"),
        Input("montage-radio", "options"),
        Input("montage-radio", "value"),
        State("channel-region-checkboxes", "options"),
        prevent_initial_call=False
    )
    def display_annotation_names_checklist(montage_option, montage_value, channel_options):
        if montage_value != 'channel selection':
            # Disable all options
            return [{'label': option['label'], 'value': option['value'], 'disabled': True} for option in channel_options]
        else:
            # Enable all options
            return [{'label': option['label'], 'value': option['value'], 'disabled': False} for option in channel_options]

def register_callbacks_sensivity_analysis():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("colors-radio", "options"),
        Input("sensitivity-analysis-store", "data"),
        Input("colors-radio", "value"),
        State("colors-radio", "options"),
        prevent_initial_call=False
    )
    def display_sensitivity_analysis_checklist(sa_store, value, default_options):
        # Create options for the checklist from the channels in montage_store
        options = [{'label': key, 'value': key} for key in sa_store.keys()]
        if options[-1] not in default_options:
            updated_options = default_options + options
            # If value is valid, keep the current selection
            return updated_options
        else:
            return dash.no_update
    