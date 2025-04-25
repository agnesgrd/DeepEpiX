from dash import html
import dash_bootstrap_components as dbc

from layout.selection_sidebar_layout import create_selection
from layout.analyze_sidebar_layout import create_analyze
from layout.predict_sidebar_layout import create_predict
from layout.save_sidebar_layout import create_save

def create_sidebar():
    return html.Div([
        dbc.Tabs([
                dbc.Tab(create_selection(
                    montage_radio_id="montage-radio", 
                    check_all_button_id="check-all-channels-btn", 
                    clear_all_button_id="clear-all-channels-btn",
                    channel_region_checkboxes_id="channel-region-checkboxes", 
                    check_all_annotations_btn_id="check-all-annotations-btn", 
                    clear_all_annotations_btn_id="clear-all-annotations-btn", 
                    delete_annotations_btn_id="delete-annotations-btn",
                    annotation_checkboxes_id="annotation-checkboxes",
                    delete_confirmation_modal_id="delete-confirmation-modal",
                    cancel_delete_btn_id="cancel-delete-btn",
                    confirm_delete_btn_id="confirm-delete-btn",
                    offset_decrement_id="offset-decrement", 
                    offset_display_id="offset-display", 
                    offset_increment_id="offset-increment", 
                    colors_radio_id="colors-radio"
                ), label='Select', tab_id='selection-tab'),
                dbc.Tab(create_analyze(), label='Analyze', tab_id='analyzing-tab'),
                dbc.Tab(create_predict(), label='SpikePred', tab_id='prediction-tab'),
                dbc.Tab(create_save(), label='Save', tab_id='saving-tab'),
            ],
            id="sidebar-tabs",
            persistence = True,
            persistence_type = "local"
        ),

    ], style={
        "padding": "0 20px",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "flex-start",  # Align content at the top
        "gap": "20px",  # Space between elements
        "width": "275px",  # Sidebar width is now fixed
        "boxSizing": "border-box",
        "fontSize": "12px",
        "borderRadius": "10px",  # Rounded corners for the sidebar itself
        "overflowX": "auto"
    })