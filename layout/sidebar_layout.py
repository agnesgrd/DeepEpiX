from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles
from layout.selection_sidebar_layout import create_selection
from layout.analyze_sidebar_layout import create_analyze
from layout.predict_sidebar_layout import create_predict
from layout.save_sidebar_layout import create_save
from layout.anom_detect_sidebar_layout import create_anom_detect



# Helper function to create the sidebar with checkboxes
def create_sidebar():
    return html.Div([
        dbc.Tabs(
                [
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
                ), label='Select', tab_id='selection-tab'), # create_selection()
                dbc.Tab(create_analyze(), label='Analyze', tab_id='analyzing-tab'), # create_analyze()
                dbc.Tab(create_predict(), label='SpikePred', tab_id='prediction-tab'), #create_prediction()
                dbc.Tab(create_anom_detect(), label='AnomDetect', tab_id='anom-detection-tab'), #create_prediction()
                dbc.Tab(create_save(), label='Save', tab_id='saving-tab'), #create_prediction()
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
        # "backgroundColor": "#f9f9f9",  # Light background color for the sidebar
        "borderRadius": "10px",  # Rounded corners for the sidebar itself
        # "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",  # Subtle shadow for the whole sidebar
        "overflowY": "auto",  # Enable scrolling if content exceeds height
    })