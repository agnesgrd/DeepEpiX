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
                dbc.Tab(create_selection(), label='Select', tab_id='selection-tab'), # create_selection()
                dbc.Tab(create_analyze(), label='Analyze', tab_id='analyzing-tab'), # create_analyze()
                dbc.Tab(create_predict(), label='Predict', tab_id='prediction-tab'), #create_prediction()
                dbc.Tab(create_anom_detect(), label='Anomaly', tab_id='anom-detection-tab'), #create_prediction()
                dbc.Tab(create_save(), label='Save', tab_id='saving-tab'), #create_prediction()
            ],
            id="sidebar-tabs",
            persistence = True,
            persistence_type = "local"
        ),

    ], style={
        # "padding": "20px",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "flex-start",  # Align content at the top
        "gap": "20px",  # Space between elements
        "width": "250px",  # Sidebar width is now fixed
        "boxSizing": "border-box",
        "fontSize": "12px",
        # "backgroundColor": "#f9f9f9",  # Light background color for the sidebar
        "borderRadius": "10px",  # Rounded corners for the sidebar itself
        # "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",  # Subtle shadow for the whole sidebar
        "overflowY": "auto",  # Enable scrolling if content exceeds height
    })