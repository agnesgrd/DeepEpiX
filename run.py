# run.py
from dash import Dash, html, dcc, page_container
import dash_bootstrap_components as dbc
import uuid
from flask_caching import Cache
from dash import Input, Output, State
import dash
from layout import box_styles

# Initialize Dash app with use_pages=True
app = Dash(__name__,
           use_pages=True, 
           external_stylesheets=[dbc.themes.BOOTSTRAP])

# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory',
#     'CACHE_THRESHOLD': 10 # should be equal to maximum number of users on the app at a single time
# })


# Main layout with page container for dynamic content loading
app.layout = html.Div(
    children=[

        # This will track the URL and switch between pages based on tab selection
        dcc.Location(id='url', refresh=False),

        dcc.Store(id="folder-store", storage_type="local"),
        dcc.Store(id="raw-store", storage_type="local"),
        dcc.Store(id="plotting-data-store", data={}, storage_type="local"),
        dcc.Store(id="main-graph-resampler", storage_type="local"),
        dcc.Store(id="frequency-store", storage_type="local"),
        dcc.Store(id="annotations-store", storage_type="local"),
        dcc.Store(id="first-load-store", data=0, storage_type="local"),


        # Row for title and links
        html.Div(
            children=[
                # Title of the app
                html.H1("DeepEpiX", style={"text-align": "left", "flex": 1, "padding-top": "10px"}),

                # Panel with clickable tabs that navigate to different links
                html.Div(
                    children=[
                        dcc.Link(
                            "Home", id="link-home", href="/", style={**box_styles["panel-tabs"]} 
                        ),
                        dcc.Link(
                            "View", id="link-view", href="/view", style={**box_styles["panel-tabs"]} 
                        ),
                        dcc.Link(
                            "Analyze", id="link-analyze", href="/analyze", style={**box_styles["panel-tabs"]} 
                        ),
                        dcc.Link(
                            "Predict", id="link-predict", href="/predict", style={**box_styles["panel-tabs"]} 
                        ),
                        dcc.Link(
                            "Save", id="link-save", href="/save", style={**box_styles["panel-tabs"]} 
                        ),
                    ],
                    style={
                        "width": "100vw"
                    },
                ),
                # Main content container (display content based on the tab selected)
                html.Div(
                    children=[  
                        page_container,  # Placeholder for dynamically loaded content
                    ],
                    style={"width": "100%", "display": "inline-block", "padding": "20px"},  # Main content area
                ),
            ],
            style={"width": "100%", "padding": "10px"},  # Row layout
        ),
    ],
    style={"display": "flex", "flex-direction": "column", "height": "100vh"},  # Full-page flex layout
)

# Callback to update the style of the active tab
@dash.callback(
    [Output("link-home", "style"),
     Output("link-view", "style"),
     Output("link-analyze", "style"),
     Output("link-predict", "style"),
     Output("link-save", "style")],
    [Input("url", "pathname")]
)
def update_tab_style(pathname):

    # Highlight the active link by changing its background color
    active_style = {**box_styles["panel-tabs"], "background-color": "blue"}  # Dark Gray for active link

    # Return updated styles based on the current pathname
    return (
        active_style if pathname == "/" else {**box_styles["panel-tabs"]},
        active_style if pathname == "/view" else {**box_styles["panel-tabs"]},
        active_style if pathname == "/analyze" else {**box_styles["panel-tabs"]} ,
        active_style if pathname == "/predict" else {**box_styles["panel-tabs"]} ,
        active_style if pathname == "/save" else {**box_styles["panel-tabs"]},
    )


if __name__ == "__main__":
    app.run_server(debug=True)
