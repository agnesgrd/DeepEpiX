# run.py
from dash import Dash, html, dcc, page_container
import dash_bootstrap_components as dbc
import uuid
from flask_caching import Cache

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
        dcc.Store(id="folder-store", storage_type="local"),
        dcc.Store(id="raw-store", storage_type="local"),
        dcc.Store(id="plotting-data-store", data={}, storage_type="local"),
        dcc.Store(id="main-graph-resampler", storage_type="local"),
        dcc.Store(id="frequency-store", storage_type="local"),
        dcc.Store(id="annotations-store", storage_type="local"),

        # This will track the URL and switch between pages based on tab selection
        dcc.Location(id='url', refresh=False),

        # Main content
        html.Div(
            children=[
                html.H1("DeepEpiX"),
                page_container,  # This will hold the content of each page
            ],
            style={"width": "90%", "display": "inline-block", "padding": "10px"},
        ),

        # Navigation (replacing dcc.Tabs with dcc.Link for URL routing)
        html.Div(
            children=[
                html.H3("Navigation"),
                html.Div(
                    children=[
                        dcc.Link('Home', href='/', style={'display': 'block', 'margin-bottom': '10px'}),
                        dcc.Link('View', href='/view', style={'display': 'block', 'margin-bottom': '10px'}),
                        dcc.Link('Analyze', href='/analyze', style={'display': 'block', 'margin-bottom': '10px'}),
                        dcc.Link('Predict', href='/predict', style={'display': 'block', 'margin-bottom': '10px'}),
                        dcc.Link('Save', href='/save', style={'display': 'block', 'margin-bottom': '10px'}),
                    ]
                ),
            ],
            style={
                "width": "10%",
                "display": "inline-block",
                "vertical-align": "top",
                "padding": "10px",
                "border-left": "1px solid #ccc",
            },
        ),
    ],
    style={"display": "flex"},
)


if __name__ == "__main__":
    app.run_server(debug=True)
