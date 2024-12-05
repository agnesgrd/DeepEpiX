# run.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize Dash app with use_pages=True
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Main layout with page container for dynamic content loading
app.layout = html.Div(
    children=[
        dcc.Store(id="folder-store", storage_type="local"),
        dcc.Store(id="frequency-store", storage_type="local"),
        dcc.Store(id="preprocessed-data-store", storage_type="local"),

        # This will track the URL and switch between pages based on tab selection
        dcc.Location(id='url', refresh=False),

        # Main content
        html.Div(
            children=[
                html.H1("DeepEpiX"),
                dash.page_container,  # This will hold the content of each page
            ],
            style={"width": "80%", "display": "inline-block", "padding": "20px"},
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
                "padding": "20px",
                "border-left": "1px solid #ccc",
            },
        ),
    ],
    style={"display": "flex"},
)


if __name__ == "__main__":
    app.run_server(debug=True)
