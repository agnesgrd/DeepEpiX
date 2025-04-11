# run.py
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import os
import dash

app = Dash(__name__,
           use_pages=True,
        #    suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP])

# app.enable_dev_tools(debug=True)

# Main layout with page container for dynamic content loading
app.layout = html.Div(
    children=[

        # This will track the URL and switch between pages based on tab selection
        dcc.Location(id='url', refresh=False),

        dcc.Store(id="folder-store", storage_type="session"),
        dcc.Store(id="chunk-limits-store", data=[], storage_type="session"),
        dcc.Store(id="frequency-store", storage_type="session"),
        dcc.Store(id="annotations-store", data = [], storage_type="session"),
        dcc.Store(id="montage-store", data={}, storage_type="session"),
        dcc.Store(id="history-store", storage_type="session"),
        dcc.Store(id='sensitivity-analysis-store', data={}, storage_type='session'),
        dcc.Store(id='anomaly-detection-store', data={}, storage_type='session'),

        # Row for title and links
        html.Div(
            children=[
                # Panel with the logo and clickable tabs
                html.Div(
                    children=[
                        # Logo
                        html.Img(
                            src="/assets/deepepix-logo.jpeg",
                            style={
                                "border-radius": "10%",  # Bordure arrondie
                                "padding-top": "0px",
                                "height": "60px",  # Adjust size as needed
                            }
                        ),
                        # Panel tabs (navigation links)
                        html.I(dbc.DropdownMenu(
                            toggle_style={
                                "border": "none",
                                "background": "none",
                                "font-size": "30px",
                                "color": "white",
                            },
                            children=[
                                dbc.DropdownMenuItem("Dataset", header=True),
                                dbc.DropdownMenuItem("Home", href="/"),

                                dbc.DropdownMenuItem(divider=True),
                                
                                dbc.DropdownMenuItem("Visualization", header=True),
                                dbc.DropdownMenuItem("Raw Signal", href="/viz/raw-signal"),
                                dbc.DropdownMenuItem("ICA", href="/viz/ica"),

                                dbc.DropdownMenuItem(divider=True),

                                dbc.DropdownMenuItem("Model", header=True),
                                dbc.DropdownMenuItem("Performance", href="/model/performance"),
                                dbc.DropdownMenuItem("Fine-Tuning", href="/model/fine-tuning"),

                                dbc.DropdownMenuItem(divider=True),

                                dbc.DropdownMenuItem("Settings", header=True),
                                dbc.DropdownMenuItem("Montage", href="/settings/montage"),
                                dbc.DropdownMenuItem("Help", href="/settings/help"),
                            ],
                            style={"display": "flex"},
                            className="bi bi-list"
))
                    ],
                    style={
                        "display": "flex",
                        "font-size": "60px",  # Make the icon bigger
                        "padding-bottom": "20px"
                    },
                ),
                # Main content container (display content based on the tab selected)
                html.Div(
                    children=[
                        dash.page_container,  # Placeholder for dynamically loaded content
                    ],
                    style={"width": "100%", "display": "inline-block"}  # Main content area
                ),
            ],
            style={"width": "100%", "padding": "20px"},  # Row layout
        ),
    ],
    style={"display": "flex", "flex-direction": "column", "height": "100vh"},  # Full-page flex layout
)

server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8080)
