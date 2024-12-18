from dash import html, dcc
import static.constants as c

def get_graph_layout():

    # # Generate vertical line shapes for annotations
    # annotation_lines = [
    #     {
    #         'type': 'line',
    #         'x0': onset,
    #         'x1': onset,
    #         'y0': 0,
    #         'y1': 1,
    #         'xref': 'x',
    #         'yref': 'paper',  # 'paper' makes the y-range relative [0, 1]
    #         'line': {
    #             'color': 'red' if desc == 'spike' else 'blue',
    #             'width': 2,
    #             'dash': 'dot'
    #         }
    #     }
    #     for onset, desc in zip(annotations_df['onset'], annotations_df['description'])
    # ]

    layout = html.Div([  # Main container for the layout
        # Sidebar with Channel Slider
        html.Div([
            html.Label("Select Channels:"),
            # Checklist for selecting regions
            dcc.Checklist(
            id="channel-region-checkboxes",
            options=[
                {
                    'label': f"{region_code} ({len(channels)})", 
                    'value': region_code
                }
                for region_code, channels in c.GROUP_CHANNELS_BY_REGION.items()
            ],
            value=["RC", "LC", "ZC"],  # Default selected regions (Right, Left, Z regions)
            inline=False,
            style={"margin": "10px 0"},
            persistence=True,
            persistence_type="local"
        )
        ], style={
            "padding": "10px",
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center"
        }),

        # Graph and Time Slider Container
        html.Div([
            # Graph
            html.Div([
                dcc.Graph(id="meg-signal-graph")
            ], style={"flexGrow": 1, "padding": "10px"}),

            # Time Selector (below the graph)
            html.Div([
                html.Label("Select Time Range (s):"),
                dcc.RangeSlider(
                    id="time-slider",
                    min=0,
                    max=180,  # Default max, update dynamically
                    step=1,
                    marks={i: str(i) for i in range(0, 181, 10)},  # Add visible marks
                    value=[0, 10],  # Default range
                    tooltip={"placement": "bottom", "always_visible": True},
                    persistence=True,
                    persistence_type="local"
                ),
                html.Div([
                    html.Button("←", id="time-left", n_clicks=0, style={"font-size": "16px", "padding": "5px 10px"}),
                    html.Button("→", id="time-right", n_clicks=0, style={"font-size": "16px", "padding": "5px 10px", "margin-left": "10px"})
                ], style={"display": "flex", "flex-direction": "row", "align-items": "center", "margin-top": "10px"})
            ], style={
                "margin-top": "20px",
                "width": "100%",
                "textAlign": "center"
            })
        ], style={
            "flexGrow": 1,
            "display": "flex",
            "flexDirection": "column",  # Stack graph and time slider vertically
            "padding": "10px"
        })
    ], style={
        "display": "flex",  # Main container layout
        "height": "500px",
        "gap": "10px"
    })

    return layout