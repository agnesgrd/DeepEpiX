# Dash & Plotly
import dash
from dash import html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Local Imports
from callbacks.utils import folder_path_utils as fpu
from callbacks.utils import annotation_utils as au
from callbacks.utils import history_utils as hu
from layout import icon

def register_populate_memory_tab_contents():
    @callback(
        Output("subject-container-memory", "children"),
        Output("raw-info-container-memory", "children"),
        Output("event-stats-container-memory", "children"),
        Output("history-container-memory", "children"),
        Input("url", "pathname"),
        Input("subject-tabs-memory", "active_tab"),
        State("folder-store", "data"),
        State("chunk-limits-store", "data"),
        State("frequency-store", "data"),
        State("annotations-store", "data"),
        State("history-store", "data"),
        prevent_initial_call=False
    )
    def populate_memory_tab_contents(pathname, selected_tab, folder_path, chunk_limits, freq_data, annotations_data, history_data):
        """Populate memory tab content based on selected tab and stored folder path."""
        if not folder_path or not chunk_limits or not freq_data:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        subject_content = dash.no_update
        raw_info_content = dash.no_update
        event_stats_content = dash.no_update
        history_content = dash.no_update

        if selected_tab == "subject-tab-memory":
                subject_content = dbc.Card(
                    dbc.CardBody([
                        html.H5([html.I(className="bi bi-person-rolodex", style={"marginRight": "10px", "fontSize": "1.2em"}), "Subject"], className="card-title"),
                        html.Hr(),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Strong(folder_path)
                            ]),
                        ]),

                        html.H5([html.I(className="bi bi-sliders", style={"marginRight": "10px", "fontSize": "1.2em"}), "Frequency Parameters"], className="card-title"),
                        html.Hr(),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Strong("Resample Frequency: "),
                                html.Span(f"{freq_data.get('resample_freq', 'N/A')} Hz")
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Low-pass Filter: "),
                                html.Span(f"{freq_data.get('low_pass_freq', 'N/A')} Hz")
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("High-pass Filter: "),
                                html.Span(f"{freq_data.get('high_pass_freq', 'N/A')} Hz")
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Notch Filter: "),
                                html.Span(f"{freq_data.get('notch_freq', 'N/A')} Hz")
                            ]),
                        ])
                    ])
                )

        if selected_tab == "raw-info-tab-memory":
            raw_info_content = fpu.build_table_raw_info(folder_path)

        if selected_tab == "events-tab-memory":
            event_stats_content = au.build_table_events_statistics(annotations_data)

        if selected_tab == "history-tab-memory":

            history_content = dbc.Card(
                    dbc.CardBody([
                        html.Div([
                            html.H5([
                                html.I(className=f"bi {icon[category]}", style={"marginRight": "10px", "fontSize": "1.2em"}),
                                category.capitalize()
                            ], className="card-title"),
                            html.Hr(),
                            dbc.ListGroup([
                                dbc.ListGroupItem(entry)
                                for entry in hu.read_history_data_by_category(history_data, category)
                            ]) if hu.read_history_data_by_category(history_data, category) else
                            html.P("No entries yet.", className="text-muted")
                        ], style={"marginBottom": "10px"})

                        for category in ['annotations', 'models', 'ICA']
                        ]
                    )
                )

        return subject_content, raw_info_content, event_stats_content, history_content