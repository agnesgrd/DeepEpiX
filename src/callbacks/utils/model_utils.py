import pandas as pd
import dash_bootstrap_components as dbc
from dash import html
from config import STATIC_DIR


def render_pretrained_models_table():
    df = pd.read_csv(STATIC_DIR / "pretrained-models-info.csv")

    table_header = html.Thead(html.Tr([html.Th(col) for col in df.columns]))

    table_body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(
                        str(row[col]),
                        style={
                            # "whiteSpace": "nowrap",
                            # "overflowX": "auto",
                            "maxWidth": "200px",
                            "padding": "10px",
                        },
                    )
                    for col in df.columns
                ]
            )
            for _, row in df.iterrows()
        ]
    )

    return dbc.Table(
        children=[table_header, table_body],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
        size="sm",
        class_name="table-light text-center",
        style={
            "width": "70%",
            "margin": "0 auto",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "14px",
        },
    )
