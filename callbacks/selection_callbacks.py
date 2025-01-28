import dash
from dash import html, dcc, Input, Output, State

def register_page_buttons_display():
    # Callback to generate page buttons based on the chunk limits in the store
    @dash.callback(
        Output("page-buttons-container", "children"),
        Input("chunk-limits-store", "data"),
        prevent_initial_call=False
    )
    def update_buttons(chunk_limits):
        return html.Div(
            # RadioItems for the page buttons
            dcc.RadioItems(
                id="page-selector",
                options=[
                    {"label": html.Span(
                    str(i + 1),
                    style={
                        "textDecoration": "underline" if i == 0 else "none",  # Underline selected number
                    })
                    , "value": i} for i in range(len(chunk_limits))
                ],
                value=0,  # Default selected page
                className="btn-group",  # Group styling
                inputClassName="btn-check",  # Bootstrap class for hidden radio inputs
                labelClassName="btn btn-outline-primary",  # Default button style
                inputStyle={"display": "none"},  # Hide the input completely
            ),
        )

def register_update_page_button_styles():
    # Callback to handle button click and update styles for all buttons
    @dash.callback(
        Output("page-selector", "options"),
        Input("page-selector", "value"),
        State("chunk-limits-store", "data"),
        prevent_initial_call=True
    )
    def update_button_styles(selected_value, chunk_limits):
        # Update styles dynamically for each button
        if chunk_limits is None or selected_value is None:
            return dash.no_update  # Default to the first page
        return [
            {
                "label": html.Span(
                    str(i + 1),
                    style={
                        "textDecoration": "underline" if i == selected_value else "none",  # Underline selected number
                    },
                ),
                "value": i,
            }
            for i in range(len(chunk_limits))
        ]