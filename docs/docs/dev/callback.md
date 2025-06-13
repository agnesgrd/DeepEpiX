# 🧑‍🏭 Writing a Dash Callback
Dash callbacks define how interactive elements (like dropdowns, sliders, or buttons) update other components in your app.

---

## 🧩 Basic Structure

```python
from dash import Input, Output, State, callback

@callback(
    Output("output-component-id", "property"),
    Input("input-component-id", "property"),
)
def update_output(input_value):
    # Your logic here
    return updated_value
```

---

## 🔁 With Multiple Inputs or Outputs and a State

```python
@callback(
    [Output("graph", "figure"), Output("text-box", "children")],
    [Input("dropdown", "value"), Input("slider", "value")],
    State("parameter", "data"),
)
def update_graph_and_text(selected_option, slider_value, parameter):
    fig = generate_figure(selected_option, slider_value, parameter)
    message = f"Slider is at {slider_value}"
    return fig, message
```

---

## 📋 Key Concepts

| Term     | Purpose                                           |
| ------   | ------------------------------------------------- |
| `Input`  | Triggers the callback when the input changes      |
| `Output` | The component property to be updated              |
| `State`  | Passes data to the function without triggering it |

---

## 🗒️ Development Notes

- All Outputs must be returned in the same order they’re defined.

- If necessary, use `prevent_initial_call=True` in `@callback` to skip execution on page load:

``` python
@callback(..., prevent_initial_call=True)
```

- To avoid updating, either raise `dash.exceptions.PreventUpdate` exception to abort the whole callback, or return `dash.no_update` for each of the outputs that you do not wish to update:

    ``` python
    from dash.exceptions import PreventUpdate
    if not input_value:
        raise PreventUpdate
    ```

    or 

    ``` python
    @callback(
        Output("result", "children"),
        Input("submit-btn", "n_clicks"),
        State("input-text", "value")
    )
    def submit_text(n_clicks, input_value):
        if n_clicks:
            return f"You submitted: {input_value}"
        return dash.no_update
    ```

---

## 📚 Resources

To better understand how callbacks work in Dash, see the official documentation:

🔗 [Dash Callbacks – Official Docs](https://dash.plotly.com/basic-callbacks)
