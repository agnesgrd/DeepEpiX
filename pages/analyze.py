# analyze.py: Analyze Page
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import mne  # Import the MNE package
import plotly.graph_objs as go  # Import Plotly for graphing
from layout import input_styles
import numpy as np

dash.register_page(__name__)

layout = html.Div([
    html.H1("ANALYZE: Signal Analysis"),

    # Display folder path
    html.Div(id="display-folder-path-analyze", style={"padding": "10px", "font-style": "italic", "color": "#555"}),

    # Button to load and proceed to the analyze
    html.Div([
        dbc.Button(
            "Plot Topomap",
            id="topomap-button",
            color="success",
            disabled=True,
            n_clicks=0
        )
    ], style={"padding": "10px", "margin-top": "20px"}),

    html.Div(
        id = "topomap-display", 
        children = [
            html.Div([
            html.Label("Timestamp (s): "),
            dcc.Input(id="user-selected-time", type="number", value=10, step=1, min=0, style=input_styles["number"]),
        ], style={"padding": "10px"}),

    # Graph for the MEG signal
    html.Div(id="meg-topomap")], style={"display": "none"})  # Correct component for the graph
])

# @dash.callback(
#     Output("display-folder-path-analyze", "children"),
#     Output("topomap-button", "disabled"),
#     Input("folder-store", "data"),
#     prevent_initial_call=False
# )
# def display_folder_path_analyze(folder_path):
#     """Display the stored folder path."""
#     return (f"The selected folder path is: {folder_path}", False) if folder_path else ("No folder path has been selected.", None)

# @dash.callback(
#     Output("topomap-display", "style"),
#     Input("topomap-button", "n_clicks"),
#     prevent_initial_call=True
# )
# def handle_load_button(n_clicks):
#     """Display frequency parameters when button is clicked"""
#     if n_clicks > 0:
#         return {"display": "block"}


# @dash.callback(
#     Output("meg-topomap", "children"),  # Output a topomap figure to a dcc.Graph component
#     Input("preprocessed-data-store", "data"),  # Access the preprocessed data
#     Input("user-selected-time", "value"),  # User-input time in seconds
#     prevent_initial_call=True
# )
# def display_topomap(preprocessed_data, selected_time):
#     """Display MEG topomap for the selected time point."""
#     if preprocessed_data is None:
#         raise ValueError("No data available.")

#     try:
#         # Retrieve data and metadata
#         data = np.array(preprocessed_data["data"])  # (n_channels, n_times)
#         times = np.array(preprocessed_data["times"])  # (n_times,)
#         channel_names = preprocessed_data["channel_names"]
#         sfreq = preprocessed_data["sfreq"]  # Sampling frequency
        
#         # Simulated channel locations for visualization (for real data, use `raw.info`)
#         ch_pos = mne.channels.make_standard_montage("biosemi64").get_positions()
#         sensor_positions = np.array([ch_pos["ch_pos"][ch] for ch in channel_names if ch in ch_pos["ch_pos"]])
        
#         # Check if the user-selected time is valid
#         if selected_time is None or selected_time < 0 or selected_time > times[-1]:
#             raise ValueError("Selected time is outside the range of the data.")
        
#         # Find the index for the selected time
#         time_idx = np.argmin(np.abs(times - selected_time))
#         data_at_time = data[:, time_idx]  # Data at the selected time point

#         # Create a topomap using Plotly's scatter plot
#         fig = go.Figure()

#         # Plot topomap as a scatter plot with color scale
#         fig.add_trace(
#             go.Scatter(
#                 x=sensor_positions[:, 0],  # X-coordinates
#                 y=sensor_positions[:, 1],  # Y-coordinates
#                 mode="markers",
#                 marker=dict(
#                     size=10,
#                     color=data_at_time,  # Color corresponds to data values
#                     colorscale="Viridis",
#                     colorbar=dict(title="Amplitude"),
#                 ),
#                 text=channel_names,  # Hover text
#             )
#         )

#         # Update layout for the topomap
#         fig.update_layout(
#             title=f"Topomap at {selected_time:.2f} seconds",
#             xaxis=dict(title="X Position", showgrid=False, zeroline=False),
#             yaxis=dict(title="Y Position", showgrid=False, zeroline=False),
#             template="plotly_white",
#         )

#         return fig

#     except ValueError as ve:
#         return go.Figure().update_layout(
#             title=f"Error: {str(ve)}",
#         )
#     except Exception as e:
#         return go.Figure().update_layout(
#             title=f"Unexpected Error: {str(e)}",
#         )
