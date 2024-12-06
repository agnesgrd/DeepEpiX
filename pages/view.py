# view.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import mne  # Import the MNE package
import io
import base64
import matplotlib.pyplot as plt
import plotly.graph_objs as go  # Import Plotly for graphing
from pages.home import get_preprocessed_dataframe
import pandas as pd
import static.constants as constants

dash.register_page(__name__)

layout = html.Div([
    html.H1("VIEW: Visualize and Annotate MEG Signal"),

    # Display folder path
    html.Div(id="display-folder-path", style={"padding": "10px", "font-style": "italic", "color": "#555"}),

    # Graph for the MEG signal
    dcc.Graph(id="meg-signal-graph"),  # This is the correct component for a graph
])

# Callback to display the stored folder path
@dash.callback(
    Output("display-folder-path", "children"),
    Input("folder-store", "data"),  # Access the stored data
    prevent_initial_call=False
)
def display_folder_path(folder_path):
    """Display the stored folder path."""
    return f"The selected folder path is: {folder_path}" if folder_path else "No folder path has been selected."

# Callback to apply preprocessing and display the MEG signal
@dash.callback(
    Output("meg-signal-graph", "figure"),  # Correctly linked to the dcc.Graph component
    Input("session-id", "data"),
    State("folder-store", "data"),  # Access the stored data
    State("frequency-store", "data"),
    prevent_initial_call=False
)

def display_meg(session_id, folder_path, freq_data):
    """Preprocess and display MEG signal as a Plotly graph with a range slider."""
    if session_id is None:
        raise ValueError("No data available.")
    
    try:
        raw_df = get_preprocessed_dataframe(session_id, folder_path, freq_data)

        # Extract timestamps and ensure they are in float (seconds since start)
        times = (raw_df.index - raw_df.index[0]).total_seconds() # Convert to seconds

        # Extract channel names
        channel_names = constants.left_ch_names #raw_df.columns.tolist()

        # Create traces for each channel
        traces = []
        for channel in channel_names:
            channel_data = raw_df[channel]  # Data for the specific channel
            traces.append(go.Scatter(
                x=times,  # Time points
                y=channel_data,  # Data for the channel
                mode="lines",
                name=channel  # Label each trace with the channel name
            ))

        # Define the layout for the Plotly graph, including the range slider
        layout = go.Layout(
            title="MEG Signal Visualization (Preprocessed)",
            xaxis=dict(
                title="Time (s)",
                rangeslider=dict(visible=True),  # Enable the range slider
                type="linear"  # Ensure the x-axis is linear
            ),
            yaxis=dict(title="Amplitude"),
            legend=dict(title="Channels"),
        )

        # Return the figure to be rendered
        return go.Figure(data=traces, layout=layout)

    except FileNotFoundError:
        return go.Figure().update_layout(
            title="Error: Folder not found or incorrect path.",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude")
        )
    except ValueError as ve:
        return go.Figure().update_layout(
            title=f"Error: Invalid parameter value. {str(ve)}",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude")
        )
    except Exception as e:
        return go.Figure().update_layout(
            title=f"Error: An unexpected error occurred. {str(e)}",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(title="Amplitude")
        )
