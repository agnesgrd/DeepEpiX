# view.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import mne  # Import the MNE package
import io
import base64
import matplotlib.pyplot as plt
import plotly.graph_objs as go  # Import Plotly for graphing

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
    Input("folder-store", "data"),  # Access the stored data
    Input("frequency-store", "data"),
    State("preprocessed-data-store", "data"),
    prevent_initial_call=False
)

def display_meg(folder_path, freq_params, preprocessed_data):
    """Preprocess and display MEG signal as a Plotly graph."""
    if preprocessed_data is None:
        raise ValueError("No data available.")
    
    try:
        times = preprocessed_data["times"]
        data = preprocessed_data["data"]
        channel_names = preprocessed_data["channel_names"]

            # picks = mne.pick_types(raw.info, meg=True, exclude='bads')
            # t_idx = raw.time_as_index([0, 180.])  
            # data, times = raw[picks, t_idx[0]:t_idx[1]] 
            # # Extract the first 5 channels, first 1000 time points
            # channel_names = [raw.ch_names[pick] for pick in picks] # Get channel names for the first 5 channels

        # Create traces for each channel
        traces = []
        for i, channel in enumerate(channel_names):
            traces.append(go.Scatter(
                x=times,  # Time points
                y=data[i],  # Data for the channel
                mode="lines",
                name=channel  # Label each trace with the channel name
            ))

        # Define the layout for the Plotly graph
        layout = go.Layout(
            title="MEG Signal Visualization (Preprocessed)",
            xaxis=dict(title="Time (s)"),
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