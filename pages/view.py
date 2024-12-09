# view.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go  # Import Plotly for graphing
from plotly.subplots import make_subplots
from pages.home import get_preprocessed_dataframe
import pandas as pd
import static.constants as constants


dash.register_page(__name__)

layout = html.Div([
    html.H1("VIEW: Visualize and Annotate MEG Signal"),

    # Display folder path
    html.Div(id="display-folder-path", style={"padding": "10px", "font-style": "italic", "color": "#555"}),

        # Slider and Graph Container
    html.Div([
        # Channel Slider
        html.Div([
            html.Label(
                "Select Channels:"
            ),
            dcc.RangeSlider(
                id="channel-slider",
                min=0,
                max=len(constants.all_ch_names) - 1,
                step=1,
                marks = {i: {'label': constants.all_ch_names_prefix[i] if i % 10 == 0 else '', 'style': {'fontSize': '10px'}} for i in range(len(constants.all_ch_names_prefix))},
                value=[0, 10],
                vertical=True  # Makes the slider vertical
            )
        ], style={
            "padding": "10px", 
            "height": "100%", 
            "display": "flex", 
            "flexDirection": "column", 
            "justifyContent": "center"
        }),

        # Graph on the Right
        html.Div([
            dcc.Graph(id="meg-signal-graph")
        ], style={"flexGrow": 1, "padding": "10px"})
    ], style={
        "display": "flex", 
        "height": "500px"  # Flex container for slider and graph
    }),

    # Time Selector
    html.Div([
        html.Label("Select Time Range (s):"),
        dcc.RangeSlider(
            id="time-slider",
            min=0,
            max=100,  # Default max, update dynamically
            step=0.1,
            marks=None,
            value=[0, 10]  # Default range
        )
    ], style={"padding": "20px"})
])

# Callback to display the stored folder path
@dash.callback(
    Output("display-folder-path", "children"),
    Input("folder-store", "data"),  # Access the stored data
    prevent_initial_call=False
)
def display_folder_path(folder_path):
    """Display the stored folder path."""
    return f"{folder_path}" if folder_path else "No folder path has been selected."

@dash.callback(
    Output("meg-signal-graph", "figure"),
    Input("time-slider", "value"),  # Time range selection
    Input("channel-slider", "value"),  # Channel range selection
    State("session-id", "data"),
    State("folder-store", "data"),
    State("frequency-store", "data"),
    prevent_initial_call=False
)
def update_meg_signal(time_range, channel_range, session_id, folder_path, freq_data):
    """Update MEG signal visualization based on time and channel selection."""
    if session_id is None:
        raise ValueError("No data available.")
    
    try:
        # Preprocess data
        raw_df = get_preprocessed_dataframe(session_id, folder_path, freq_data)

        # Convert index to seconds
        times = (raw_df.index - raw_df.index[0]).total_seconds()

        # Filter time range
        time_mask = (times >= time_range[0]) & (times <= time_range[1])
        filtered_times = times[time_mask]
        filtered_df = raw_df[time_mask]

        # Filter channel range
        start_channel, end_channel = channel_range
        selected_channels = constants.all_ch_names[start_channel:end_channel + 1]
        filtered_df = filtered_df[selected_channels]

        # Downsample data if necessary
        if len(filtered_times) > 5000:  # Arbitrary threshold
            step = len(filtered_times) // 5000
            filtered_times = filtered_times[::step]
            filtered_df = filtered_df.iloc[::step, :]

        # Create subplots with shared x-axis
        fig = make_subplots(
            rows=len(selected_channels), cols=1, shared_xaxes=True,  # Shared x-axis, different y-axes
            vertical_spacing=0,  # Space between subplots
            row_heights=[1] * len(selected_channels)  # Make the rows equal height
        )

        # Add traces for each channel with a separate y-axis
        for i, channel in enumerate(selected_channels):
            fig.add_trace(
                go.Scattergl(
                    x=times,
                    y=filtered_df.get(channel, [0] * len(times)),  # Data for the specific channel
                    mode='lines',
                    name=channel
                ),
                row=i+1, col=1  # Place each trace in its own row
            )

        fig.update_xaxes(
            title_text="Time (s)", row = len(selected_channels), col = 1
        )
        
        # Update the layout to include titles and labels
        fig.update_layout(
            title="MEG Signal Visualization",
            yaxis=dict(title="Amplitude"),
            showlegend=True
        )

        # Adjust each y-axis label for better clarity (optional)
        for i in range(1, len(selected_channels) + 1):
            fig.update_yaxes(
                showline=True,
                showgrid=False,
                zeroline=True,
                tickvals=[]

            )

        return fig  # Return the figure to be rendered in the graph

    except FileNotFoundError:
        return go.Figure().update_layout(title="Error: Folder not found.")
    except ValueError as ve:
        return go.Figure().update_layout(title=f"Error: {str(ve)}")
    except Exception as e:
        return go.Figure().update_layout(title=f"Error: Unexpected error {str(e)}")


# Callback to apply preprocessing and display the MEG signal
# @dash.callback(
#     Output("meg-signal-graph", "figure"),  # Correctly linked to the dcc.Graph component
#     Input("session-id", "data"),
#     State("folder-store", "data"),  # Access the stored data
#     State("frequency-store", "data"),
#     prevent_initial_call=False
# )

# def display_meg(session_id, folder_path, freq_data):
#     """Preprocess and display MEG signal as a Plotly graph with a range slider."""
#     if session_id is None:
#         raise ValueError("No data available.")
    
#     try:
#         raw_df = get_preprocessed_dataframe(session_id, folder_path, freq_data)

#         Extract timestamps and ensure they are in float (seconds since start)
#         times = (raw_df.index - raw_df.index[0]).total_seconds() # Convert to seconds

#         Extract channel names
#         channel_names = constants.all_ch_names #raw_df.columns.tolist()

#         Create traces for each channel
#         traces = []
#         for channel in channel_names:
#             channel_data = raw_df[channel]  # Data for the specific channel
#             traces.append(go.Scattergl(
#                 x=times,  # Time points
#                 y=channel_data,  # Data for the channel
#                 mode="lines",
#                 name=channel  # Label each trace with the channel name
#             ))

#         Define the layout for the Plotly graph, including the range slider
#         layout = go.Layout(
#             title="MEG Signal Visualization (Preprocessed)",
#             xaxis=dict(
#                 title="Time (s)",
#                 rangeslider=dict(visible=True),  # Enable the range slider
#                 type="linear"  # Ensure the x-axis is linear
#             ),
#             yaxis=dict(title="Amplitude"),
#             legend=dict(title="Channels"),
#         )

#         Return the figure to be rendered
#         return go.Figure(data=traces, layout=layout)

#     except FileNotFoundError:
#         return go.Figure().update_layout(
#             title="Error: Folder not found or incorrect path.",
#             xaxis=dict(title="Time (s)"),
#             yaxis=dict(title="Amplitude")
#         )
#     except ValueError as ve:
#         return go.Figure().update_layout(
#             title=f"Error: Invalid parameter value. {str(ve)}",
#             xaxis=dict(title="Time (s)"),
#             yaxis=dict(title="Amplitude")
#         )
#     except Exception as e:
#         return go.Figure().update_layout(
#             title=f"Error: An unexpected error occurred. {str(e)}",
#             xaxis=dict(title="Time (s)"),
#             yaxis=dict(title="Amplitude")
#         )
