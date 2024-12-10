# view.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go  # Import Plotly for graphing
from plotly.subplots import make_subplots
from pages.home import get_preprocessed_dataframe
import pandas as pd
import static.constants as constants
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB
import numpy as np
import traceback


dash.register_page(__name__)

layout = html.Div([
    html.H1("VIEW: Visualize and Annotate MEG Signal"),

    # Display folder path
    html.Div(id="display-folder-path", style={"padding": "10px", "font-style": "italic", "color": "#555"}),

    # Slider and Graph Container
    html.Div([
        # Channel Slider
        html.Div([
            html.Label("Select Channels:"),
            dcc.RangeSlider(
                id="channel-slider",
                min=0,
                max=len(constants.all_ch_names) - 1,
                step=1,
                marks={i: {'label': constants.all_ch_names_prefix[i] if i % 10 == 0 else '', 'style': {'fontSize': '10px'}} for i in range(len(constants.all_ch_names_prefix))},
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
        "height": "500px",  # Flex container for slider and graph
        "gap":"10px"
        
    }),

    # Time Selector
	html.Div([
		html.Label("Select Time Range (s):"),
		html.Div([
			dcc.RangeSlider(
				id="time-slider",
				min=0,
				max=100,  # Default max, update dynamically
				step=0.1,
				marks={i: str(i) for i in range(0, 101, 10)},  # Add visible marks
				value=[0, 10],  # Default range
				tooltip={"placement": "bottom", "always_visible": True}
			),
		], style={"width": "80%"}),  # Ensure the slider has enough width
		html.Div([
			html.Button("←", id="time-left", n_clicks=0, style={"font-size": "16px", "padding": "5px 10px"}),
			html.Button("→", id="time-right", n_clicks=0, style={"font-size": "16px", "padding": "5px 10px", "margin-left": "10px"})
		], style={"display": "flex", "flex-direction": "row", "align-items": "center", "margin-top": "10px"})
	], style={"display": "flex", "flex-direction": "column", "align-items": "center", "width": "100%", "margin-top": "50px"}),

	html.Div(id="python-error", style={"padding": "10px", "font-style": "italic", "color": "#555"})
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
    Output("python-error", "children"),
    # Input("time-left", "n_clicks"),
    # Input("time-right", "n_clicks"),
    Input("time-slider", "value"),  # Time range selection
    State("channel-slider", "value"),  # Channel range selection
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

        # Offset channel traces along the y-axis
        channel_offset = 500
        y_axis_ticks = np.arange(len(selected_channels))*channel_offset
        filtered_df += np.tile(y_axis_ticks, (len(filtered_df),1))

        # Create the resampled figure
        fig = FigureResampler(
            go.Figure(),
            default_downsampler=MinMaxLTTB(parallel=True),
            resampled_trace_prefix_suffix=("<b style='color:#2bb1d6'>[R]</b> ", ''),
            show_mean_aggregation_size=False
        )

        # Add traces to the figure
        for i, channel_data in enumerate(filtered_df):
            fig.add_trace(go.Scatter(x=filtered_times, y=filtered_df[channel_data], mode='lines')) #name=f'{selected_channels[i]}'))

        # Customize layout
        fig.update_layout(
            xaxis=dict(title='Time (s)'),
            yaxis=dict(
                title='Channels',
                showgrid=True,
                tickvals=y_axis_ticks,  # Align ticks with channels
                ticktext=[f'{selected_channels[i]}' for i in range(len(selected_channels))],  # Label each channel,
                ticklabelposition="outside right"
            ),
            title='MEG Signal Visualization',
            height=600,
            showlegend=False
        )

        return fig, None  # Return the figure to be rendered in the graph

    except FileNotFoundError:
        return go.Figure(), f"Error: Folder not found."
    except ValueError as ve:
        return go.Figure(), f"Error: {str(ve)}.\n Details: {traceback.format_exc()}"
    except Exception as e:
        return go.Figure(), f"Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}"
    
@dash.callback(
    Output("time-slider", "value"),
    Input("time-left", "n_clicks"),
    Input("time-right", "n_clicks"),
    State("time-slider", "value"),
    prevent_initial_call=True
)
def shift_time_range(left_clicks, right_clicks, current_range):
    """Shift the time range left or right."""
    # Define shift step (e.g., 1 second)
    shift_step = 1.0

    # Determine how much to shift based on button clicks
    shift = (right_clicks - left_clicks) * shift_step

    # Update the time range
    start, end = current_range
    new_range = [start + shift, end + shift]

    # Ensure the range stays within bounds
    new_range = [
        max(0, new_range[0]),  # Prevent start from going below 0
        min(100, new_range[1])  # Prevent end from exceeding max time (100 in this case)
    ]

    return new_range