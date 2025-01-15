# from dash_extensions.enrich import Output, Input, State, Patch
import traceback
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash import Patch
from pages.home import get_preprocessed_dataframe
import static.constants as c
import callbacks.utils.graph_utils as gu
import callbacks.utils.annotation_utils as au
import numpy as np
import traceback
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

def register_callbacks_annotation_names():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("annotation-checkboxes", "options"),
        Output("annotation-checkboxes", "value"),
        Input("annotations-store", "data"),
        prevent_initial_call = False
    )
    def display_annotation_names_checklist(annotations_store):
        annotation_names = au.get_annotation_descriptions(annotations_store)
        options = [{'label': name, 'value': name} for name in annotation_names]
        return options, annotation_names  # Set all annotations as default selected
    
def generate_graph_time_channel(channel_region, annotations_to_show, folder_path, freq_data, annotations):
    """Handles the preprocessing and figure generation for the MEG signal visualization."""
    import time  # For logging execution times

    start_time = time.time()
    # Preprocess data
    raw_df = get_preprocessed_dataframe(folder_path, freq_data)
    print(f"Step 1: Preprocessing completed in {time.time() - start_time:.2f} seconds.")

    # Filter time range
    filter_start_time = time.time()
    filtered_times, filtered_raw_df = gu.get_raw_df_filtered_on_time([0, 180], raw_df)
    print(f"Step 2: Time filtering completed in {time.time() - filter_start_time:.2f} seconds.")

    # Get the selected channels based on region
    channel_start_time = time.time()
    selected_channels = [
        channel
        for region_code in channel_region
        if region_code in c.GROUP_CHANNELS_BY_REGION
        for channel in c.GROUP_CHANNELS_BY_REGION[region_code]
    ]
    if not selected_channels:
        raise ValueError("No channels selected from the given regions.")
    print(f"Step 3: Channel selection completed in {time.time() - channel_start_time:.2f} seconds.")

    # Filter the dataframe based on the selected channels
    filter_df_start_time = time.time()
    filtered_raw_df = filtered_raw_df[selected_channels]
    print(f"Step 4: Dataframe filtering completed in {time.time() - filter_df_start_time:.2f} seconds.")

    # Offset channel traces along the y-axis
    offset_start_time = time.time()
    channel_offset = gu.calculate_channel_offset(len(selected_channels)) / 12
    y_axis_ticks = gu.get_y_axis_ticks(selected_channels, channel_offset)
    shifted_filtered_raw_df = filtered_raw_df + np.tile(y_axis_ticks, (len(filtered_raw_df), 1))
    print(f"Step 5: Channel offset calculation completed in {time.time() - offset_start_time:.2f} seconds.")
    # Create a dictionary mapping channels to their colors
    color_map = {channel: c.CHANNEL_TO_COLOR[channel] for channel in selected_channels}
    # Use Plotly Express for efficient figure generation
    fig_start_time = time.time()
    shifted_filtered_raw_df["Time"] = filtered_times  # Add time as a column for Plotly Express
    fig = px.line(
        shifted_filtered_raw_df,
        x="Time",
        y=shifted_filtered_raw_df.columns[:-1],  # Exclude the Time column from y
        labels={"value": "Value", "variable": "Channel", "Time": "Time (s)"},
        color_discrete_map=color_map
    ).add_traces(px.scatter(
        shifted_filtered_raw_df,
        x="Time",
        y=shifted_filtered_raw_df.columns[:-1],  # Exclude the Time column from y
        labels={"value": "Value", "variable": "Channel", "Time": "Time (s)"},
        color_discrete_map=color_map,
        opacity=0).data
    )

    print(f"Step 6: Figure creation completed in {time.time() - fig_start_time:.2f} seconds.")

    # Update layout with x-axis range and other customizations
    layout_start_time = time.time()
    fig.update_layout(
        autosize=True,
        xaxis=dict(
            title='Time (s)',
            range=[0,10],
            fixedrange=False,
            rangeslider=dict(visible=True, thickness=0.02),
        ),
        yaxis=dict(
            title='Channels',
            showgrid=True,
            tickvals=y_axis_ticks,
            ticktext=[f'{selected_channels[i]}' for i in range(len(selected_channels))],
            ticklabelposition="outside right",
            side="right",
            automargin=True,
        ),
        title={
            'text': folder_path if folder_path else 'Select a folder path in Home Page',
            'x': 0.5,
            'font': {'size': 12},
            'automargin': True,
            'yref': 'paper',
        },
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode ='select',
        hovermode ='closest'
    )
    # Update the line width after creation
    for trace in fig.data:
        trace.update(line=dict(width=1))
    print(f"Step 7: Layout update completed in {time.time() - layout_start_time:.2f} seconds.")

    # Total execution time
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")

    return fig

def register_update_graph_time_channel(): 
    @dash.callback(
        Output("meg-signal-graph", "figure"),
        Output("python-error", "children"),
        Input("channel-region-checkboxes", "value"),
        Input("annotation-checkboxes", "value"),
        Input("folder-store", "data"),
        State("frequency-store", "data"),
        State("annotations-store", "data"),
        prevent_initial_call=False
    )
    def update_graph_time_channel(channel_region, annotations_to_show, folder_path, freq_data, annotations):
        """Update MEG signal visualization based on time and channel selection."""
        try:
            if not channel_region or not folder_path or not freq_data:  # Check if data is missing
                return go.Figure(), "Missing data for graph rendering."
            
            fig = generate_graph_time_channel(channel_region, annotations_to_show, folder_path, freq_data, annotations)

            return fig, None
        except FileNotFoundError:
            return go.Figure(), f"Error: Folder not found."
        except ValueError as ve:
            return go.Figure(), f"Error: {str(ve)}.\n Details: {traceback.format_exc()}"
        except Exception as e:
            return go.Figure(), f"Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}"
        
def register_update_annotations():
    @dash.callback(
        Output("meg-signal-graph", "figure", allow_duplicate=True),
        Output("first-load-store", "data", allow_duplicate=True),
        Input("meg-signal-graph", "figure"),  # Current figure to update
        Input("annotation-checkboxes", "value"),  # Annotations to show based on the checklist
        State("annotations-store", "data"),
        State("first-load-store", "data"),
        prevent_initial_call=True,
        supress_callback_exceptions=True
    )
    def update_annotations(fig_dict, annotations_to_show, annotations, first_load):
        """Update annotations visibility based on the checklist selection."""
        # Default time range in case the figure doesn't contain valid x-axis range data
        time_range = [0, 180]
        
        print('updating annotations')

        # Create a Patch for the figure
        fig_patch = Patch()

        # Check if fig_dict is None (i.e., if it is the initial empty figure)
        if fig_dict is None or 'layout' not in fig_dict or 'yaxis' not in fig_dict['layout']:
            # Set default y_min and y_max if the figure layout is not available
            y_min, y_max = 0, 1  # Set default range for the y-axis
        else:
            # Get the current y-axis range from the figure
            y_min, y_max = fig_dict['layout']['yaxis'].get('range', [0, 1])

        # Convert annotations to DataFrame
        annotations_df = pd.DataFrame(annotations).set_index('onset')

        # Filter annotations based on the current time range
        filtered_annotations_df = gu.get_annotations_df_filtered_on_time(time_range, annotations_df)

        # Prepare the shapes and annotations for the selected annotations
        new_shapes = []
        new_annotations = []
        for _, row in filtered_annotations_df.iterrows():
            description = row["description"]
            if description in annotations_to_show:
                new_shapes.append(
                    dict(
                        type="line",
                        x0=row.name,
                        x1=row.name,
                        y0=y_min,
                        y1=y_max,
                        xref="x",
                        yref="y",
                        line=dict(color="red", width=2, dash="dot"),
                        opacity=0.25
                    )
                )
                # Add the label in the margin
                new_annotations.append(
                    dict(
                        x=row.name,
                        y=0.98,  # Slightly above the graph in the margin
                        xref="x",
                        yref="paper",  # Use paper coordinates for the y-axis (margins)
                        text=description,  # Annotation text
                        showarrow=False,  # No arrow needed
                        font=dict(size=10, color="red"),  # Customize font
                        align="center",
                    )
                )

        # Update the figure with the new shapes and annotations
        fig_patch["layout"]["shapes"] = new_shapes
        fig_patch["layout"]["annotations"] = new_annotations

        print("finished adding annotations on main graph")

        return fig_patch, 1
        
def register_update_annotation_graph():
    @dash.callback(
        Output("annotation-graph", "figure"),
        Output("first-load-store", "data"),
        Input("annotation-checkboxes", "value"),
        State("annotations-store", "data"),
        State("annotation-graph", "figure"),
        State("first-load-store", "data"),
        prevent_initial_call=True
    )
    def update_annotation_graph(annotations_to_show, annotations, annotation_fig, first_load):
        print("Callback triggered")
        print("Annotations to show:", annotations_to_show)
        print("Annotations:", annotations)
        print("First load:", first_load)

        if not annotations or not isinstance(annotations, list):
            return dash.no_update, 1

        time_range = [0, 180]

        # Convert annotations to DataFrame
        try:
            annotations_df = pd.DataFrame(annotations).set_index("onset")
            print("Annotations DataFrame:", annotations_df)
        except Exception as e:
            print("Error creating DataFrame:", e)
            return dash.no_update, 1

        # Filter annotations based on the current time range
        try:
            filtered_annotations_df = gu.get_annotations_df_filtered_on_time(time_range, annotations_df)
            print("Filtered Annotations DataFrame:", filtered_annotations_df)
        except Exception as e:
            print("Error filtering annotations:", e)
            return dash.no_update, 1

        # Create the annotation graph
        tick_vals = []
        tick_labels = []
        for _, row in filtered_annotations_df.iterrows():
            if row["description"] in annotations_to_show:
                tick_vals.append(row.name)  # Use the onset time as the tick position
                tick_labels.append(row["description"])  # Use the annotation description as the tick label

        # Update the figure with the new shapes and annotations
        fig_patch = go.Figure(annotation_fig)
    
        fig_patch.update_layout(
            xaxis=dict(
                showgrid=True,
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_labels,
                showticklabels=True,
                tickfont=dict(size=10),
                gridcolor="red"
            )
        )

        return fig_patch, 1
    
def register_manage_channels_checklist():
    @dash.callback(
        Output("channel-region-checkboxes", "value"),
        [Input("check-all-btn", "n_clicks"),
        Input("clear-all-btn", "n_clicks")],
        prevent_initial_call = False
    )
    def manage_checklist(check_all_clicks, clear_all_clicks):
        # Determine which button was clicked
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        all_regions = list(c.GROUP_CHANNELS_BY_REGION.keys())

        if triggered_id == "check-all-btn":
            return all_regions  # Select all regions
        elif triggered_id == "clear-all-btn":
            return []  # Clear all selections

        return dash.no_update
    
def register_move_time_slider():
    @dash.callback(
        Output("meg-signal-graph", "figure", allow_duplicate = True),
        Input("keyboard", "keydown"),
        State("meg-signal-graph", "figure"),
        prevent_initial_call=True,
        supress_callback_exceptions=True
    )
    def move_time_slider(keydown, fig):
        print(f'Key Pressed: {keydown}')

        # Get the current x-axis range
        xaxis_range = fig["layout"]["xaxis"]["range"]
        move_amount = 1/3*(xaxis_range[1]-xaxis_range[0])  # Number of seconds to move

        # Define the bounds for the x-axis (adjust based on your data)
        min_bound = 0
        max_bound = 180

        # Update the range based on the key press
        if keydown["key"] == "ArrowLeft":
            print("left")
            new_range = [xaxis_range[0] - move_amount, xaxis_range[1] - move_amount]
            if new_range[0] < min_bound:
                new_range = [min_bound, min_bound + move_amount]
        elif keydown["key"] == "ArrowRight":
            new_range = [xaxis_range[0] + move_amount, xaxis_range[1] + move_amount]
            print('right')
            if new_range[1] > max_bound:
                new_range = [max_bound - move_amount, max_bound]
        else:
            print("problem")
            return fig  # Return the figure unchanged if the key is not handled

        fig_patch = Patch()
        # Update the figure with the new x-axis range
        fig_patch["layout"]["xaxis"]["range"] = new_range

        return fig_patch