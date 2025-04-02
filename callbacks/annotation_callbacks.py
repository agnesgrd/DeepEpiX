import plotly.graph_objects as go
import dash
from dash import Patch, Input, Output, State, callback
import callbacks.utils.graph_utils as gu
import plotly.graph_objects as go
import pandas as pd
import itertools



def register_update_annotations():
    @callback(
        Output("meg-signal-graph", "figure", allow_duplicate=True),
        Input("meg-signal-graph", "figure"),  # Current figure to update
        Input("annotation-checkboxes", "value"),  # Annotations to show based on the checklist
        Input("annotation-checkboxes", "options"),
        Input("page-selector", "value"),
        State("annotations-store", "data"),
        State("chunk-limits-store", "data"),
        prevent_initial_call=True,
        supress_callback_exceptions=True
    )
    def update_annotations(fig_dict, annotations_to_show, annotation_options, page_selection, annotations, chunk_limits):
        """Update annotations visibility based on the checklist selection."""
        # Default time range in case the figure doesn't contain valid x-axis range data

        if not annotations_to_show:
            return dash.no_update  # No annotations available, return the same graph
        
        if len(annotations_to_show)==0:
            return dash.no_update

        time_range = chunk_limits[int(page_selection)]

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

        # Define a color palette (extend as needed)
        color_palette = itertools.cycle(["red", "blue", "green", "purple", "orange", "brown", "pink"])

        # Dictionary to store description-to-color mapping
        description_colors = {}

        for _, row in filtered_annotations_df.iterrows():
            description = row["description"]
            duration = row['duration']

                # Assign a consistent color for each description
            if description not in description_colors:
                description_colors[description] = next(color_palette)

            color = description_colors[description]  # Get assigned color

            if str(description) in annotations_to_show:
                 #Check the duration and add either a vertical line or a rectangle
                if duration == 0:
                    # Vertical line if duration is 0
                    new_shapes.append(
                        dict(
                            type="line",
                            x0=row.name,
                            x1=row.name,
                            y0=y_min,
                            y1=y_max,
                            xref="x",
                            yref="y",
                            line=dict(color=color, width=2, dash="dot"),
                            opacity=0.25
                        )
                    )
                else:
                    # Rectangle if duration > 0
                    new_shapes.append(
                        dict(
                            type="rect",
                            x0=row.name,
                            x1=row.name + duration,
                            y0=y_min,
                            y1=y_max,
                            xref="x",
                            yref="y",
                            line=dict(color=color, width=2),
                            fillcolor=color,  # Set the color of the rectangle
                            opacity=0.25
                        )
                    )
                # Add the label in the margin
                new_annotations.append(
                    dict(
                        x=row.name - 0.02,
                        y=0.98,  # Slightly above the graph in the margin
                        xref="x",
                        yref="paper",  # Use paper coordinates for the y-axis (margins)
                        text=description,  # Annotation text
                        showarrow=False,  # No arrow needed
                        font=dict(size=10, color=color),  # Customize font
                        align="center",
                        textangle =-90
                    )
                )

        # Update the figure with the new shapes and annotations
        fig_patch["layout"]["shapes"] = new_shapes
        fig_patch["layout"]["annotations"] = new_annotations

        return fig_patch
        
def register_update_annotation_graph():
    @callback(
        Output("annotation-graph", "figure"),
        Input("annotation-checkboxes", "options"),
        Input("annotation-checkboxes", "value"),
        Input("page-selector", "value"),
        State("annotations-store", "data"),
        State("annotation-graph", "figure"),
        State("chunk-limits-store", "data"),
        prevent_initial_call=True
    )
    def update_annotation_graph(annotation_options, annotations_to_show, page_selection, annotations_data, annotation_fig, chunk_limits):

        if not annotations_data or not isinstance(annotations_data, list):
            return dash.no_update
        
        if not annotations_to_show and not isinstance(annotations_to_show, list):
            return dash.no_update

        time_range = chunk_limits[int(page_selection)]

        # Convert annotations to DataFrame
        try:
            annotations_df = pd.DataFrame(annotations_data).set_index("onset")
        except Exception as e:
            print("Error creating DataFrame:", e)
            return dash.no_update

        # Create the annotation graph
        tick_vals = []
        tick_labels = []
        shapes = []

        for _, row in annotations_df.iterrows():
            if row["description"] in annotations_to_show:
                if row["duration"] == 0:
                    tick_vals.append(row.name)  # Use the onset time as the tick position
                    tick_labels.append(round(row.name, 2))

                if row["duration"] > 0:
                    shapes.append({
                        'type': 'rect',
                        'x0': row.name,
                        'x1': row.name + row["duration"],  # Duration defines the rectangle width
                        'y0': 0,  # Starting y-position of rectangle
                        'y1': 1,  # Ending y-position of rectangle
                        'line': {'color': 'rgba(0,0,255,0.6)', 'width': 2},  # Rectangle line color
                        'fillcolor': 'rgba(0,0,255,0.2)',  # Rectangle fill color
                    })

        # Update the figure with the new shapes and annotations
        fig_patch = go.Figure(annotation_fig)

        # Add styling improvements
        fig_patch.update_layout(
            xaxis=dict(
                range=time_range,
                showgrid=True,
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_labels,
                # ticklen=0.1,  # Shorter ticks
                tickfont=dict(size=10),  # Smaller font size
                showticklabels=True,
                gridcolor="red",  # Subtle grid color
                tickangle=0  # Angle for better readability
            ),
            yaxis=dict(
                range=[0, 1],  # More generous y-axis range
                showgrid=False
            ),
            shapes=shapes,
            paper_bgcolor="white",  # Light background for clarity
            plot_bgcolor= "rgba(0,0,0,0)",  # Keep the plot transparent
            margin=dict(l=10, r=0, t=10, b=100)  # Adjust margins for better spacing
        )


        return fig_patch
    
def register_move_to_next_annotation():
    @callback(
        Output("meg-signal-graph", "figure", allow_duplicate=True),
        Input("prev-spike", "n_clicks"),
        Input("next-spike", "n_clicks"),
        State("meg-signal-graph", "figure"),
        State("annotation-checkboxes", "value"),
        State("annotations-store", "data"),
        State("page-selector", "value"),
        State("chunk-limits-store", "data"),
        State("montage-radio", "value"),
        State("channel-region-checkboxes", "value"),
        State("montage-store", "data"),
        State("offset-display", "children"),
        State("folder-store", "data"),
        State("frequency-store", "data"),
        State("colors-radio", "value"),
        State("sensitivity-analysis-store", "data"),
        prevent_initial_call = True
    )
    def move_to_next_annotation(prev_spike, next_spike, graph, annotations_to_show, annotations_data, page_selection, chunk_limits, montage_selection, channel_selection, montage_store, offset_selection, folder_path, freq_data, color_selection, sensitivity_analysis_store):

        if not annotations_data or not annotations_to_show:
            return dash.no_update  # No annotations available, return the same graph
        
        if len(annotations_to_show)==0 or len(annotations_data)==0:
            return dash.no_update

        # Extract x-coordinates (onset times) of spikes from annotations
        spike_x_positions = [
            ann["onset"] for ann in annotations_data if ann["description"] in annotations_to_show
        ]
        spike_x_positions = sorted(spike_x_positions)  # Ensure sorted order

        if not spike_x_positions:
            return graph  # No spikes to navigate

        # Get the current x-axis center
        xaxis_range = graph["layout"]["xaxis"].get("range", [])
        if xaxis_range:
            current_x_center = sum(xaxis_range) / 2  # Midpoint of current view
        else:
            current_x_center = spike_x_positions[0]  # Default to first spike

        # Determine next or previous spike
        if dash.ctx.triggered_id == "next-spike":
            next_spike_x = next((x for x in spike_x_positions if x > current_x_center), spike_x_positions[-1])
        elif dash.ctx.triggered_id == "prev-spike":
            next_spike_x = next((x for x in reversed(spike_x_positions) if x < current_x_center), spike_x_positions[0])
        else:
            return graph  # No valid button click
        
        time_range_limits = chunk_limits[int(page_selection)]
                
        # Extract time range limits
        time_range_min, time_range_max = time_range_limits

        # Compute x-axis range offset
        x_range_offset = (xaxis_range[1] - xaxis_range[0]) / 2 if xaxis_range else 10

        # Default centered range
        proposed_x_min = next_spike_x - x_range_offset
        proposed_x_max = next_spike_x + x_range_offset

        # if next_spike_x < time_range_min:
        #     if montage_selection == "channel selection":
        #         # Get the selected channels based on region
        #         selected_channels = [
        #             channel
        #             for region_code in channel_selection
        #             if region_code in c.GROUP_CHANNELS_BY_REGION
        #             for channel in c.GROUP_CHANNELS_BY_REGION[region_code]
        #         ]

        #         if not selected_channels:
        #             raise ValueError("No channels selected from the given regions.")

        #     else: 

        #         # If montage selection is not "channel selection", use montage's corresponding channels
        #         selected_channels = montage_store.get(montage_selection, [])

        #     # Reading back
        #     if "smoothGrad" in color_selection:
        #         with open(sensitivity_analysis_store['smoothGrad'], 'rb') as f:
        #             sensitivity_analysis = pickle.load(f)
        #     else:
        #         sensitivity_analysis = {}


        #     # Adjust if near the edges
        #     if proposed_x_min < time_range_min:
        #         x_min, x_max = time_range_min, time_range_min + 2 * x_range_offset
        #     elif proposed_x_max > time_range_max:
        #         x_min, x_max = time_range_max - 2 * x_range_offset, time_range_max
        #     else:
        #         x_min, x_max = proposed_x_min, proposed_x_max

        #     # Ensure the adjusted range is within valid limits
        #     x_min = max(x_min, time_range_min)
        #     x_max = min(x_max, time_range_max)
        #     xaxis_range=[x_min, x_max]
        #     return gu.generate_graph_time_channel(selected_channels, float(offset_selection), chunk_limits[int(page_selection)-1], folder_path, freq_data, color_selection, sensitivity_analysis, xaxis_range)

        # Adjust if near the edges
        if proposed_x_min < time_range_min:
            x_min, x_max = time_range_min, time_range_min + 2 * x_range_offset
        elif proposed_x_max > time_range_max:
            x_min, x_max = time_range_max - 2 * x_range_offset, time_range_max
        else:
            x_min, x_max = proposed_x_min, proposed_x_max

        # Ensure the adjusted range is within valid limits
        x_min = max(x_min, time_range_min)
        x_max = min(x_max, time_range_max)

        # Update the graph layout
        graph["layout"]["xaxis"]["range"] = [x_min, x_max]

        return graph