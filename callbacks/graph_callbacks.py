# from dash_extensions.enrich import Output, Input, State, Patch
import traceback
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash import Patch
import static.constants as c
import callbacks.utils.graph_utils as gu
import callbacks.utils.annotation_utils as au
import traceback
import plotly.graph_objects as go
import pandas as pd

def register_callbacks_annotation_names():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("annotation-checkboxes", "options"),
        Output("annotation-checkboxes", "value"),
        Input("annotations-store", "data"),
        prevent_initial_call = False
    )
    def display_annotation_names_checklist(annotations_store):
        description_counts = au.get_annotation_descriptions(annotations_store)

        options = [{'label': f"{name} ({count})", 'value': f"{name}"} for name, count in description_counts.items()]
        value = [f"{name}" for name in description_counts.keys()]
        return options, dash.no_update  # Set all annotations as default selected
    
def register_callbacks_montage_names():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("montage-radio", "options"),
        Output("montage-radio", "value"),
        Input("montage-store", "data"),
        State("montage-radio", "value"),
        prevent_initial_call=False
    )
    def display_annotation_names_checklist(montage_store, value):
        # Create options for the checklist from the channels in montage_store
        options = [{'label': key, 'value': key} for key in montage_store.keys()]
        options.append({'label': 'channel selection', 'value': 'channel selection'})

        # If montage_store is empty or the current value is not valid, select the first option
        valid_values = [option['value'] for option in options]
        if not montage_store or value not in valid_values:
            # Return the first option as default
            return options, options[0]['value']

        # If value is valid, keep the current selection
        return options, value
    
# def register_callbacks_montage_names():
#     # Callback to populate the checklist options and default value dynamically
#     @dash.callback(
#         Output("montage-radio", "options"),
#         Output("montage-radio", "value"),
#         Input("montage-store", "data"),
#         State("montage-radio", "value"),
#         prevent_initial_call=False
#     )
#     def display_annotation_names_checklist(montage_store, value):
#         # Create options for the checklist from the channels in montage_store
#         options = []
        
#         for key, channels in montage_store.items():
#             options.extend([{'label': key, 'value': key}])

#         options.extend([{'label':'channel selection', 'value': 'channel selection'}])

#         if montage_store == {} or value is None:
#             return options, 'channel selection'

#         else:
#             return options, dash.no_update
        
def register_hide_channel_selection_when_montage():
    # Callback to populate the checklist options and default value dynamically
    @dash.callback(
        Output("channel-region-checkboxes", "options"),
        Input("montage-radio", "options"),
        Input("montage-radio", "value"),
        State("channel-region-checkboxes", "options"),
        prevent_initial_call=False
    )
    def display_annotation_names_checklist(montage_option, montage_value, channel_options):
        if montage_value != 'channel selection':
            # Disable all options
            return [{'label': option['label'], 'value': option['value'], 'disabled': True} for option in channel_options]
        else:
            # Enable all options
            return [{'label': option['label'], 'value': option['value'], 'disabled': False} for option in channel_options]
    
def register_update_graph_time_channel(): 
    @dash.callback(
        Output("meg-signal-graph", "figure"),
        Output("python-error", "children"),
        Input("montage-radio", "value"),
        Input("channel-region-checkboxes", "value"),
        #Input("annotation-checkboxes", "value"),
        Input("folder-store", "data"),
        State("frequency-store", "data"),
        #State("annotations-store", "data"),
        State("montage-store", "data"),
        State("meg-signal-graph", "figure"),
        prevent_initial_call=False
    )
    def update_graph_time_channel(montage_selection, channel_selection, folder_path, freq_data, montage_store, graph):
        """Update MEG signal visualization based on time and channel selection."""

        time_range = [0,180]

        try:
            if montage_selection == "channel selection" and not channel_selection or not folder_path or not freq_data:  # Check if data is missing
                return go.Figure(), "Missing data for graph rendering."
            
            else:

                if montage_selection == "channel selection":
                    # Get the selected channels based on region
                    selected_channels = [
                        channel
                        for region_code in channel_selection
                        if region_code in c.GROUP_CHANNELS_BY_REGION
                        for channel in c.GROUP_CHANNELS_BY_REGION[region_code]
                    ]

                    if not selected_channels:
                        raise ValueError("No channels selected from the given regions.")

                else: 

                    # If montage selection is not "channel selection", use montage's corresponding channels
                    selected_channels = montage_store.get(montage_selection, [])
                    
                    # If there are no channels for the selected montage
                    if not selected_channels:
                        raise ValueError(f"No channels available for the selected montage: {montage_selection}")
                
                fig = gu.generate_graph_time_channel(selected_channels, time_range, folder_path, freq_data)

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
        Input("meg-signal-graph", "figure"),  # Current figure to update
        Input("annotation-checkboxes", "value"),  # Annotations to show based on the checklist
        Input("annotation-checkboxes", "options"),
        State("annotations-store", "data"),
        prevent_initial_call=True,
        supress_callback_exceptions=True
    )
    def update_annotations(fig_dict, annotations_to_show, annotation_options, annotations):
        """Update annotations visibility based on the checklist selection."""
        # Default time range in case the figure doesn't contain valid x-axis range data

        time_range = [0, 180]

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

        return fig_patch
        
def register_update_annotation_graph():
    @dash.callback(
        Output("annotation-graph", "figure"),
        Input("annotation-checkboxes", "options"),
        Input("annotation-checkboxes", "value"),
        State("annotations-store", "data"),
        State("annotation-graph", "figure"),
        prevent_initial_call=True
    )
    def update_annotation_graph(annotation_options, annotations_to_show, annotations_data, annotation_fig):

        if not annotations_data or not isinstance(annotations_data, list):
            return dash.no_update

        time_range = [0, 180]

        # Convert annotations to DataFrame
        try:
            annotations_df = pd.DataFrame(annotations_data).set_index("onset")
        except Exception as e:
            print("Error creating DataFrame:", e)
            return dash.no_update

        # Filter annotations based on the current time range
        try:
            filtered_annotations_df = gu.get_annotations_df_filtered_on_time(time_range, annotations_df)
        except Exception as e:
            print("Error filtering annotations:", e)
            return dash.no_update

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

        return fig_patch
    
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

        # Get the current x-axis range
        xaxis_range = fig["layout"]["xaxis"]["range"]
        move_amount = 1/3*(xaxis_range[1]-xaxis_range[0])  # Number of seconds to move

        # Define the bounds for the x-axis (adjust based on your data)
        min_bound = 0
        max_bound = 180

        # Update the range based on the key press
        if keydown["key"] == "ArrowLeft":
            new_range = [xaxis_range[0] - move_amount, xaxis_range[1] - move_amount]
            if new_range[0] < min_bound:
                new_range = [min_bound, min_bound + move_amount]
        elif keydown["key"] == "ArrowRight":
            new_range = [xaxis_range[0] + move_amount, xaxis_range[1] + move_amount]
            if new_range[1] > max_bound:
                new_range = [max_bound - move_amount, max_bound]
        else:
            return fig  # Return the figure unchanged if the key is not handled

        fig_patch = Patch()
        # Update the figure with the new x-axis range
        fig_patch["layout"]["xaxis"]["range"] = new_range

        return fig_patch