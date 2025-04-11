# analyze.py: Analyze Page
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from layout import input_styles
import static.constants as c
import mne
import os
import io
import random
import base64
import numpy as np
import matplotlib.pyplot as plt


dash.register_page(__name__, name="Settings", path="/settings/montage")

layout = html.Div([

    dcc.Location(id="url", refresh=True),

    html.Div([
    
        html.Div(
            
            id="your-montage-container",
            children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "center", "alignItems": "center", "gap": "20px", "margin": "30px"},
                        children=[
                            html.H1([
                                "Your Montage ",
                                html.I(className="bi bi-info-circle-fill", id="montage-help-icon", style={
                                    "fontSize": "0.8em",
                                    "cursor": "pointer",
                                    "verticalAlign": "middle"
                                })
                            ], style={"margin": 0}),

                            # Refresh Button
                            dbc.Button(
                                html.I(className="bi bi-arrow-clockwise"),
                                id="refresh-button",
                                color="primary",
                                style={
                                    "marginLeft": "10%",
                                    "fontSize": "1.2em"
                                },
                                title="Refresh table"
                            ),

                            # Delete All Button
                            dbc.Button(
                                html.I(className="bi bi-trash3-fill"),
                                id="delete-all-button",
                                color="danger",
                                style={
                                    "marginLeft": "5px",
                                    "fontSize": "1.2em"
                                },
                                title="Delete all"
                            ),
                        ]
                    ),
                    # Tooltip for the info icon
                    dbc.Tooltip(
                        "Here you can see which montage you have already created.",
                        target="montage-help-icon",
                        placement="right"
                    ),
                dash.dash_table.DataTable(
                    id="saved-montages-table",
                    columns=[
                        {"name": "Montage Name", "id": "montage_name"},
                        {"name": "Channels", "id": "channels"},
                        {"name": "Actions", "id": "actions"},
                    ],
                    data=[],  # To be populated with saved montages
                    style_table={
                        "overflowX": "auto", 
                        "width": "50%",  # Reduce table width
                        "margin": "0 auto"  # Center the table on the page
                        },
                    style_cell={
                        "textAlign": "center",  # Center text in all cells
                        "padding": "5px",  # Add padding for better readability
                        "fontFamily": "Arial, sans-serif",  # Consistent font
                        "fontSize": "14px",  # Adjust font size
                        "border": "1px solid #ddd"  # Add light borders
                    },
                    style_header={
                        "fontWeight": "bold",  # Bold headers
                        "textAlign": "center",  # Center text in headers
                        "backgroundColor": "#f4f4f4",  # Light background for headers
                        "borderBottom": "2px solid #ccc"  # Slightly thicker bottom border
                    },
                    style_data_conditional=[
                        {
                            "if": {"column_id": "channels"},
                            "whiteSpace": "nowrap",  # Prevent text wrapping
                            "overflow": "auto",  # Allow scrolling
                            "maxWidth": "150px",  # Adjust width as needed
                        }
                    ]
                )
            ],  # This closes the `children` list
            style={
                "padding": "15px", 
                "backgroundColor": "#fff", 
                "border": "1px solid #ddd",
                "borderRadius": "8px", 
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", 
                "marginBottom": "20px"
            }
        )
    ]),
    
    html.Div([
        # Left Side: Montage Name and Create Button
        html.Div(
            id="montage-name-container",
            children=[
                html.H3([
                    html.I(className="bi bi-1-circle-fill", style={"marginRight": "10px", "fontSize": "1.2em"}),
                    "Create a new montage"]
                    , style={"margin-bottom": "15px"}),

                # Input for Montage Name
                dbc.Input(
                    id="new-montage-name",
                    type="text",
                    placeholder="Montage name...",
                    style={"marginBottom": "10px", "width": "100%"}
                ),

                # Create Button
                dbc.Button(
                    "Create",
                    id="create-button",
                    color="success",
                    disabled=True,
                    n_clicks=0,
                    style={"marginTop": "5px", "width": "100%"}
                ),
            ],
            style={
                "padding": "15px",
                "backgroundColor": "#fff",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "width": "20%",  # Set width for left panel
                "marginRight": "20px"  # Add spacing between elements
            }
        ),

        # Right Side: Montage Selection
        html.Div(
            id="montage-selection-container",
            children=[
                html.H3([
                    html.I(className="bi bi-2-circle-fill", style={"marginRight": "10px", "fontSize": "1.2em"}),  # Example icon
                    "Select channels"
                ], style={"margin-bottom": "15px"}),

                # Dropdown for Selection Method
                dcc.Dropdown(
                    id="selection-method-dropdown",
                    options=[
                        {"label": "Checklist", "value": "checklist"},
                        {"label": "Random Pick", "value": "random"}
                    ],
                    value="checklist",  # Default value
                    clearable=False,
                    style={"width": "100%", "marginBottom": "20px"}
                ),

                html.Div(
                    id="checklist-method-container",
                    children=[
                        *[
                            html.Div(
                                [
                                    html.H3(group),  # Smaller font size for the group name
                                    dbc.Button(
                                        "Check All",
                                        id=f"check-all-channels-to-pick-btn-{group}",
                                        color="success",
                                        outline=True,
                                        size="sm",
                                        n_clicks=0,
                                        style={
                                            "fontSize": "10px",
                                            "padding": "6px 12px",
                                            "borderRadius": "5px",
                                            "width": "100%",  # Adjusted width
                                        }
                                    ),
                                    dbc.Button(
                                        "Clear All",
                                        id=f"clear-all-channels-to-pick-btn-{group}",
                                        color="warning",
                                        outline=True,
                                        size="sm",
                                        n_clicks=0,
                                        style={
                                            "fontSize": "10px",
                                            "padding": "6px 12px",
                                            "borderRadius": "5px",
                                            "width": "100%",  # Adjusted width
                                        }
                                    ),
                                    dcc.Checklist(
                                        id=f"montage-checklist-{group}",
                                        options=[{"label": ch, "value": ch} for ch in channels],
                                        value=[],
                                        style={"marginTop": "10px", "fontSize": "10px"}  # Smaller font size for checklist items
                                    )
                                ],
                                style={"flex": "1 0 6%", "padding": "5px"}
                            )
                            for idx, (group, channels) in enumerate(c.GROUP_CHANNELS_BY_REGION.items())
                        ],
                    ],
                    style={"display": "none"}
                ),

                # Random Pick Method Container (initially hidden)
                html.Div(
                    id="random-pick-method-container",
                    children=[
                        *[
                            html.Div(
                                [
                                    html.H3(group),
                                    dbc.Input(
                                        id=f"random-pick-count-{group}",
                                        type="number",
                                        min=0,
                                        step=1,
                                        value=0,
                                        max=len(channels),
                                        style={"width": "100%"}
                                    )
                                ],
                                style={"flex": "1 0 6%", "padding": "5px"}
                            )
                            for group, channels in c.GROUP_CHANNELS_BY_REGION.items()
                        ],
                    ], style={"display": "none"},
                ),
            ],
            style={
                "padding": "15px",
                "backgroundColor": "#fff",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "width": "60°%",  # Set width for right panel
                "display": "none"
            }
        ), 

        html.Div(
            id="channels-layout-container",
            children=[
                html.H3([
                    html.I(className="bi bi-3-circle-fill", style={"marginRight": "10px", "fontSize": "1.2em"}),  # Example icon
                    "Save montage"
                ], style={"margin-bottom": "15px"}),

                html.Div(id="channels-layout-display",
                         children=html.Img(id="channels-layout-img", style={"width": "100%"})),

                                # Create Button
                dbc.Button(
                    "Save",
                    id="save-button-ica",
                    color="success",
                    disabled=False,
                    n_clicks=0,
                    style={"marginTop": "5px", "width": "100%"}
                ),
            ],
            style={
                "padding": "15px",
                "backgroundColor": "#fff",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "width": "20%",  # Set width for left panel
                "marginRight": "20px",  # Add spacing between elements
                "display":"none"
            }
        ),
    ], style={
        "display": "flex",  # Initially hidden
        "flexDirection": "row",  # Side-by-side layout
        "alignItems": "flex-start",  # Align to top
        "gap": "20px",  # Add spacing between elements
        "width": "100%"  # Ensure full width
    })
])

@callback(
    Output("saved-montages-table", "data"),  # Update the DataTable with new montage data
    Input("refresh-button", "n_clicks"),  # Triggered by clicking the refresh button
    Input("montage-store", "data"),  # Get the data from the montage-store component
    prevent_initial_call=False
)
def display_montage(n_clicks_refresh, montage_store_data):
    """Update the DataTable with a new montage when the 'Save' button is clicked"""

    if not montage_store_data:
        return None  # Do nothing if the montage store is empty
    
    # Prepare the data for the DataTable
    saved_montages = []

    # Iterate through the montage store data to prepare rows for the table
    for montage_name, channels in montage_store_data.items():
        saved_montages.append({
            "montage_name": montage_name,
            "channels": ", ".join(channels),  # Join selected channels into a single string
            "actions": '🗑️'

        })

    # Return the data for the DataTable
    return saved_montages

@callback(
    Output("montage-store", "data"),
    Input("saved-montages-table", "active_cell"),  # Trigger on click on the table cell
    State("saved-montages-table", "data"),
    State("montage-store", "data"),  # Get the current montage store data
    prevent_initial_call=True
)
def delete_montage(active_cell, montages_tab, montage_store_data):
    print(active_cell)
    if active_cell:
        # Get the row index and column index of the clicked cell
        row_index = active_cell['row']

        # Get the name of the montage in the clicked row
        montage_to_delete = montages_tab[row_index]
        print(montage_to_delete)

        montage_name_to_delete = montage_to_delete["montage_name"]

        # Remove the montage from the montage store data
        montage_store_data.pop(montage_name_to_delete)
        
        return montage_store_data
    return dash.no_update

@callback(
    Output("montage-store", "data", allow_duplicate=True),
    Output("saved-montages-table", "data", allow_duplicate=True),
    Input("delete-all-button", "n_clicks"), # Trigger on click on the table cell
    prevent_initial_call=True
)
def delete_all_montage(n_clicks):
    if n_clicks and n_clicks>0:
        return {}, []
    return dash.no_update, dash.no_update

@callback(
    Output("create-button", "disabled"),
    Input("new-montage-name", "value"),
    State("montage-store", "data"),
    prevent_initial_call=True
)
def handle_valid_montage_name(name, montage_store_data):
    """Validate montage name"""
    if name:
        if name in montage_store_data:
            return True
        # Check if folder exists and finish by .ds, then make "load" button clickable
        return False
    return True

@callback(
    Output("montage-selection-container", "style"),
    Output("channels-layout-container", "style"),
    Output("new-montage-name", "disabled"),
    Input("create-button", "n_clicks"),
    State("new-montage-name", "value"),
    State("montage-store", "data"),
    prevent_initial_call=True
)
def handle_create_button(n_clicks, new_montage_name, montage_store_data):
    # Check if the name already exists in the montage store
    if n_clicks > 0:
        return (
            {"padding": "15px", "backgroundColor": "#fff", "border": "1px solid #ddd", "borderRadius": "8px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "width": "60%"},
            {"padding": "15px", "backgroundColor": "#fff", "border": "1px solid #ddd", "borderRadius": "8px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "width": "20%"},
            True
        )

    
@callback(
    Output("checklist-method-container", "style"),
    Output("random-pick-method-container", "style"),
    Input("selection-method-dropdown", "value"),
    prevent_initial_call=False
)
def update_selection_method_ui(method):

    if method == "checklist":
        return {"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between"}, {"display": "none"}
    
    elif method == "random":
        return {"display": "none"}, {"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between"}

    return dash.no_update, dash.no_update  # fallback

@callback(
    Output("montage-store", "data", allow_duplicate=True),
    Output("url", "href"),
    # Output("edit-montage", "style", allow_duplicate=True),
    Input("save-button-ica", "n_clicks"),
    State("new-montage-name", "value"),
    State("selection-method-dropdown", "value"),
    State("montage-store", "data"),
    # Flat list of checklist values followed by random pick values
    *[
        State(f"montage-checklist-{group}", "value")
        for group in c.GROUP_CHANNELS_BY_REGION_PREFIX
    ],
    *[
        State(f"random-pick-count-{group}", "value")
        for group in c.GROUP_CHANNELS_BY_REGION_PREFIX
    ],
    prevent_initial_call=True
)
def update_montage_store(n_clicks, new_montage_name, selection_method, montage_store_data, *values):
    if n_clicks <= 0:
        return dash.no_update, dash.no_update

    if not new_montage_name:
        return dash.no_update, dash.no_update

    if not montage_store_data:
        montage_store_data = {}

    num_groups = len(c.GROUP_CHANNELS_BY_REGION_PREFIX)
    checked_values = values[:num_groups]
    pick_values = values[num_groups:]

    selected_channels = []

    if selection_method == "checklist":
        for group_selected in checked_values:
            if group_selected:
                selected_channels.extend(group_selected)

    elif selection_method == "random":
        for i, group in enumerate(c.GROUP_CHANNELS_BY_REGION_PREFIX):
            pick_count = pick_values[i] or 0
            try:
                pick_count = int(pick_count)
            except ValueError:
                pick_count = 0

            available_channels = c.GROUP_CHANNELS_BY_REGION[group]
            if pick_count > 0:
                import random
                selected_channels.extend(random.sample(available_channels, min(pick_count, len(available_channels))))

    if not selected_channels:
        return dash.no_update, dash.no_update

    # Save to store
    montage_store_data[new_montage_name] = selected_channels

    return montage_store_data, "/settings/montage"

@callback(
    Output("channels-layout-img", "src"),
    *[
        Input(f"montage-checklist-{group}", "value")
        for group in c.GROUP_CHANNELS_BY_REGION_PREFIX
    ],
    *[
        Input(f"random-pick-count-{group}", "value")
        for group in c.GROUP_CHANNELS_BY_REGION_PREFIX
    ],
    State("folder-store", "data"),
    State("selection-method-dropdown", "value"),
    prevent_initial_call=True
)
def update_meg_layout(*args):
    num_groups = len(c.GROUP_CHANNELS_BY_REGION_PREFIX)
    checked_values = args[:num_groups]
    pick_values = args[num_groups:2*num_groups]
    folder_path = args[-2]
    selection_method = args[-1]

    selected_channels_by_group = [[] for _ in c.GROUP_CHANNELS_BY_REGION_PREFIX]

    if selection_method == "checklist":
        for i, group_selected in enumerate(checked_values):
            if group_selected:
                selected_channels_by_group[i].extend(group_selected)

    elif selection_method == "random":
        for i, group in enumerate(c.GROUP_CHANNELS_BY_REGION_PREFIX):
            pick_count = pick_values[i] or 0
            try:
                pick_count = int(pick_count)
            except ValueError:
                pick_count = 0

            available_channels = c.GROUP_CHANNELS_BY_REGION[group]
            if pick_count > 0:
                selected = random.sample(available_channels, min(pick_count, len(available_channels)))
                selected_channels_by_group[i].extend(selected)

    # Step 1: Load info from file
    raw = mne.io.read_raw_ctf(folder_path, preload=False, verbose=False)
    info = raw.info

    highlighted = [
        mne.pick_channels(info.ch_names, group)
        for group in selected_channels_by_group if group
    ]

    fig = mne.viz.plot_sensors(
        info,
        kind="topomap",
        ch_groups=highlighted,
        show_names=False,
        pointsize=100,
        linewidth=0.5,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


@callback(
    # Outputs: Reset all checklist values to empty
    [
        Output(f"montage-checklist-{group}", "value", allow_duplicate=True) for group in c.GROUP_CHANNELS_BY_REGION_PREFIX
    ],
    # Inputs: Detecting "Clear All" button clicks
    [
        Input(f"clear-all-channels-to-pick-btn-{group}", "n_clicks") for group in c.GROUP_CHANNELS_BY_REGION_PREFIX
    ],
    prevent_initial_call=True
)
def clear_all_checklists(*n_clicks):
    # Return an empty list for each group when any "Clear All" button is clicked
    return [[] for _ in n_clicks]

# Iterate over each group in GROUP_CHANNELS_BY_REGION_PREFIX

def register_check_all_channels_by_group(group):
    @callback(
        # Outputs: Set all checklists' value to the full list of channels for the specific group
        Output(f"montage-checklist-{group}", "value"),
        # Input: Detect "Check All" button click for the specific group
        Input(f"check-all-channels-to-pick-btn-{group}", "n_clicks"),
        prevent_initial_call=True
    )
    def check_all_channels(n_click):
        # Only perform action if button is clicked
        if n_click and n_click > 0:
            # Get the group name directly from the group being processed in the loop
            print(c.GROUP_CHANNELS_BY_REGION[group])
            return c.GROUP_CHANNELS_BY_REGION[group]  # Return the full list of channels for that group
        return dash.no_update
    
for group in c.GROUP_CHANNELS_BY_REGION_PREFIX:
    register_check_all_channels_by_group(group)

def register_clean_all_channels_by_group(group):
    @callback(
        # Outputs: Clear the checklist for the specific group
        Output(f"montage-checklist-{group}", "value", allow_duplicate=True),
        # Input: Detect "Clean All" button click for the specific group
        Input(f"clear-all-channels-to-pick-btn-{group}", "n_clicks"),
        prevent_initial_call=True
    )
    def clean_all_channels(n_click):
        # Only perform action if button is clicked
        if n_click and n_click > 0:
            # Return an empty list to clear the channels selected for this group
            return []
        return dash.no_update
    
for group in c.GROUP_CHANNELS_BY_REGION_PREFIX:
    register_clean_all_channels_by_group(group)


