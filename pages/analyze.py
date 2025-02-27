# analyze.py: Analyze Page
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from layout import input_styles
import static.constants as c


dash.register_page(__name__)

layout = html.Div([

    html.Div([
    
        html.Div(
            
            id="your-montage-container",
            children=[
                html.H1("Your Montage"),
                # Section to display saved montages
                html.P("Here you can see which montage you have already created.", 
                    style={"marginBottom": "10px"}),
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
                ),
                # Refresh Icon (use a button with an icon)
                dbc.Button(
                    "Refresh table",  # Using Bootstrap Icons
                    id="refresh-button",
                    color="primary",
                    style={"marginLeft": "10px", "marginTop": "20px"}  # Adjust position if needed
                ),
                # Refresh Icon (use a button with an icon)
                dbc.Button(
                    "Delete all",  # Using Bootstrap Icons
                    id="delete-all-button",
                    color="danger",
                    style={"marginLeft": "10px", "marginTop": "20px"}  # Adjust position if needed
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
    
    # Explanation of what the user needs to do
    html.Div([
        html.H2("Edit Montage"),
        html.P("Here you can create your montage, give a name and pick it for visualization."),
        html.Div([
            dbc.Input(
                id="new-montage-name",
                type="text",
                placeholder="Montage name...",
                style=input_styles["path"]
            )
        ], style={"padding": "10px"}),

        dbc.Button(
            "Create",
            id="create-button",
            color="success",
            disabled=True,
            n_clicks=0
        )
    ], style={"padding": "15px", "backgroundColor": "#fff", "border": "1px solid #ddd","borderRadius": "8px", "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "marginBottom": "20px"}),


    # Section for frequency parameters, initially hidden
    html.Div(
        id="edit-montage",
        children=[
            html.H2("Selection Method:"),
            dcc.Dropdown(
                id="selection-method-dropdown",
                options=[
                    {"label": "Checklist", "value": "checklist"},
                    {"label": "Random Pick", "value": "random"}
                ],
                value="checklist",  # Default value
                clearable=False,
                style={"width": "50%", "marginBottom": "20px"}
            ),

            # Container for Checklist or Random Pick UI
            html.Div(
                id="selection-method-container",  # Container to hold checklists
                children=[
                    html.Div(
                        [
                            html.H3(group),  # Group name as header
                            dcc.Checklist(
                                id=f"montage-checklist-{group}",
                                options=[{"label": ch, "value": ch} for ch in channels],
                                value=[],  # Initially, no selection
                                style={"marginTop": "10px"}
                            )
                        ],
                        style={"flex": "1 0 6%", "padding": "5px"}  # Style for column width and spacing
                    )
                    for group, channels in c.GROUP_CHANNELS_BY_REGION.items()  # Loop through groups
                ],
                style={
                    "display": "none",  # Initially hide the checklists
                    "flexWrap": "wrap", 
                    "justifyContent": "space-between"
                }
            ),


            # Button and status display with loading spinner
            html.Div([
                dbc.Button(
                    "Save",
                    id="save-button",
                    color="success",
                    disabled=True,
                    n_clicks=0
                ),
                # Loading spinner wraps only the elements that require loading
                dcc.Loading(
                    id="loading",
                    type="default", 
                    children=[
                        html.Div(id="saving-status", style={"margin-top": "10px"})
                    ]
                )
            ], style={"padding": "10px", "margin-top": "20px"})
        ],
        style={
            "padding": "15px",
            "backgroundColor": "#fff",
            "border": "1px solid #ddd",  # Grey border
            "borderRadius": "8px",  # Rounded corners
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            "marginBottom": "20px",
            "display": "none"
        }
    )
])

@dash.callback(
    Output("saved-montages-table", "data"),  # Update the DataTable with new montage data
    Input("refresh-button", "n_clicks"),  # Triggered by clicking the refresh button
    Input("save-button", "n_clicks"),
    Input("montage-store", "data"),  # Get the data from the montage-store component
    prevent_initial_call=False
)
def display_montage(n_clicks_refresh, n_clicks_save, montage_store_data):
    """Update the DataTable with a new montage when the 'Save' button is clicked"""

    if not montage_store_data:
        return dash.no_update  # Do nothing if the montage store is empty
    
    # Prepare the data for the DataTable
    saved_montages = []

    # Iterate through the montage store data to prepare rows for the table

    for montage_name, channels in montage_store_data.items():
        saved_montages.append({
            "montage_name": montage_name,
            "channels": ", ".join(channels),  # Join selected channels into a single string
            "actions": "DELETE"

        })

    # Return the data for the DataTable
    return saved_montages

@dash.callback(
    Output("montage-store", "data"),
    Input("saved-montages-table", "active_cell"),  # Trigger on click on the table cell
    State("saved-montages-table", "data"),
    State("montage-store", "data"),  # Get the current montage store data
    prevent_initial_call=True
)
def delete_montage(active_cell, montages_tab, montage_store_data):
    if active_cell:
        # Get the row index and column index of the clicked cell
        row_index = active_cell['row']

        # Get the name of the montage in the clicked row
        montage_to_delete = montages_tab[row_index]

        montage_name_to_delete = montage_to_delete["montage_name"]

        # Remove the montage from the montage store data
        montage_store_data.pop(montage_name_to_delete)
        
        return montage_store_data
    return dash.no_update

@dash.callback(
    Output("montage-store", "data", allow_duplicate=True),
    Output("saved-montages-table", "data", allow_duplicate=True),
    Input("delete-all-button", "n_clicks"), # Trigger on click on the table cell
    prevent_initial_call=True
)
def delete_all_montage(n_clicks):
    if n_clicks and n_clicks>0:
        return {}, []
    return dash.no_update, dash.no_update

@dash.callback(
    Output("create-button", "disabled"),
    Input("new-montage-name", "value"),
    prevent_initial_call=True
)
def handle_valid_montage_name(name):
    """Validate montage name"""
    if name:
        # Check if folder exists and finish by .ds, then make "load" button clickable
        return False
    return True

@dash.callback(
    Output("edit-montage", "style"),
    Input("create-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_load_button(n_clicks):
    """Display frequency parameters when button is clicked"""
    if n_clicks > 0:
        return {"padding": "15px",
            "backgroundColor": "#fff",
            "border": "1px solid #ddd",  # Grey border
            "borderRadius": "8px",  # Rounded corners
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            "marginBottom": "20px",
            "display": "block"}
    

@dash.callback(
    Output("selection-method-container", "style"),
    Input("create-button", "n_clicks"),
    State("selection-method-dropdown", "value"),
    prevent_initial_call=True
)
def toggle_checklists_visibility(n_clicks, selection_method):
    """Toggle visibility of checklists based on the selected method."""
    if n_clicks > 0:
        if selection_method == "checklist":
            return {"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between"}  # Show checklists
        elif selection_method == "random":
            return {"display": "none"}  # Hide checklists for "random" selection
        else:
            return {"display": "none"}  # Default to hidden

@dash.callback(
    Output("save-button", "disabled"),
    [Input(f"montage-checklist-{group}", "value") for group in c.GROUP_CHANNELS_BY_REGION_PREFIX],
    prevent_initial_call=True
)
def handle_load_button(*values):
    """Enable save button if at least one checkbox is selected"""
    # Flatten the list of selected values from all checklists
    selected_channels = [channel for group_value in values for channel in group_value]

    # Disable button if no channels are selected
    if len(selected_channels) == 0:
        return True  # Disable button
    else:
        return False  # Enable button
    

@dash.callback(
    Output("saving-status", "children"),
    Output("montage-store", "data", allow_duplicate=True),
    Output("selection-method-container", "style", allow_duplicate=True),
    Output("saved-montages-table", "data", allow_duplicate=True),  # Update the DataTable with new montage data
    Input("save-button", "n_clicks"),
    State("new-montage-name", "value"),
    State("montage-store", "data"),
    [State(f"montage-checklist-{group}", "value") for group in c.GROUP_CHANNELS_BY_REGION_PREFIX],
    prevent_initial_call=True
)
def save_montage(n_clicks, new_montage_name, montage_store_data, *selected_values):
    """Save the montage and update the montage-store with the new montage"""
    
    if n_clicks > 0:

        if not montage_store_data:
            montage_store_data = {}

        # Flatten the selected channels into a single list
        selected_channels = []
        for selected in selected_values:
            if selected:
                selected_channels.extend(selected)
        
        # Add the new montage name and its selected channels to the store
        montage_store_data[new_montage_name] = selected_channels

        # Prepare the data for the DataTable
        saved_montages = []
        for montage_name, channels in montage_store_data.items():
            saved_montages.append({
                "montage_name": montage_name,
                "channels": ", ".join(channels),  # Join selected channels into a single string
                "actions": "DELETE"

            })

        # Return the success message and updated montage store data
        return "Montage saved successfully!", montage_store_data, {"display": "none"}, saved_montages

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update