# view.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import mne  # Import the MNE package
import io
import base64
import matplotlib.pyplot as plt

dash.register_page(__name__)

layout = html.Div([
    html.H1("VIEW: Visualize and Annotate MEG Signal"),

    # Display folder path
    html.Div(id="display-folder-path", style={"padding": "10px", "font-style": "italic", "color": "#555"}),

    # Placeholder for the MEG signal plot
    html.Div(id="meg-signal-plot"),
])

# Callback to display the stored folder path
@dash.callback(
    Output("display-folder-path", "children"),
    Input("folder-store", "data"),  # Access the stored data
    prevent_initial_call=False
)
def display_folder_path(folder_path):
    """Display the stored folder path."""
    if folder_path:
        return f"The selected folder path is: {folder_path}"
    return "No folder path has been selected."

# Callback to display MEG plot
@dash.callback(
    Output("meg-signal-plot", "children"),
    Input("folder-store", "data"),  # Access the stored data
    prevent_initial_call=False
)
def display_meg_data(folder_path):
    """Display the stored folder path and MEG signal plot."""
    if not folder_path:
        return None, None

    # Attempt to load the MEG data
    try:
        # Read raw MEG data
        raw = mne.io.read_raw_ctf(str(folder_path), preload=True, verbose=False)

        # Create a plot using MNE
        fig = raw.plot(show=False)  # Create PSD plot (Power Spectral Density)

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format="png")  # Save the figure as a PNG
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode("utf-8")  # Encode the image to base64
        buf.close()

        # Return the folder path and the image
        image_html = html.Img(src=f"data:image/png;base64,{encoded_image}", style={"width": "100%"})
        return image_html

    except Exception as e:
        # Handle errors gracefully
        return f"Error loading MEG data from {folder_path}: {str(e)}", html.Div("Could not display data.")
