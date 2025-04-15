from dash import html, dcc
import dash_bootstrap_components as dbc
import static.constants as c
from layout import input_styles, box_styles, button_styles
from layout.selection_sidebar_layout import create_selection

def create_compute():
    return html.Div([

        html.Div([
            # Label and input field for timepoint entry
            html.Label(
                "ICA Parameters",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),
            dbc.Input(
                id="n-components",  # Unique ID for each input
                type="number",
                placeholder="N components ...",
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Tooltip(
                """
                n_components : int | float | None\n

                Number of principal components (from the pre-whitening PCA step) 
                that are passed to the ICA algorithm during fitting.

                - int: Must be >1 and ≤ number of channels.
                - float (0 < x < 1): Selects smallest number of components needed 
                to exceed this cumulative variance threshold.
                - None: Uses 0.999999 for numerical stability with rank-deficient data.

                The actual number used is stored in n_components_.
                """,
                target="n-components",
                placement="right",
                class_name="custom-tooltip"
            ),

            dcc.Dropdown(
                id="ica-method",
                options=[
                    {"label": "fastica", "value": "fastica"},
                    {"label": "infomax", "value": "infomax"},
                    {"label": "picard", "value": "picard"},
                ],
                placeholder="Select ICA method...",
                persistence=True,
                persistence_type="local",
            ),

            dbc.Tooltip(
                """
                ICA Method:

                - fastica: A FastICA algorithm from sklearn.
                - infomax: Infomax ICA from MNE (uses CUDA if available).
                - picard: Picard algorithm, optimized for speed and stability.

                Choose depending on speed vs. accuracy needs.

                - Use the `fit_params` argument to set additional parameters.
                - Specifically, if you want Extended Infomax, set `method='infomax'` and `fit_params=dict(extended=True)`
                (this also works for method='picard').
                - Defaults to 'fastica'. 
                """,
                target="ica-method",
                placement="right",
                class_name="custom-tooltip",
            ),

            dbc.Input(
                id="max-iter",  # Unique ID for each input
                type="number",
                placeholder="Max iterations for convergence (s) ...",
                step=1,
                min=1,
                max=2000,
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),
            dbc.Tooltip(
                """
                Maximum number of iterations during ICA fitting.

                - If 'auto':
                • Sets max_iter = 1000 for 'fastica'
                • Sets max_iter = 500 for 'infomax' or 'picard'
                - The actual number of iterations used will be stored in `n_iter_`.

                Increase this value if ICA does not converge.
                """,
                target="max-iter",
                placement="right",
                class_name="custom-tooltip",
            ),

            dbc.Input(
                id="decim",  # Unique ID for each input
                type="number",
                placeholder="Temporal decimation (s) ...",
                step=1,
                min=1,
                max=50,
                size="sm",
                persistence=True,
                persistence_type="local",
                style={**input_styles["small-number"]}
            ),

            dbc.Tooltip(
                """
                Decimation factor used when fitting ICA.

                - Reduces the number of time points by selecting every N-th sample.
                - Helps speed up ICA computation, especially on long recordings.
                - Must be ≥ 1 (1 = no decimation).
                - Typical values: 1 (no decimation), 2, 5, 10.

                Use higher values for faster computation at the cost of temporal precision.
                """,
                target="decim",
                placement="right",
                class_name="custom-tooltip",
            ),
                    
            html.Div([
                dbc.Button(
                    "Compute",
                    id="compute-ica-button",
                    color="warning",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                    disabled=False,
                    style=button_styles["big"]
                ),
            ]),

        ], style=box_styles["classic"]),

        html.Div([
            # Label and input field for timepoint entry
            html.Label(
                "ICA Results",
                style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "8px"}
            ),

            dcc.Loading(
                id="ica-loading",
                type="dot",  # You can use "circle", "dot", etc. for different spinner styles
                children=html.Div(
                    id="ica-results",  # Dynamic log area
                    style={
                        "height": "150px",  # Adjust the height as needed
                        "overflowY": "auto",  # Scrollable if content exceeds height
                        "border": "1px solid #ccc",  # Light border for clarity
                        "borderRadius": "5px",
                        "padding": "5px",
                        "backgroundColor": "#f9f9f9",  # Light background
                        "fontSize": "12px",
                    }
                )
            ),

        ], style=box_styles["classic"]),
    ])

# Helper function to create the sidebar with checkboxes
def create_sidebar():
    return html.Div([
        dbc.Tabs(
                [
                dbc.Tab(create_compute(), label='Compute', tab_id='ica-compute-tab'),
                dbc.Tab(create_selection(
                    montage_radio_id="montage-radio-ica", 
                    check_all_button_id="check-all-channels-btn-ica", 
                    clear_all_button_id="clear-all-channels-btn-ica",
                    channel_region_checkboxes_id="channel-region-checkboxes-ica", 
                    check_all_annotations_btn_id="check-all-annotations-btn-ica", 
                    clear_all_annotations_btn_id="clear-all-annotations-btn-ica", 
                    delete_annotations_btn_id="delete-annotations-btn-ica",
                    annotation_checkboxes_id="annotation-checkboxes-ica",
                    delete_confirmation_modal_id="delete-confirmation-modal-ica",
                    cancel_delete_btn_id="cancel-delete-btn-ica",
                    confirm_delete_btn_id="confirm-delete-btn-ica",
                    offset_decrement_id="offset-decrement-ica", 
                    offset_display_id="offset-display-ica", 
                    offset_increment_id="offset-increment-ica", 
                    colors_radio_id="colors-radio-ica"
                ), label='Select', tab_id='ica-selection-tab'),
            ],
            id="ica-sidebar-tabs",
            persistence = True,
            persistence_type = "local"
        ),
    ], style={
        "padding": "0 20px",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "flex-start",  # Align content at the top
        "gap": "20px",  # Space between elements
        "width": "250px",  # Sidebar width is now fixed
        "boxSizing": "border-box",
        "fontSize": "12px",
        # "backgroundColor": "#f9f9f9",  # Light background color for the sidebar
        "borderRadius": "10px",  # Rounded corners for the sidebar itself
        # "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",  # Subtle shadow for the whole sidebar
        "overflowY": "auto",  # Enable scrolling if content exceeds height
    })