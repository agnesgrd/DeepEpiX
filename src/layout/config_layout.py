# Common styles

INPUT_STYLES = {
    "path":{
        "width": "100%",
        "padding": "10px",
        "fontSize": "16px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
    },
    "name":{
        "width": "20%",
        "padding": "10px",
        "fontSize": "16px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
    },
    "number-in-box":{
        "width": "30%",
        "padding": "10px",
        "fontSize": "16px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "margin": "10px"
    },
    "number":{
        "width": "15%",
        "padding": "10px",
        "fontSize": "16px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "margin": "10px"
    },
    "small-number":{
        "width": "100%",
        "padding": "8px",
        "fontSize": "12px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "margin": "10px 0"
    }
}

BUTTON_STYLES = {
    "big": {
        "fontSize": "12px",
        "padding": "8px",
        "borderRadius": "5px",
        "width": "100%",
        "margin": "10px 0"
    },
    "tiny": {
        "fontSize": "10px",
        "padding": "6px 12px",
        "borderRadius": "5px",
        "width": "48%"
    }
}

BOX_STYLES = {
    "classic": {
        "padding": "15px",
        "border": "1px solid #ddd",  # Grey border
        "borderRadius": "8px",  # Rounded corners for the channel section
        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
        "marginBottom": "20px"  # Space between the sections
    }
}

FLEXDIRECTION = {
    "row-flex": {
        "flexDirection": "row",  # Side-by-side layout
        "alignItems": "flex-start",  # Align to top
        "gap": "20px",  # Add spacing between elements
        "width": "100%",  # Ensure full width
        "padding": "20px"
    },
    "row-tabs": {
        "display": "flex",
        "flexDirection": "row",  # Ensure tabs are displayed in a row (horizontal)
        "alignItems": "center",  # Center the tabs vertically within the parent container
        "width": "100%",  # Full width of the container
        "borderBottom": "1px solid #ddd"  # Optional, adds a bottom border to separate from content
    }
}

LABEL_STYLES = {
    "classic": {
        "fontWeight": "bold",
        "fontSize": "14px",
        "marginBottom": "5px",
    },
    "info": {
        "fontWeight": "lighter",
        "fontSize": "14px",
        "marginBottom": "5px",
    }
}

ICON = {
        "annotations": "bi-activity",
        "models": "bi-stars",
        "ICA": "bi-noise-reduction"
    }

# Color palette for events
COLOR_PALETTE = [
        "#e6194b",  # strong red
        "#3cb44b",  # vivid green
        "#0082c8",  # vivid blue
        "#f58231",  # bright orange
        "#911eb4",  # strong purple
        "#46f0f0",  # cyan
        "#f032e6",  # magenta (stronger than light pink)
        "#d62728",  # deep red
        "#2ca02c",  # dark green
        "#1f77b4",  # standard matplotlib blue
    ]

DEFAULT_FIG_LAYOUT = dict(
    # autosize=True,
    xaxis=dict(
        title=None,
        minallowed=None,  # Will be overridden dynamically
        maxallowed=None,  # Will be overridden dynamically
        fixedrange=False,
        rangeslider=dict(
            visible=True,
            thickness=0.02
        ),
        showspikes=True,
        spikemode="across+marker",
        spikethickness=1,
    ),
    yaxis=dict(
        title=None,
        showticklabels=False,
        showgrid=False,
        spikethickness=0,
    ),
    title=dict(
        text='',
        x=0.5,
        font=dict(size=12),
        automargin=True,
        yref='paper',
    ),
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
    dragmode='select',
    selectdirection='h',
    hovermode='closest',
    paper_bgcolor='rgba(0,0,0,0)',
    autosize=True
)

# Define the region-to-color mapping
REGION_COLOR_PALETTE = [
    "#1f77b4",  # Muted Blue
    "#ff7f0e",  # Safety Orange
    "#2ca02c",  # Cooked Asparagus Green
    "#d62728",  # Brick Red
    "#9467bd",  # Muted Purple
    "#8c564b",  # Chestnut Brown
    "#e377c2",  # Raspberry Yogurt Pink
    "#7f7f7f",  # Middle Gray
    "#bcbd22",  # Curry Yellow-Green
    "#17becf",  # Blue-Teal
    "#8B0000",  # DarkRed
    "#00008B",  # DarkBlue
    "#006400",  # DarkGreen
    "#FF8C00",  # DarkOrange
]

ERROR = {       
    "position": "absolute",
    "top": "20%",  # or adjust as needed
    "left": "50%",
    "transform": "translateX(-50%)",
    "zIndex": 9999,
    "fontWeight": "bold",
    "backgroundColor": "rgba(255,255,255,0.8)",
    "padding": "5px 10px",
    "borderRadius": "5px",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.2)"
}