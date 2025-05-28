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
    "panel-tabs": {
        "padding": "15px",  # More spacious padding
        "text-decoration": "none", 
        "font-size": "18px", 
        "color": "white", 
        "border-radius": "12px",  # Rounded corners for a modern feel
        #"margin": "10px",  # Increased margin for better separation
        "display": "inline-block",
        "box-shadow": "0px 4px 12px rgba(0, 0, 0, 0.15)",  # Deeper shadow for depth
        "transition": "all 0.3s ease",  # Smooth transition for hover
        "background-color": "#6c757d",  # Soft gray background
        "cursor": "pointer",  # Pointer cursor to indicate interactivity
    },
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
    autosize=True,
    xaxis=dict(
        title=None,
        minallowed=None,  # Will be overridden dynamically
        maxallowed=None,  # Will be overridden dynamically
        fixedrange=False,
        rangeslider=dict(
            visible=True,
            thickness=0.02,
            bgcolor='rgba(128, 128, 128, 0.5)',
            bordercolor='rgba(64, 64, 64, 1)'
        ),
        showspikes=True,
        spikemode="across+marker",
        spikethickness=1,
    ),
    yaxis=dict(
        title=None,
        showticklabels=False,
        autorange=True,
        showgrid=True,
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
    template="plotly_dark",
    font=dict(color='white'),
    paper_bgcolor='rgba(0,0,0,1)',
    plot_bgcolor='rgba(0,0,0,1)',
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