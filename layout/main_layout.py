# Common styles

input_styles = {
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
        "width": "50%",
        "padding": "10px",
        "fontSize": "16px",
        "borderWidth": "1px",
        "borderStyle": "solid",
        "borderRadius": "5px",
        "margin": "10px"
    },
    "number":{
        "width": "10%",
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

button_styles = {
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

box_styles = {
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

flexDirection = {
    "row-flex": {
        "flexDirection": "row",  # Side-by-side layout
        "alignItems": "flex-start",  # Align to top
        "gap": "20px",  # Add spacing between elements
        "width": "100%"  # Ensure full width
    },
    "row-tabs": {
        "display": "flex",
        "flexDirection": "row",  # Ensure tabs are displayed in a row (horizontal)
        "alignItems": "center",  # Center the tabs vertically within the parent container
        "width": "100%",  # Full width of the container
        "borderBottom": "1px solid #ddd"  # Optional, adds a bottom border to separate from content
    }
}

label_styles = {
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

icon = {
        "annotations": "bi-activity",
        "models": "bi-stars",
        "ICA": "bi-noise-reduction"
    }

# Color palette for events
color_palette = [
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