import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path
import mne
from dash import html
import dash_bootstrap_components as dbc
from collections import Counter

# Function to get model options
def get_folder_path_options():
    data_dir = Path.cwd() / "data"  # Proper path handling
    data = list(data_dir.glob("*.ds"))  # List all matching files
    
    return (
        [{"label": d.name, "value": str(d.resolve())} for d in data]
        if data
        else [{"label": "No data available", "value": ""}]
    )


def browse_folder():
    root = tk.Tk()
    root.withdraw()  # Hide root window
    root.attributes("-topmost", True)  # Make sure dialog appears on top
    folder_path = filedialog.askdirectory(title="Select a folder", parent=root)
    root.destroy()  # Destroy the root window after selection
    return folder_path

def test_ds_folder(path):
    parts = path.split(os.sep)  # Split path by OS separator ('/' or '\')
    for part in reversed(parts):  # Iterate from the end
        if part.endswith(".ds"):  # Check if it ends with ".ds"
            return True
    return False

def get_ds_folder(path):
    parts = path.split(os.sep)  # Split path by OS separator ('/' or '\')
    for part in reversed(parts):  # Iterate from the end
        if part.endswith(".ds"):  # Check if it ends with ".ds"
            return part
    return None  # Return None if no matching folder is found

def build_table_raw_info(folder_path):

    raw = mne.io.read_raw_ctf(folder_path, preload=False, verbose=False)
    info = raw.info

    data = [
        {"Property": "File name", "Value": raw.filenames[0] if raw.filenames else "Unknown"},
        {"Property": "Number of channels", "Value": info['nchan']},
        {"Property": "Sampling frequency (Hz)", "Value": info['sfreq']},
        {"Property": "Highpass filter", "Value": info['highpass']},
        {"Property": "Lowpass filter", "Value": info['lowpass']},
        {"Property": "Duration (s)", "Value": round(raw.times[-1], 2)},
        {"Property": "Channel names (preview)", "Value": ', '.join(info['ch_names'][:5]) + "..." if len(info['ch_names']) > 5 else ', '.join(info['ch_names'])},
        {"Property": "Bad channels", "Value": ', '.join(info['bads']) if info['bads'] else "None"},
        {"Property": "Measurement date", "Value": str(info['meas_date'])},
        {"Property": "Experimenter", "Value": info.get('experimenter', 'Unknown')},
        {"Property": "Comps (SSP/ICA)", "Value": f"{len(info.get('comps', []))} components"},
        {"Property": "Projections (SSP)", "Value": f"{len(info.get('projs', []))} projections"},
        {"Property": "Digitized points", "Value": f"{len(info.get('dig', []))} points" if info.get('dig') else "None"},
        {"Property": "CTF Head Transform", "Value": "Available" if info.get('ctf_head_t') else "None"},
        {"Property": "Device to Head Transform", "Value": "Available" if info.get('dev_head_t') else "None"},
    ]

    info_table_header = html.Thead(html.Tr([html.Th("Property"), html.Th("Value")]))
    info_table_body = html.Tbody([
        html.Tr([html.Td(row["Property"]), html.Td(str(row["Value"]))]) for row in data
    ])
    info_table = dbc.Table(
        [info_table_header, info_table_body],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
    )

    return html.Div([info_table])

def build_table_events_statistics(folder_path):

    raw = mne.io.read_raw_ctf(folder_path, preload=False, verbose=False)
    annotations = raw.annotations

    if len(annotations) == 0:
        return html.P("No annotations found in this recording.")

    # Count annotation descriptions
    description_counts = Counter(annotations.description)

    # Build a stats table
    table_header = [html.Thead(html.Tr([html.Th("Event Name"), html.Th("Count")]))]
    table_body = [
        html.Tr([html.Td(desc), html.Td(count)]) for desc, count in description_counts.items()
    ]
    annotation_table = dbc.Table(table_header + [html.Tbody(table_body)], bordered=True, striped=True, hover=True, size="sm")

    # Show total number and a few more stats
    stats_summary = html.Ul([
        html.Li(f"Total annotations: {len(annotations)}"),
        html.Li(f"Unique event types: {len(description_counts)}"),
        html.Li(f"First event starts at {annotations.onset[0]:.2f} s"),
        html.Li(f"Last event ends at {(annotations.onset[-1] + annotations.duration[-1]):.2f} s"),
    ])

    return html.Div([
                annotation_table,
                html.Hr(),
                html.H5("Event Summary"),
                stats_summary,
            ])