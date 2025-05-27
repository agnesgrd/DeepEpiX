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
    data_dir = Path("data")
    data = list(data_dir.glob("*.ds")) + list(data_dir.glob("*.fif"))

    # Dossiers 4D Neuroimaging
    folders = []
    for folder in data_dir.iterdir():
        if folder.is_dir():
            files = list(folder.glob("*"))
            if any(f.name == 'hs_file' for f in files):
                folders.append(folder)

    all_data = data + folders

    return (
        [{"label": d.name, "value": str(d.resolve())} for d in all_data]
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
        if part.endswith((".ds", ".fif")):  # Check if it ends with ".ds"
            return True
    
    p = Path(path)
    if p.is_dir():
        files = list(p.glob("*"))
        if any(f.name == 'hs_file' for f in files):
            return True

    return False

def get_ds_folder(path):
    parts = path.split(os.sep)  # Split path by OS separator ('/' or '\')
    for part in reversed(parts):  # Iterate from the end
        if part.endswith((".ds", ".fif")):  # Check if it ends with ".ds"
            return part
    return None  # If no matching folder is found

def get_bad_channels(raw, new_bad_channels):
    bad_channels = raw.info.get("bads", [])
    all_bad_channels = []
    if new_bad_channels:
        if isinstance(new_bad_channels, str):
            new_bad_channels_list=[ch.strip() for ch in new_bad_channels.split(",") if ch.strip()]
        else:
            new_bad_channels_list=list(new_bad_channels)  # if it's already a list (e.g., from a previous state)
        all_bad_channels = list(set(bad_channels + new_bad_channels_list))
    return all_bad_channels

def read_raw(folder_path, preload, verbose, bad_channels=None):
    folder_path = Path(folder_path)

    if folder_path.suffix == ".ds":
        raw = mne.io.read_raw_ctf(str(folder_path), preload=preload, verbose=verbose)


    elif folder_path.suffix == ".fif":
        raw = mne.io.read_raw_fif(str(folder_path), preload=preload, verbose=verbose)

    
    elif folder_path.is_dir():
        # Assume BTi/4D format: folder must contain 3 specific files
        files = list(folder_path.glob("*"))
        # Try to identify the correct files by names
        raw_fname = next((f for f in files if "rfDC" in f.name and f.suffix==""), None)
        config_fname = next((f for f in files if "config" in f.name.lower()), None)
        hs_fname = next((f for f in files if "hs" in f.name.lower()), None)

        if not all([raw_fname, config_fname, hs_fname]):
            raise ValueError("Could not identify raw, config, or hs file in BTi folder.")

        raw = mne.io.read_raw_bti(
            pdf_fname=str(raw_fname),
            config_fname=str(config_fname),
            head_shape_fname=str(hs_fname),
            preload=preload,
            verbose=verbose,
        )
    
    else:
        raise ValueError("Unrecognized file or folder type for MEG data.")

    if bad_channels:
        raw.drop_channels(bad_channels)
    
    return raw


def build_table_raw_info(folder_path):

    raw = read_raw(folder_path, preload=False, verbose=False)
    info = raw.info

    table = dbc.Card(
        dbc.CardBody([
            html.H5([
                html.I(className="bi bi-clipboard-data", style={"marginRight": "10px", "fontSize": "1.2em"}),
                "Raw Data Overview"
            ], className="card-title"),
            html.Hr(),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("File: "),
                    html.Span(f"{raw.filenames[0] if raw.filenames else 'Unknown'}")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Number of Channels: "),
                    html.Span(f"{info['nchan']}")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Sampling Frequency: "),
                    html.Span(f"{info['sfreq']} Hz")
                ]),
                dbc.ListGroupItem([
                    html.Strong("High-pass Filter: "),
                    html.Span(f"{info['highpass']} Hz")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Low-pass Filter: "),
                    html.Span(f"{info['lowpass']} Hz")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Duration: "),
                    html.Span(f"{round(raw.times[-1], 2)} seconds")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Channel Names: "),
                    html.Span('...' + f"{', '.join(info['ch_names'][30:35]) + '...' if len(info['ch_names']) > 35 else ', '.join(info['ch_names'])}")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Bad Channels: "),
                    html.Span(f"{', '.join(info['bads']) if info['bads'] else 'None'}")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Measurement Date: "),
                    html.Span(f"{str(info['meas_date'])}")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Experimenter: "),
                    html.Span(f"{info.get('experimenter', 'Unknown')}")
                ]),
                dbc.ListGroupItem([
                    html.Strong("SSP/ICA Components: "),
                    html.Span(f"{len(info.get('comps', []))} components")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Projections (SSP): "),
                    html.Span(f"{len(info.get('projs', []))} projections")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Digitized Points: "),
                    html.Span(f"{len(info.get('dig', []))} points" if info.get('dig') else "None")
                ]),
                dbc.ListGroupItem([
                    html.Strong("CTF Head Transform: "),
                    html.Span("Available" if info.get('ctf_head_t') else "None")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Device to Head Transform: "),
                    html.Span("Available" if info.get('dev_head_t') else "None")
                ])
            ])
        ])
    )

    return table

def build_table_events_statistics(folder_path):

    raw = read_raw(folder_path, preload=False, verbose=False)
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

    stats_summary = dbc.Card(
        dbc.CardBody([
            html.H5([
                html.I(className="bi bi-bar-chart-line", style={"marginRight": "10px", "fontSize": "1.2em"}),
                "Event Summary"
            ], className="card-title"),
            html.Hr(),
            dbc.ListGroup([
                dbc.ListGroupItem(f"Total annotations: {len(annotations)}"),
                dbc.ListGroupItem(f"Unique event types: {len(description_counts)}"),
                annotation_table,
                dbc.ListGroupItem(f"First event starts at {annotations.onset[0]:.2f} s"),
                dbc.ListGroupItem(f"Last event ends at {(annotations.onset[-1] + annotations.duration[-1]):.2f} s"),
            ])
        ])
    )

    return stats_summary