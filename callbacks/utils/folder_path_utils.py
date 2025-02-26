import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path

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



