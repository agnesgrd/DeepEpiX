from pathlib import Path
from itertools import chain


# Function to get model options
def get_model_options():
    model_dir = Path.cwd() / "models"

    models = [f for f in model_dir.iterdir() if f.suffix in {".pth", ".keras", ".h5"}]

    return (
        [{"label": d.name, "value": str(d.resolve())} for d in models]
        if models
        else [{"label": "No data available", "value": ""}]
    )
    