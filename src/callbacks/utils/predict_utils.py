from pathlib import Path
from config import MODELS_DIR


def get_model_options():
    model_dir = Path(MODELS_DIR)

    items = list(model_dir.iterdir())
    if items:
        return [{"label": d.name, "value": str(d.resolve())} for d in items]
    else:
        return [{"label": "No data available", "value": ""}]
