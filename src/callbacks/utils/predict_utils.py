from pathlib import Path

# Function to get model options
def get_model_options(model_type):
    model_dir = Path("models")

    if model_type == "AE":
        models = [f for f in model_dir.iterdir() if f.suffix in {".pth", ".keras", ".h5"} and ('AE' in str(f)) is True]
    
    elif model_type == "CNN":
        models = [f for f in model_dir.iterdir() if f.suffix in {".pth", ".keras", ".h5"} and ('AE' in str(f)) is False]
    
    elif model_type == "all":
        models = [f for f in model_dir.iterdir() if f.suffix in {".pth", ".keras", ".h5"}]
    
    return (
        [{"label": d.name, "value": str(d.resolve())} for d in models]
        if models
        else [{"label": "No data available", "value": ""}]
    )
    