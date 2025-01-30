import os

# Function to get model options
def get_model_options():
    model_dir = "/home/admin_mel/Code/DeepEpiX/models/"
    models = [f for f in os.listdir(model_dir) if f.endswith(('.pth', '.keras', '.h5'))]
    
    return [{"label": model, "value": os.path.join(model_dir, model)} for model in models] if models else [{"label": "No model available", "value": ""}]

