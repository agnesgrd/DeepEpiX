import model_pipeline.params as params
from model_pipeline.utils import save_data_matrices, create_windows, generate_database

def run_model_pipeline(
        model_name, 
        model_type, 
        good_channels_file,
        subject,
        output_path,
        threshold):
    
    if "features" in model_name:
        tf_model = "features"
    else:
        tf_model = None

    # Model Selection
    if model_type == "TensorFlow":
        if tf_model == "features":
            from model_pipeline.tensorflow_models import test_model_dash, load_generators_memeff_feat_only
        else:
            from model_pipeline.tensorflow_models import test_model_dash, load_generators_memeff
    elif model_type == 'PyTorch':
        from model_pipeline.pytorch_models import test_model, load_generators_memeff

    # Data Preparation
    save_data_matrices(good_channels_file, subject, output_path)
    total_nb_windows = create_windows(output_path)
    X_test_ids = generate_database(total_nb_windows)

    # Data Generator
    if tf_model == "features":
        testing_generator = load_generators_memeff_feat_only(X_test_ids, output_path)
    else:
        testing_generator = load_generators_memeff(X_test_ids, output_path)

    # Model Testing
    return test_model_dash(model_name, testing_generator, X_test_ids, output_path, threshold)


