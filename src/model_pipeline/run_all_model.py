from model_pipeline.utils import save_data_matrices, create_windows, generate_database
import sys
import params
from pathlib import Path
import ast

def run_model_pipeline(
		model_name, 
		model_type,
		subject,
		output_path,
		threshold,
		adjust_onset,
		channel_groups):
	  
	# Model Selection
	if "TensorFlow" in model_type:
			from model_pipeline.tensorflow_models import test_model_dash
			window_size = params.window_size_ms
	elif 'PyTorch' in model_type:
			from model_pipeline.pytorch_models import test_model_dash
			window_size = params.window_size_ms
			
	# Data Preparation
	save_data_matrices(subject, output_path, channel_groups)
	total_nb_windows = create_windows(output_path, window_size)
	X_test_ids = generate_database(total_nb_windows)

	# Model Testing
	return test_model_dash(model_name, X_test_ids, output_path, threshold, adjust_onset, subject)

if __name__ == "__main__":
	model_path = sys.argv[1]
	model_type = sys.argv[2]
	subject_folder_path = sys.argv[3]
	results_path = sys.argv[4]
	threshold = float(sys.argv[5])  # Convert back to float
	adjust_onset = sys.argv[6]
	channel_groups = ast.literal_eval(sys.argv[7])

	threshold = 0

	data_dir = Path("/home/admin_mel/Code/DeepEpiX/data/testData")
	for subject_folder_path in data_dir.rglob("*.ds"):
		print(subject_folder_path)
		if 'hz' not in str(subject_folder_path):
			for model_path in ['/home/admin_mel/Code/DeepEpiX/src/models/model_CNN.keras', '/home/admin_mel/Code/DeepEpiX/src/models/model_features_only.keras']:
				print(model_path)
				run_model_pipeline(model_path, model_type, subject_folder_path, results_path, threshold, adjust_onset, channel_groups)