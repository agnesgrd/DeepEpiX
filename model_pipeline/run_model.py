from model_pipeline.utils import save_data_matrices, create_windows, generate_database
import sys
import params
import os

def run_model_pipeline(
		model_name, 
		model_type, 
		good_channels_file,
		subject,
		output_path,
		threshold,
		adjust_onset):
	  
	# Model Selection
	if "TensorFlow" in model_type:
			from model_pipeline.tensorflow_models import test_model_dash
			window_size = params.window_size_ms
	elif 'PyTorch' in model_type:
			from model_pipeline.pytorch_models import test_model_dash
			window_size = params.window_size_ms
			
	# Data Preparation
	save_data_matrices(good_channels_file, subject, output_path)
	total_nb_windows = create_windows(output_path, window_size)
	X_test_ids = generate_database(total_nb_windows)

	# Model Testing
	return test_model_dash(model_name, X_test_ids, output_path, threshold, adjust_onset, subject)

if __name__ == "__main__":
	model_path = sys.argv[1]
	model_type = sys.argv[2]
	subject_folder_path = sys.argv[3]
	good_channels_path = sys.argv[4]
	results_path = sys.argv[5]
	threshold = float(sys.argv[6])  # Convert back to float
	adjust_onset = sys.argv[7]

	run_model_pipeline(model_path, model_type, good_channels_path, subject_folder_path, results_path, threshold, adjust_onset)




