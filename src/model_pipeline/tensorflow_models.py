import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import os.path as op
import model_pipeline.params as params
import pandas as pd
import gc
from model_pipeline.utils import load_obj, compute_window_ppa, compute_window_upslope, compute_window_std, compute_window_average_slope, compute_window_downslope, compute_window_sharpness, compute_gfp, find_peak_gfp


def get_win_data_feat(sample_norm, compute_features = {"ppa" : compute_window_ppa, "std": compute_window_std, "upslope" : compute_window_upslope, "downslope": compute_window_downslope, "average_slope": compute_window_average_slope, "sharpness" : compute_window_sharpness}):

    sample_norm = np.squeeze(sample_norm)
    features_sample = np.empty((1, len(list(compute_features)), sample_norm.shape[1]))
    for feat, func in compute_features.items():
        features_sample[0,list(compute_features).index(feat)] = func(sample_norm)

    return features_sample

def get_win_data_signal(f,win,sub,dim):

    # Store sample 
    f.seek(dim[0]*dim[1]*win*4) #4 because its float32 and dtype.itemsize = 4
    sample = np.fromfile(f, dtype='float32', count=dim[0]*dim[1])
    sample = sample.reshape(dim[1],dim[0])
    sample = np.swapaxes(sample,0,1)
    sample = np.expand_dims(sample,axis=-1)
    sample = np.expand_dims(sample,axis=0)

    mean = np.mean(sample)
    std = np.std(sample)
    sample_norm = (sample - mean)/std

    return sample_norm

def test_model_dash(model_name, X_test_ids, output_path, threshold=0.5, adjust_onset = True, subject = None):

    model = keras.models.load_model(model_name, compile=False)
    model.compile()

    f = open(op.join(output_path, "data_raw_"+str(params.subject_number)+'_windows_bi'))

    y_pred_probas=[]
    adjusted_onsets = []

        # Use GPU if available
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    with tf.device(device):
        for ind in range(0,X_test_ids.shape[0]):

            cur_sub = X_test_ids[ind,1]
            cur_win = X_test_ids[ind,0]

            sample = get_win_data_signal(f,cur_win,cur_sub,params.dim)
            
            if "features" in model_name:
                sample = get_win_data_feat(sample)

            y_pred_probas.append(model(sample).numpy()[0][0])

            del sample

    del model

    gc.collect()
    keras.backend.clear_session()

    # Load timing data
    y_timing_data = load_obj("data_raw_" + str(params.subject_number) + '_timing.pkl', output_path)

    if adjust_onset == "Yes":
        # Compute adjusted onsets based on GFP peaks
        for win in range(0,X_test_ids.shape[0]):
            
            if y_pred_probas[win] > threshold:

                cur_sub = X_test_ids[win,1]
                cur_win = X_test_ids[win,0]

                window = get_win_data_signal(f,cur_win,cur_sub,params.dim).squeeze()

                gfp = compute_gfp(window.T)  # Compute GFP
                times = np.linspace(0, window.shape[0] / params.sfreq, window.shape[0])  # Time vector
                
                peak_time = find_peak_gfp(gfp, times)  # Find max GFP time
                adjusted_onset = ((y_timing_data[win] - window.shape[0]/2) / params.sfreq) + peak_time  # Align event to GFP peak
                adjusted_onsets.append(round(adjusted_onset, 3))
            else:
                adjusted_onsets.append(round(y_timing_data[win]/params.sfreq, 3))

    else:
        # Extract onset times for predicted events (convert to seconds)
        adjusted_onsets = (y_timing_data / params.sfreq).round(3).tolist()

    # Create DataFrame with onsets, duration, and probability scores
    df = pd.DataFrame({
        "onset": adjusted_onsets,
        "duration": 0,  # To fit MNE annotation format
        "probas": y_pred_probas  # Store raw probabilities
    })

    # Save DataFrame as CSV
    df.to_csv(f'{output_path}/{os.path.basename(model_name)}_predictions.csv', index=False)

    ### === Generate final summary row for global evaluation CSV === ###
    
    # Select predicted spikes above threshold
    pred_spike_times = [onset for onset, prob in zip(adjusted_onsets, y_pred_probas) if prob > threshold]
    pred_spike_probs = [round(prob, 3) for prob in y_pred_probas if prob > threshold]

    # Create DataFrame for predictions
    pred_df = pd.DataFrame({
        "Patient": [os.path.basename(subject)] * len(pred_spike_times),
        "Model": [os.path.basename(model_name)] * len(pred_spike_times),
        "SpikeTime_s": pred_spike_times,
        "Probability": pred_spike_probs
    })

    pred_df.to_csv(os.path.join(output_path, "predictions.csv"), mode='a', index=False, header=not os.path.exists(os.path.join(output_path, "predictions.csv")))
