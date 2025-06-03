import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import model_pipeline.utils as utils
from scipy import signal
import math
import pickle
import sys
import pandas as pd
import params

##################### FUNCTIONS

# Computes the SmoothGrad input (repeat nb_repeat_sg times the original signal while adding noise)
def generate_noisy_input(f, win_idx, nb_repeat_sg, noise_val, dim):

    f.seek(dim[0]*dim[1]*win_idx*4)
    sample_non_norm = np.fromfile(f, dtype='float32', count=dim[0]*dim[1])
    sample_non_norm = sample_non_norm.reshape(dim[1],dim[0])
    sample_non_norm = np.swapaxes(sample_non_norm,0,1)

    mean = np.mean(sample_non_norm)
    std = np.std(sample_non_norm)
    sample = (sample_non_norm - mean)/std
    sample=np.expand_dims(sample,0)

    repeated_images = np.repeat(sample, nb_repeat_sg, axis=0)
    noise = np.random.normal(0, noise_val, repeated_images.shape).astype(np.float32)

    noisy_images = repeated_images+noise
    
    # Returns original signal over window win_idx and SmoothGrad input
    return sample_non_norm, noisy_images

# Computes the averaged gradient over the SmoothGrad inputs corresponsing to a given window
def get_av_grad(noisy_images,model,expected_output, num_samples):

    with tf.GradientTape() as tape:
        inputs = tf.cast(noisy_images, tf.float32)
        tape.watch(inputs)
        predictions = model(inputs)
        loss_func = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,alpha=0.25,gamma=2)
        loss = loss_func(expected_output, predictions)
        #loss = model.compute_loss(inputs, expected_output, predictions)   
        grads = tape.gradient(loss, inputs)
        
    grads_per_image = tf.reshape(grads, (-1, num_samples, *grads.shape[1:]))
    averaged_grads = tf.abs(grads_per_image)
    averaged_grads = tf.reduce_mean(averaged_grads, axis=1)

    # Returns averaged gradient and averaged predictions over the SmoothGrad inputs
    return averaged_grads, np.mean(predictions.numpy())

# # Postprocessing of the averaged gradient
# def postprocess_grad(av_grad, axis = 0):
#     av_grad_np = av_grad[0,:,:].numpy() #already abs values
#     mean = np.mean(av_grad_np, axis=axis, keepdims=True)
#     std = np.std(av_grad_np, axis=axis, keepdims=True)
#     norm_grads = np.abs((av_grad_np - mean)/std + 1e-8)
#     # Returns normalized averaged gradient 
#     return norm_grads

def postprocess_grad(av_grad):
    av_grad_np = av_grad[0,:,:].numpy() #already abs values.
    thresh = np.quantile(av_grad_np,0.75)
    av_grad_np[av_grad_np<thresh] = np.min(av_grad_np[av_grad_np>thresh])/2
    av_grad_np[av_grad_np>thresh] = 0.25* ((av_grad_np[av_grad_np>thresh] -  np.min(av_grad_np[av_grad_np>thresh]))/(np.max(av_grad_np[av_grad_np>thresh])-np.min(av_grad_np[av_grad_np>thresh]))) + 0.75
    av_grad_np = signal.wiener(av_grad_np, (7,7))
    return av_grad_np


##################### MAIN

def run_smoothgrad(model_file, model_type, path_to_files, y_pred_path, threshold):

    f = open(f'{path_to_files}/data_raw_1_windows_bi')
    blocks_file = utils.load_obj('data_raw_1_blocks.pkl', path_to_files)
    data_file = utils.load_obj('data_raw_1.pkl', path_to_files)
    full_result = pd.read_csv(y_pred_path)
    # Convert 'probas' column to NumPy array
    y_pred = full_result["probas"].to_numpy()  # or df["probas"].values

    total_nb_windows = len(blocks_file)
    total_nb_points = data_file['meg'][0].shape[1]

    # -- instantiate arrays to store the full signal portion between start_win and stop_win and the corresponding gradient values 
    full_grads = np.zeros((total_nb_points, 274))

    my_labels = np.ones((params.nb_repeat_sg, 1))

    dim = (int(params.sfreq*params.window_size_ms), 274)

    # Model Selection
    if "TensorFlow" in model_type:
        # -- get model
        model = keras.models.load_model(model_file, compile=False)

        # For each window
        for w,i in enumerate(range(0,total_nb_windows)):
            if y_pred[i]>threshold:

                # Noise Augmentation (generating multiple noisy copies of the input before averaging the gradients to reduce variance)
                sample_non_norm, noisy_images = generate_noisy_input(f, w, params.nb_repeat_sg, params.noise_val, dim)


                # Compute SmoothGrad (average and normalizing)
                av_grad, pred = get_av_grad(noisy_images,model,my_labels,params.nb_repeat_sg)

                norm_grads = postprocess_grad(av_grad)

                #If the model predicts a spike then fill the gradient array over the full window (comprising the beggining and end window overlaps)
                full_grads[(w*params.total_lenght)-math.floor(params.overlap/2):(w*params.total_lenght)+params.total_lenght + math.ceil(params.overlap/2),:] = norm_grads[:,:]

    grad_path = f"{path_to_files}/{os.path.basename(model_file)}_smoothGrad.pkl"
    with open(grad_path, 'wb') as f:
        pickle.dump(full_grads, f)

if __name__ == "__main__":
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    path_to_files = sys.argv[3]
    y_pred_path = sys.argv[4]
    threshold = float(sys.argv[5])  # Convert back to float

    run_smoothgrad(model_path, model_type, path_to_files, y_pred_path, threshold)




