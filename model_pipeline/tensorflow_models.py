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
from model_pipeline.utils import load_obj
from model_pipeline.utils import compute_window_ppa, compute_window_upslope, compute_window_std, compute_window_average_slope, compute_window_downslope, compute_window_sharpness, compute_gfp, find_peak_gfp

#####################################################################Running the model

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# def classify_sfcn(first_direction,sfreq,window_size):

#     reg = 0

#     if first_direction == 'channel':
#         input_shape = (274, int(sfreq*window_size), 1)
#     elif first_direction == 'time':
#         input_shape = (int(sfreq*window_size), 274, 1)


#     input_layer = keras.layers.Input(input_shape)

#     #block 1
#     x=keras.layers.Conv2D(filters=32, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(input_layer)
#     x=keras.layers.BatchNormalization(name="t1_norm1")(x)

#     x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
#     x=keras.layers.LeakyReLU()(x)

#     #block 2
#     x=keras.layers.Conv2D(filters=64, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
#     x=keras.layers.BatchNormalization()(x)
#     x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
#     x=keras.layers.LeakyReLU()(x)

#     #block 3
#     x=keras.layers.Conv2D(filters=128, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
#     x=keras.layers.BatchNormalization()(x)
#     x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
#     x=keras.layers.LeakyReLU()(x)

#     #block 4
#     x=keras.layers.Conv2D(filters=256, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
#     x=keras.layers.BatchNormalization()(x)
#     x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
#     x=keras.layers.LeakyReLU()(x)

#     #block 6
#     x=keras.layers.Conv2D(filters=64, kernel_size=(1, 1),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
#     x=keras.layers.BatchNormalization()(x)
#     x=keras.layers.LeakyReLU()(x)

#     #block 7, different from paper
#     x=keras.layers.AveragePooling2D((2,2),padding='same')(x)
#     x=keras.layers.Dropout(.5)(x)
#     x=keras.layers.Flatten()(x)    
#     output_layer = keras.layers.Dense(1, activation='sigmoid')(x)

#     model = keras.models.Model(inputs=input_layer, outputs=output_layer)

#     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
#                                                       min_lr=0.00001)
    
#     model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,gamma=2), optimizer=keras.optimizers.Adam(0.0001),
#                       metrics=['accuracy','Precision','Recall',f1_m])#0.0001#Adam(0.0001)


#     return model, reduce_lr

# def classify_features_only(first_direction,sfreq,window_size,initial_bias, alpha=0.25, nb_features=1):

#     if first_direction == 'channel':
#         (274, int(sfreq*window_size), 1)
#     elif first_direction == 'time':
#         (int(sfreq*window_size), 274, 1)

#     input_features = keras.layers.Input(shape=(nb_features,274))

#     x_features=keras.layers.Flatten()(input_features)
    
#     x=keras.layers.Dense(512, activation="relu")(x_features)#64
#     x=keras.layers.Dropout(.2)(x)
#     x=keras.layers.Dense(128, activation="relu")(x)#32
#     x=keras.layers.Dropout(.2)(x)
#     x=keras.layers.Dense(16, activation="relu")(x)#16
#     x=keras.layers.Dropout(.2)(x)
    
#     output_layer = keras.layers.Dense(1, activation='sigmoid',bias_initializer=tf.constant_initializer(initial_bias))(x)

#     model = keras.models.Model(inputs=input_features, outputs=output_layer)

#     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
#                                                       min_lr=0.00001)
    
#     model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,gamma=2, alpha=alpha), optimizer=keras.optimizers.Adam(0.0001),
#                       metrics=['accuracy','Precision','Recall'])#tf.keras.losses.BinaryCrossentropy()


#     return model, reduce_lr

# class DataGenerator_memeff(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, batch_size, dim, path):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.list_IDs = list_IDs
#         self.path = path
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#         list_IDs_temp = np.array(list_IDs_temp)
#         list_IDs_temp = list_IDs_temp[list_IDs_temp[:, 1].argsort()].tolist()

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

#     def on_epoch_end(self):
#         #'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))

#     def __data_generation(self, list_IDs_temp):
#        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization   
#         X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1])) # contains images
#         y_batch = np.empty((self.batch_size), dtype=int)

#         if len(self.dim) == 3:
#             X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], 1)) # contains images

#         prefixe = 'data_raw_'
#         suffixe = '_windows_bi' 

#         f = open(op.join(self.path, prefixe+'1'+suffixe))

#         for i, ID in enumerate(list_IDs_temp):

#             win = np.array(ID)[0]
#             np.array(ID)[1]
#             label = np.array(ID)[2]

#             # Store sample 
#             f.seek(self.dim[0]*self.dim[1]*win*4) #4 because its float32 and dtype.itemsize = 4
#             sample = np.fromfile(f, dtype='float32', count=self.dim[0]*self.dim[1])
#             sample = sample.reshape(self.dim[1],self.dim[0])
#             sample = np.swapaxes(sample,0,1)
#             if len(self.dim) == 3:
#                 sample = np.expand_dims(sample,axis=-1)

#             sample_augmented = sample

#             mean = np.mean(sample_augmented)
#             std = np.std(sample_augmented)
#             sample_augmented = (sample_augmented - mean)/std

#             X_batch[i,] = sample_augmented

#             # Store class
#             y_batch[i,] = label
        
#         f.close()

#         return X_batch, y_batch

# # Data generator to load images batch per batch
# class DataGenerator_memeff_feat_only(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, batch_size, dim, path, compute_features = {"ppa" : compute_window_ppa, "std": compute_window_std, "upslope" : compute_window_upslope, "downslope": compute_window_downslope, "average_slope": compute_window_average_slope, "sharpness" : compute_window_sharpness}):
    
#         'Initialization'
#         self.dim = dim # dimension of the window
#         self.batch_size = batch_size
#         self.list_IDs = list_IDs
#         self.path = path # where the .pkl and binary files are
#         self.comute_features = compute_features
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#         list_IDs_temp = np.array(list_IDs_temp)
#         list_IDs_temp = list_IDs_temp[list_IDs_temp[:, 1].argsort()].tolist()

#         # Generate data
#         features, y = self.__data_generation(list_IDs_temp)

#         return features, y

#     def on_epoch_end(self):
#         #'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))

#     def __data_generation(self, list_IDs_temp):
#        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization   
#         features_batch = np.empty((self.batch_size, len(list(self.comute_features)), self.dim[1]))
#         y_batch = np.empty((self.batch_size), dtype=int) # contains window labels
#         prefixe = 'data_raw_'
#         suffixe = '_windows_bi'       
#         #if len(self.dim) == 3:
#         #X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], 1)) # contains images

#         # Generate data

#         # if the subject is different from the previous one, then open the subject file
#         f = open(op.join(self.path, prefixe+'1'+suffixe))

#         for i, ID in enumerate(list_IDs_temp):
#             # get infos from the "ids" array
#             win = np.array(ID)[0]
#             np.array(ID)[1]
#             label = np.array(ID)[2]

#             if np.array(ID).shape[0]>3:
#                 duplicate = np.array(ID)[3]
#             else:
#                duplicate = 0

#             # Read the window data from the binary file (seek=move the cursor, fromfile=extract the correct nb of bytes) 
#             f.seek(self.dim[0]*self.dim[1]*win*4) #4 because its float32 and dtype.itemsize = 4
#             sample = np.fromfile(f, dtype='float32', count=self.dim[0]*self.dim[1])
#             # reshape the window data
#             sample = sample.reshape(self.dim[1],self.dim[0])
#             sample = np.swapaxes(sample,0,1)
#             # add a "channel" dimension needed for a CNN (last dim in tensorflow)

#             sample_augmented = sample

#             mean = np.mean(sample_augmented)
#             std = np.std(sample_augmented)
#             sample_augmented = (sample_augmented - mean)/std

#             if duplicate == 1:
#                 print("in duplicate")
#                 noise = np.random.normal(loc=0.0, scale=0.2, size=sample_augmented.shape)
#                 sample_augmented = sample_augmented + noise


#             for feat, func in self.comute_features.items():
#                 features_batch[i,list(self.comute_features).index(feat)] = func(sample_augmented)

#             # Store classf
#             y_batch[i,] = label
        
#         f.close()

#         return features_batch, y_batch

# def load_generators_memeff(X_test_ids, output_path):

#     testing_generator = DataGenerator_memeff(X_test_ids.tolist(), 1, params.dim, output_path)

#     return testing_generator

# def load_generators_memeff_feat_only(X_test_ids, output_path):

#     testing_generator = DataGenerator_memeff_feat_only(X_test_ids.tolist(), 1, params.dim, output_path)

#     return testing_generator

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

# def test_model(model_name, testing_generator, X_test_ids):

#     model = keras.models.load_model(model_name, compile=False)
#     model.compile()
#     y_pred_probas = model.predict(testing_generator)
#     y_test = X_test_ids[:,2]

#     y_pred = (y_pred_probas > 0.5).astype("int32")

#     sub = X_test_ids[:,1]
#     win = X_test_ids[:,0]

#     y_timing_data = load_obj("data_raw_"+str(params.subject_number)+'_timing.pkl',params.args.path_output)
#     y_block_data = load_obj("data_raw_"+str(params.args.subject_number)+'_blocks.pkl',params.args.path_output)

#     print('writing here: ',params.args.path_output+'subject_'+str(params.args.subject_number)+'_predictions.csv')
#     with open(params.args.path_output+'subject_'+str(params.args.subject_number)+'_predictions.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["subject","block","timing","pred"])

#     for ind, i in enumerate(sub):
#         y_timing = y_timing_data[win[ind]]
#         y_block = y_block_data[win[ind]]

#         with open(params.args.path_output+'subject_'+str(params.args.subject_number)+'_predictions.csv', 'a', newline='') as f:
#             writer = csv.writer(f)           
#             writer.writerow([i,y_block,y_timing,y_pred[ind][0]])

#     keras.backend.clear_session()

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