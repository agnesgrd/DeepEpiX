# Classification
sfreq = 150  # sampling frequency of the data in Hz
window_size_ms = 0.2
spike_spacing_from_border_ms = 0.03
dim = (int(sfreq*window_size_ms), 274,1) # sample shape
tf_model = "features"
subject_number = 1
loc_meg_channels_path = "model_pipeline/loc_meg_channels.pkl"

# Smoothgrad
nb_repeat_sg = 10
noise_val = 0.1
centre_unique = 12
overlap = 9
total_lenght = (centre_unique + overlap)

# Anomaly detection
sfreq_ae = 150
window_size_ms_ae = 0.4
dim_ae = (int(sfreq_ae*window_size_ms_ae), 274, 1)
