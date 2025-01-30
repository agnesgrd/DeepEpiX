# import argparse

# #example submission command-line
# # python test_model.py '/sps/crnl/pmouches/testJulien/' '/sps/crnl/pmouches/testJulien/good_channels' '/sps/crnl/pmouches/Chekh/chekh_Epi-001_20071022_03.ds' 1 '/sps/crnl/pmouches/testJulien/sfcn_time_all_best_model.h5' 'new_spike' 'Tensorflow'
# parser = argparse.ArgumentParser(description='Model testing.')
# # ou sauver les fichiers intermediaires (fenetres extraites du .ds et resultats bruts du modèle)
# # /!\ ends with "/"
# parser.add_argument('path_output',type=str)
# # chemin vers le fichier good_channels
# parser.add_argument('good_channels_file',type=str)
# # Chemin vers le .ds
# parser.add_argument('subject',type=str)
# # Numéro du sujet. Il sera utilisé pour nommer les fichiers intermediairs sauvés dans "path_output". Ca peut juste être un numéro arbitraire
# parser.add_argument('subject_number',type=int)
# # Chemin vers le fichier contenant les poids du modèle à utiliser
# parser.add_argument('model_name',type=str)
# # Nom à donner au nouvelles annotations générées par le modèle
# parser.add_argument('marker_name',type=str,default='detected_spike')
# # Type de modèle
# parser.add_argument('model_type',type=str,default='Tensorflow')

# args = parser.parse_args()

sfreq = 150  # sampling frequency of the data in Hz
window_size_ms = 0.2
spike_spacing_from_border_ms = 0.03
dim = (int(sfreq*window_size_ms), 274,1) # sample shape
tf_model = "features"
subject_number = 1
