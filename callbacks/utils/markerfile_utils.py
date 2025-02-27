import os


# def save_mrk_file(folder_path, mrk_name, annot_dict):
# 	mrk_path = folder_path+'/' + mrk_name + '.mrk'
# 	# print(filepath+'/' + mrk_name + '.mrk')#AnotMarkerFile.mrk
# 	nb_annot = len(annot_dict)
# 	with open(mrk_path, 'w') as f:#AnnotMarkerFile.mrk
# 		f.write('PATH OF DATASET:\n'+folder_path+' \n\n\nNUMBER OF MARKERS:\n'+str(nb_annot)+'\n\n\n')
# 		for cle, valeur in annot_dict.items():
# 			f.write('CLASSGROUPID:\n3\nNAME:\n'+cle+'\nCOMMENT:\n\nCOLOR:\ngreen\nEDITABLE:\nYes\nCLASSID:\n1\nNUMBER OF SAMPLES:\n' + str(len(valeur)) + '\nLIST OF SAMPLES:\nTRIAL NUMBER		TIME FROM SYNC POINT (in seconds)\n')
# 			for v in valeur:
# 				f.write('                  +0				     +'+str(v))
# 				f.write('\n')
# 			f.write('\n\n')

def list_to_dict_of_lists(list_of_dicts):
    # Initialize an empty dictionary to store the results
    dict_of_lists = {}
    
    # Iterate over each dictionary in the list
    for d in list_of_dicts:
        for key, value in d.items():
            # If the key is not in the dict_of_lists, initialize it with an empty list
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            # Append the value to the list for this key
            dict_of_lists[key].append(value)
    
    return dict_of_lists

def annotation_by_description(annotations):
	annotations_by_description = {}

	for annotation in annotations:
		description = annotation['description']
		if description not in annotations_by_description:
			annotations_by_description[description] = []
		annotations_by_description[description].append({
			'onset': annotation['onset'],
			'duration': annotation['duration']
		})

	return annotations_by_description

def save_mrk_file(folder_path, old_mrk_name, new_mrk_name, annotations_to_save,annotations):
	"""Saves annotation data to a .mrk file in the specified folder."""
	
	# Ensure folder exists
	if not os.path.exists(folder_path):
		print(f"Error: Folder '{folder_path}' does not exist.")
		return
	
	# Construct marker file paths safely
	old_mrk_path = os.path.join(folder_path, old_mrk_name + '.mrk')
	new_mrk_path = os.path.join(folder_path, new_mrk_name + '.mrk')
	
	# Check if old marker file exists
	if os.path.exists(old_mrk_path):
		try:
			# Copy the old file to the new file path
			# shutil.copy(old_mrk_path, new_mrk_path)
			# print(f"Old marker file copied to {new_mrk_path}")
			
			annotations_dict = annotation_by_description(annotations)
			nb_annot = len(annotations_to_save)
			# Now rewrite the new marker file with the new annotations

			
			# Open the new file and overwrite it with the new annotations
			with open(new_mrk_path, 'w') as f:
			# Write metadata
				f.write(f"PATH OF DATASET:\n{folder_path} \n\n\nNUMBER OF MARKERS:\n{nb_annot}\n\n\n")

					# Write annotations
				for description, annot_list in annotations_dict.items():
						if description in annotations_to_save:
							f.write(
								f"CLASSGROUPID:\n3\n"
								f"NAME:\n{description}\n"
								f"COMMENT:\n\n"
								f"COLOR:\ngreen\n"
								f"EDITABLE:\nYes\n"
								f"CLASSID:\n1\n"
								f"NUMBER OF SAMPLES:\n{len(annot_list)}\n"
								f"LIST OF SAMPLES:\nTRIAL NUMBER\t\tTIME FROM SYNC POINT (in seconds)\n"
							)
							for v in annot_list:
								f.write(f"      +0\t\t\t+{v['onset']}\n")
							f.write("\n\n")

			print(f"File saved successfully: {new_mrk_path}")

		except Exception as e:
			print(f"Error saving file: {e}")

def modify_name_oldmarkerfile(folder_path, old_mrk_name):
	"""Renames 'MarkerFile.mrk' to 'OldMarkerFile.mrk' in the given folder."""
	old_name = os.path.join(folder_path, "MarkerFile.mrk")
	new_name = os.path.join(folder_path, f"{old_mrk_name}.mrk")

	if os.path.exists(old_name):
		os.rename(old_name, new_name)
		print(f"Renamed '{old_name}' to '{new_name}'")
	else:
		print(f"File '{old_name}' not found!")


# filepath = "E:/BlackHardDrive/IterativeLearningNewAnnots/IterativeLearningFeedback9/Louil_AllDataset1200Hz/louil_Epi-001_20091112_05.ds"#"E:/BlackHardDrive/IterativeLearningNewAnnots/IterativeLearningFeedback7/Melpa_AllDataset1200Hz/melpa_Epi-001_20090320_07.ds"
# mne_annot = filepath+'/louil_ds_05.txt'
# data = pd.read_csv(mne_annot, sep=",", skiprows=2)
# data.insert(2, "timing", data['# onset']+(data[' duration']/2.) , True)

# annot_dict = {}

# labels= np.unique(data[' description'])
# for l in labels:	
# 	annot_dict[l] = data.loc[data[' description'] == l]['timing'].values.tolist()


# paths=["E:/BlackHardDrive/IterativeLearningNewAnnots/Holdout/Chaco/chaco_Epi-001_20080306_02.ds/"]

# mrk_list = ["MarkerFile_jj.mrk","MarkerFile_rm.mrk"]

# for path in paths:
# 	nb_markers = 0
# 	header=list()
# 	content=list()

# 	for file in mrk_list:
# 		print(file)
# 		if os.path.isfile(path+file):
# 			print(file)
# 			spl = list()

# 			f=open(path+file)
# 			lines = f.readlines()
			
# 			#Get number of markers and sums based on prev mrk files
# 			nb_markers= nb_markers + int(lines[lines.index("NUMBER OF MARKERS:\n")+1])

# 			#Split by groupID to get single marker infos if file contains several markers
# 			w = 'CLASSGROUPID:\n'
# 			spl = [list(y) for x, y in itertools.groupby(lines, lambda z: z == w) if not x]
# 			spl.pop(0)
# 			for l in spl:
# 				l.insert(0,'CLASSGROUPID:\n')

# 			#Adjusts CLASSGROUPID:\n value. should be "3" for the first marker and "+3" for next markers (I don't know why)
# 			if content:
# 				for l in spl:
# 					l[1]='+3\n'
# 			else:
# 				if (len(spl)>1):
# 					for l in spl[1:]:
# 						l[1]='+3\n'
# 				else:
# 					l[1]='3\n'

# 			#Append next marker infos to the content
# 			for l in spl:		
# 				content.append(l)

# 			print(content)

# 	with open(path+'MergedMarkerFile.mrk', 'w') as f:
# 		f.write('PATH OF DATASET:\n'+path+' \n\n\nNUMBER OF MARKERS:\n'+str(nb_markers)+'\n\n\n')
# 		for markerinfo in content:
# 			for line in markerinfo:
# 				print(line)
# 				f.write(line)