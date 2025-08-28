import os


def list_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists


def annotation_by_description(annotations):
    annotations_by_description = {}
    for annotation in annotations:
        description = annotation["description"]
        if description not in annotations_by_description:
            annotations_by_description[description] = []
        annotations_by_description[description].append(
            {"onset": annotation["onset"], "duration": annotation["duration"]}
        )

    return annotations_by_description


def modify_name_oldmarkerfile(folder_path, old_mrk_name):
    """Renames 'MarkerFile.mrk' to 'OldMarkerFile.mrk' in the given folder."""
    old_name = os.path.join(folder_path, "MarkerFile.mrk")
    new_name = os.path.join(folder_path, f"{old_mrk_name}.mrk")

    if os.path.exists(old_name):
        os.rename(old_name, new_name)
        print(f"Renamed '{old_name}' to '{new_name}'")
    else:
        print(f"File '{old_name}' not found!")


def save_mrk_file(folder_path, new_mrk_name, annotations_to_save, annotations):
    """Saves annotation data to a .mrk file in the specified folder."""
    if not os.path.exists(folder_path):
        return f"⚠️ Error: Folder '{folder_path}' does not exist."

    new_mrk_path = os.path.join(folder_path, new_mrk_name + ".mrk")

    try:
        annotations_dict = annotation_by_description(annotations)
        nb_annot = len(annotations_to_save)

        with open(new_mrk_path, "w") as f:
            f.write(
                f"PATH OF DATASET:\n{folder_path} \n\n\nNUMBER OF MARKERS:\n{nb_annot}\n\n\n"
            )

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
        print(f"⚠️ Error saving file: {e}")
