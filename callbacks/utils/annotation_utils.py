from collections import Counter
import mne
import pandas as pd
import math

def get_annotation_descriptions(annotations_store):
        """
        Extracts the list of unique annotation description names 
        from the annotations-store.

        Parameters:
        annotations_store (list): A list of dictionaries representing annotations.

        Returns:
        list: A list of unique description names.
        """
        if not annotations_store or not isinstance(annotations_store, list):
            return []

        # Extract descriptions
        descriptions = [annotation.get('description') for annotation in annotations_store if 'description' in annotation]

        # Count occurrences of each description
        description_counts = Counter(descriptions)

        # Return unique descriptions
        return description_counts

def get_heartbeat_event(raw, ch_name):
    # Find ECG events using the `find_ecg_events` function
    events, _, _ = mne.preprocessing.find_ecg_events(
        raw,
        ch_name = ch_name

    )
    
    # Get the sampling frequency (in Hz)
    sfreq = raw.info['sfreq']

    # Create a list to store the event information
    event_list = []

    # For each ECG event, create a dictionary with onset (in seconds), description, and duration
    for event in events:
        onset_sample = event[0]  # The event onset in samples
        onset_sec = onset_sample / sfreq  # Convert to seconds
        description = 'ECG Event'  # You can customize this
        duration = 0  # Duration in seconds (for simplicity, we'll assume a 1-second duration)
                
        # Append to the event list
        event_list.append({
            'onset': onset_sec,
            'description': description,
            'duration': duration
        })

    return pd.DataFrame(event_list)

def get_annotations_dataframe(raw, heartbeat_ch_name):
    
    annotations_df = raw.annotations.to_data_frame()
    
    # Convert the 'onset' column to datetime and localize it to UTC
    annotations_df['onset'] = pd.to_datetime(annotations_df['onset']).dt.tz_localize('UTC')
    
    # Calculate onset relative to origin_time in seconds
    origin_time = pd.Timestamp(raw.annotations.orig_time)
    annotations_df['onset'] = (annotations_df['onset'] - origin_time).dt.total_seconds()
    
    time_secs = raw.times

    heartbeat_df = get_heartbeat_event(raw, heartbeat_ch_name)

    df_combined = pd.concat([annotations_df, heartbeat_df], ignore_index=True)

    # Convert to dictionary format
    annotations_dict = df_combined.to_dict(orient="records")

    return annotations_dict, math.floor(time_secs[-1]*100)/100

def get_annotations(prediction_or_truth, annotations_df):
    """
    Function to retrieve annotation onsets (timestamps) based on the selected prediction or ground truth.
    
    Parameters:
    - prediction_or_truth (str): Either the model's prediction or the ground truth label.
    - annotations_df (pandas.DataFrame): DataFrame containing the annotations with onset times and other info.
    
    Returns:
    - List of onset times for the selected annotation type.
    """
    if isinstance(prediction_or_truth, str):
        prediction_or_truth = [prediction_or_truth]

    # Filter annotations where 'description' matches any of the provided descriptions
    filtered_annotations = annotations_df[annotations_df["description"].isin(prediction_or_truth)]

    # Return the onsets (index) as a list
    return filtered_annotations.index.tolist()