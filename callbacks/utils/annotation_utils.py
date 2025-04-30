from collections import Counter
import mne
import pandas as pd
import math
from dash import dcc,html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


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

def build_table_events_statistics(annotations):

    if len(annotations) == 0:
        return html.P("No annotations found in this recording.")

    # Extract values from list of dicts
    descriptions = [ann["description"] for ann in annotations]
    onsets = [ann["onset"] for ann in annotations]
    durations = [ann["duration"] for ann in annotations]

    # Count descriptions
    description_counts = Counter(descriptions)

    # Build table
    table_header = html.Thead(html.Tr([html.Th("Event Name"), html.Th("Count")]))
    table_body = html.Tbody([
        html.Tr([html.Td(desc), html.Td(count)]) for desc, count in description_counts.items()
    ])

    annotation_table = dbc.Table(
        [table_header, table_body],
        bordered=True,
        striped=True,
        hover=True,
        size="sm"
    )

    # Summary stats
    stats_summary = dbc.Card(
        dbc.CardBody([
            html.H5([
                html.I(className="bi bi-bar-chart-line", style={"marginRight": "10px", "fontSize": "1.2em"}),
                "Event Summary"
            ], className="card-title"),
            html.Hr(),
            dbc.ListGroup([
                dbc.ListGroupItem(f"Total annotations: {len(annotations)}"),
                dbc.ListGroupItem(f"Unique event types: {len(description_counts)}"),
                annotation_table,
                dbc.ListGroupItem(f"First event starts at {min(onsets):.2f} s"),
                dbc.ListGroupItem(f"Last event ends at {max([o + d for o, d in zip(onsets, durations)]):.2f} s"),
            ])
        ])
    )

    return stats_summary

def build_table_prediction_statistics(df, threshold):
     
    # Separate spike vs non-spike
    df['probas'] = df['probas'].round(2)
    df_spike = df[df["probas"] > threshold]
    df_non_spike = df[df["probas"] <= threshold]

    if len(df_spike) == 0:
        return html.P("No events found in this recording.")

    # General counts
    total_windows = len(df)
    spike_count = len(df_spike)
    spike_ratio = (spike_count / total_windows) * 100 if total_windows else 0

    if spike_count > 0:
        min_prob = df_spike['probas'].min()
        max_prob = df_spike['probas'].max()
        mean_prob = df_spike['probas'].mean()
        median_prob = df_spike['probas'].median()
    else:
        min_prob = max_prob = mean_prob = median_prob = 0

    stats_summary = dbc.ListGroup([
        dbc.ListGroupItem(f"Total Windows: {total_windows}"),
        dbc.ListGroupItem(f"Spike Events Detected): {spike_count}"),
        dbc.ListGroupItem(f"Spike Ratio: {spike_ratio:.2f}%"),
        dbc.ListGroupItem(f"Min Spike Probability: {min_prob:.2f}"),
        dbc.ListGroupItem(f"Max Spike Probability: {max_prob:.2f}"),
        dbc.ListGroupItem(f"Mean Spike Probability: {mean_prob:.2f}"),
        dbc.ListGroupItem(f"Median Spike Probability: {median_prob:.2f}")
    ])

    return stats_summary

def build_prediction_distribution_statistics(df, threshold):
    # Filter the DataFrame based on the threshold
    df_below_threshold = df[df['probas'] <= threshold]
    df_above_threshold = df[df['probas'] > threshold]

    # Create the histogram for probabilities below the threshold
    hist_below = go.Histogram(
        x=df_below_threshold['probas'],
        nbinsx=10,
        name=f"Below {threshold}",
        marker=dict(color='yellow'),  # Color for below threshold
        opacity=0.7,
        showlegend=False
    )

    # Create the histogram for probabilities above the threshold
    hist_above = go.Histogram(
        x=df_above_threshold['probas'],
        nbinsx=10,
        name=f"Above {threshold}",
        marker=dict(color='red'),  # Color for above threshold
        opacity=0.7,
        showlegend=False
    )

    # Combine the two histograms into one figure
    hist_fig = go.Figure(data=[hist_below, hist_above])
    
    # Customize the layout with smaller font sizes
    hist_fig.update_layout(
        barmode='overlay',
        bargap=0.2,
        xaxis_title='Probability',
        yaxis_title='Count',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', size=10),  # Smaller overall font
        xaxis=dict(title_font=dict(size=10), tickfont=dict(size=9)),
        yaxis=dict(title_font=dict(size=10), tickfont=dict(size=9)),
        margin=dict(t=0, b=0, l=0, r=0)
    )

    return dcc.Graph(
        figure=hist_fig,
        config={
            'staticPlot': False,           # ← This makes the plot static
            'displayModeBar': True,      # Hides the mode bar
            'scrollZoom': False
        },
        style={'height': '300px'}  # Optional: set size explicitly
    )