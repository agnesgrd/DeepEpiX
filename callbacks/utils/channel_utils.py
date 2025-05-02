import re
from collections import defaultdict
import mne

def get_grouped_channels_by_prefix(raw):
    """
    Load channels from a .ds folder and group them by their 3-letter prefix.

    Args:
        folder_path (str): Path to the .ds folder.

    Returns:
        dict: Dictionary where keys are 3-letter prefixes and values are lists of channel names.
    """
    grouped_channels = defaultdict(list)
    prefix_pattern = re.compile(r'^[A-Z]{3}$')

    # Get only MEG channels (both magnetometers and gradiometers)
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False)
    meg_ch_names = [raw.info['ch_names'][i] for i in meg_picks]
    print(meg_ch_names)

    for ch_name in meg_ch_names:
        prefix = ch_name.split('-')[0][:3]
        if prefix_pattern.match(prefix):
            grouped_channels[prefix].append(ch_name)

    return dict(grouped_channels)