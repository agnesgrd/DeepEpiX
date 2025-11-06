import re
from collections import defaultdict
import mne
import config
from pathlib import Path
import json


def get_grouped_channels_meg(grouped_channels, ch_names):
    prefix_pattern = re.compile(r"^[A-Z]{3}$")

    if "MEG" in ch_names[int(len(ch_names) / 2)]:

        with open(Path(config.MONTAGES_DIR / "montage_MEG...123.json"), "r") as f:
            CHANNEL_GROUPS = json.load(f)

        for region, channels in CHANNEL_GROUPS.items():
            filtered_channels = [ch for ch in channels if ch in ch_names]
            grouped_channels[region] = filtered_channels

    elif "M" in ch_names[int(len(ch_names) / 2)]:
        for ch_name in ch_names:
            prefix = ch_name.split("-")[0][:3]
            if prefix_pattern.match(prefix):
                grouped_channels[prefix].append(ch_name)

    elif "A" in ch_names[int(len(ch_names) / 2)]:

        with open(Path(config.MONTAGES_DIR / "montage_A1...json"), "r") as f:
            CHANNEL_GROUPS = json.load(f)

        for region, channels in CHANNEL_GROUPS.items():
            filtered_channels = [ch for ch in channels if ch in ch_names]
            grouped_channels[region] = filtered_channels

    return grouped_channels


def get_grouped_channels_by_prefix(raw, modality, bad_channels=None):
    """
    Load channels from raw data and group them by their 3-letter prefix.

    Returns:
        dict: Dictionary where keys are 3-letter prefixes and values are lists of channel names.
    """
    grouped_channels = defaultdict(list)

    if modality == "meg":
        # Get only MEG channels (both magnetometers and gradiometers)
        ch_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False)
        ch_names = [raw.info["ch_names"][i] for i in ch_picks]
        grouped_channels = get_grouped_channels_meg(grouped_channels, ch_names)

    elif modality == "eeg":
        ch_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
        ch_names = [raw.info["ch_names"][i] for i in ch_picks]
        grouped_channels["EEG"] = ch_names

    elif modality == "mixed":
        # Get only MEG channels (both magnetometers and gradiometers)
        ch_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False)
        ch_names = [raw.info["ch_names"][i] for i in ch_picks]
        grouped_channels = get_grouped_channels_meg(grouped_channels, ch_names)
        eeg_ch_picks = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False
        )
        grouped_channels["EEG"] = [raw.info["ch_names"][i] for i in eeg_ch_picks]

    elif modality == "unkown":
        raise Exception(
            "Cannot determine the modality of the raw data: no EEG or MEG channels found."
        )

    if bad_channels:
        if isinstance(bad_channels, str):
            bad_channels_list = [
                ch.strip() for ch in bad_channels.split(",") if ch.strip()
            ]
        else:
            bad_channels_list = list(bad_channels)
        grouped_channels["bad"] = bad_channels_list

    return grouped_channels
