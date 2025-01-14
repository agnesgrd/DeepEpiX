import matplotlib
matplotlib.use('Agg')
import mne
from io import BytesIO
import base64
import matplotlib.pyplot as plt

def create_topomap(raw, timepoint):
    """
    Create a topomap image at a specific timepoint.
    Parameters:
    - raw: MNE Raw object
    - timepoint: Time in seconds
    - meg_type: Type of MEG channel to use ('mag' or 'grad')
    Returns:
    - Base64-encoded string of the topomap image
    """
    raw.pick_types(meg=True, ref_meg=False)
  
    # Extract the sampling frequency and calculate the time index
    timepoint = float(timepoint)
    sfreq = raw.info['sfreq']
    time_idx = int(timepoint * sfreq)
    # Extract the data at the specified time index
    data = raw.get_data()  # Shape (n_channels, n_times)
    if time_idx < 0 or time_idx >= data.shape[1]:
        raise ValueError("Timepoint is out of range for the provided data.")
    mean_data = data[:, time_idx]
    fig, ax = plt.subplots()
    im, _ = mne.viz.plot_topomap(
        mean_data,
        raw.info,
        axes=ax,
        show=False
    )
    # Save the image to a buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    # Encode the image in Base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str