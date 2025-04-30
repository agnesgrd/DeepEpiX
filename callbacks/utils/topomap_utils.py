import matplotlib
matplotlib.use('Agg')
import mne
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import callbacks.utils.graph_utils as gu
import config
import callbacks.utils.graph_utils as gu


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
    fig.patch.set_facecolor('black')   # Set the figure background
    ax.set_facecolor('black')          # Set the axes background

    im, _ = mne.viz.plot_topomap(
        mean_data,
        raw.info,
        axes=ax,
        show=False,
        cmap='coolwarm',  # Choose a visually pleasing color map
        contours=6,  # Add contour lines
        sensors=True,  # Display the sensor locations on the topomap
        res=128  # Resolution of the topomap
    )

    # Customize the plot appearance
    ax.set_title(f'Time: {timepoint:.3f}s', fontsize=20, color="white")  # Add a title
    ax.axis('off')  # Hide axes for a clean look

    # Save the image to a buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight", pad_inches=0.1,  facecolor=fig.get_facecolor())  # Tight bounding box
    buf.seek(0)
    
    # Encode the image in Base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    plt.close('all')
    
    return img_str