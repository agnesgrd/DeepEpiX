import matplotlib
matplotlib.use('Agg')
import mne
from io import BytesIO
import base64
import matplotlib.pyplot as plt

def create_topomap_from_raw(raw, sfreq, t0, t):
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
    timepoint = float(t-t0)
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
    ax.set_title(f'Time: {t:.3f}s', fontsize=20, color="white")  # Add a title
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

def create_topomap_from_preprocessed(original_raw, raw_ddf, sfreq, t0, t, bad_channels):
    """
    Create a topomap using preprocessed Dask data and original Raw metadata.
    
    Parameters:
    - raw_ddf: Dask DataFrame with shape (time, channels)
    - original_raw: MNE Raw object used to get info structure
    - timepoint: time in seconds at which to extract data
    
    Returns:
    - base64-encoded topomap image string
    """
    # Step 1: Compute Dask DataFrame to NumPy
    preprocessed_df = raw_ddf.drop(columns=bad_channels).compute()
    
    # Step 2: Make sure data is in shape (n_channels, n_times)
    # Dask DFs are usually (n_times, n_channels), so we transpose
    data = preprocessed_df.values.T
    
    # Step 3: Create MNE RawArray using original metadata
    info = original_raw.info.copy()

    if bad_channels:
        info['bads'] = bad_channels
    
    # Optional: filter only MEG channels if needed
    picks = mne.pick_types(info, meg=True, exclude='bads')
    picked_info = mne.pick_info(info, picks)
    
    raw_processed = mne.io.RawArray(data, picked_info).pick('mag')
    
    # Step 4: Use your existing topomap function
    return create_topomap_from_raw(raw_processed, sfreq, t0, t)