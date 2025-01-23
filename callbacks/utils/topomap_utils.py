import matplotlib
matplotlib.use('Agg')
import mne
import io
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import callbacks.utils.graph_utils as gu
import static.constants as c
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
    ax.set_title(f'Time: {timepoint:.3f}s', fontsize=13)  # Add a title
    ax.set_facecolor('white')  # Set background color to white
    ax.axis('off')  # Hide axes for a clean look

    # Save the image to a buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight", pad_inches=0.1)  # Tight bounding box
    buf.seek(0)
    
    # Encode the image in Base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    plt.close('all')
    
    return img_str

# def create_signal_plot(raw, min_time, max_time):
#     # Extract signal data between min_time and max_time


#     step_size = max(1 / 100, (max_time-min_time)/3)  #freq_data.get("resample_freq")

#     time_points = np.arange(float(min_time), float(max_time)+step_size, step_size)

#     start, stop = raw.time_as_index([min_time, max_time])
#     signal, times = raw[::5, start:stop]
#     channel_offset = 0.0000000000003
#     y_axis_ticks = np.arange(signal.shape[0]) * channel_offset
#     signal = signal + y_axis_ticks[:, np.newaxis] 
    
#     # Create the plot
#     plt.figure(figsize=(12, 10))
#     plt.plot(times, signal.T, lw=0.5)  # Transpose for correct plotting
#     plt.xticks(time_points)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.title("Selected Signal")
#     plt.tight_layout()

#     # Save plot to a BytesIO buffer
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format="png")
#     plt.close()
#     buffer.seek(0)

#     # Encode image to base64
#     img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
#     buffer.close()
    
#     return img_str

def create_small_graph_time_channel(min_time, max_time, folder_path, freq_data, time_points):

    selected_channels=c.ALL_CH_NAMES[::5]

    if max_time - min_time < 4:
        min_time = (min_time + max_time)/2 - 2
        max_time = (min_time + max_time)/2 + 2

    fig = gu.generate_small_graph_time_channel(selected_channels, [min_time, max_time], folder_path, freq_data, time_points)

    return fig