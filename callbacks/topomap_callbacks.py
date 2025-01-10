import dash
from dash import Input, Output, State
import mne
import os
import plotly.io as pio
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import pandas as pd
from pages.home import get_preprocessed_dataframe
import callbacks.utils.graph_utils as gu
import numpy as np

# Callback to handle the plotting of the topomap
def register_display_topomap():
    @dash.callback(
        Output("topomap-img", "src"),
        Output("topomap-popup", "is_open"),
        [Input("plot-topomap-btn-2", "n_clicks")],
        [State("topomap-min-range", "value"), 
        State("topomap-max-range", "value"),
        State("folder-store", "data"),
        State("frequency-store", "data") ,
        State("topomap-popup", "is_open")],
        prevent_initial_call=True
    )
    def plot_topomap(n_clicks, min_time, max_time, folder_path,freq_data, is_open):
        if n_clicks>0 and min_time is not None and max_time is not None:
            try:
                # Retrieve preprocessed data from cache
                raw_df = get_preprocessed_dataframe(folder_path, freq_data)
                if "Error" in raw_df:
                    raise ValueError(raw_df)

                # Select the time range
                _, selected_data = gu.get_raw_df_filtered_on_time([min_time, max_time], raw_df)

                # Compute the average across the selected time range
                mean_data = selected_data.mean(axis=0)

                # Get channel positions
                raw = mne.io.read_raw_ctf(folder_path, preload=False, verbose=False)  # Load metadata only
                raw = raw.pick('meg')

                # Get channel positions (2D) from raw.info
                pos = np.array([ch['loc'][:2] for ch in raw.info['chs'] if ch['kind'] == mne.constants.FIFF.FIFFV_MEG_CH])
                ch_names = [ch['ch_name'] for ch in raw.info['chs'] if ch['kind'] == mne.constants.FIFF.FIFFV_MEG_CH]

                # Ensure the mean data aligns with the channel names
                mean_data = mean_data[ch_names]

                # Plot the topomap
                fig, ax = plt.subplots()
                mne.viz.plot_topomap(raw.get_data(), raw.info, ch_type='meg', axes=ax, show=False)

                # Save the figure as a PNG in memory
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)

                # Encode the image in Base64
                img_str = base64.b64encode(buf.read()).decode('utf-8')

                # Return the Base64 string as a data URL for Dash
                return f"data:image/png;base64,{img_str}", not is_open
            
            except Exception as e:
                print(f"Error in plot_topomap: {str(e)}")
                return "https://via.placeholder.com/150", is_open
            
        # If no button click or invalid input, return a placeholder image
        return "https://via.placeholder.com/150", is_open
    
            # resample_freq = freq_data.get("resample_freq")
            # low_pass_freq = freq_data.get("low_pass_freq")
            # high_pass_freq = freq_data.get("high_pass_freq")

            # # Load the raw data
            # raw = mne.io.read_raw_ctf(folder_path, preload=True, verbose=False)
            # raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=8)
            # raw.resample(resample_freq)

            # # Crop the data to the specified time range and compute the average
            # cropped_raw = raw.copy().crop(tmin=min_time, tmax=max_time)
            # data, times = cropped_raw[:, :][0], cropped_raw.times
            # mean_data = np.mean(data, axis=1)

            # # Get channel positions
            # pos = mne.find_layout(raw.info).pos

            # # Create the topomap
            # fig, ax = plt.subplots()
            # mne.viz.plot_topomap(mean_data, pos, axes=ax, show=False)

            # # Save the figure as a PNG in memory
            # buf = BytesIO()
            # fig.savefig(buf, format="png", bbox_inches="tight")
            # plt.close(fig)
            # buf.seek(0)

            # # Encode the image in Base64
            # img_str = base64.b64encode(buf.read()).decode('utf-8')

            # # Return the Base64 string as a data URL for Dash
            # return f"data:image/png;base64,{img_str}", not is_open
        
        # If no button click or invalid input, return the placeholder image
        return "https://via.placeholder.com/150", is_open
    
# # Callback to open the modal when the "Plot Topomap" button is clicked
# def register_display_topomap():
#     @dash.callback(
#         Output("topomap-modal", "is_open"),
#         [Input("plot-topomap-btn-1", "n_clicks")],
#         [State("topomap-modal", "is_open")],
#         prevent_initial_call=True
#     )
#     def toggle_modal(n_clicks, is_open):
#         if n_clicks:
#             return not is_open
#         return is_open

def register_close_topomap():
    # Callback to close the modal when the close button inside the modal is clicked
    @dash.callback(
        Output("topomap-modal", "is_open", allow_duplicate=True),
        [Input("close-topomap-modal", "n_clicks")],
        [State("topomap-modal", "is_open")],
        prevent_initial_call=True
    )
    def close_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open