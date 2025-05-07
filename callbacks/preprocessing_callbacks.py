# Dash & Plotly
import dash
from dash import Input, Output, State, callback

# External Libraries
import mne

# Local Imports
from callbacks.utils import folder_path_utils as fpu
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import annotation_utils as au
from callbacks.utils import channel_utils as chu

def register_handle_frequency_parameters():
    @callback(
        Output("preprocess-status", "children"),
        Input("resample-freq", "value"),
        Input("high-pass-freq", "value"),
        Input("low-pass-freq", "value"),
        Input("notch-freq", "value"),
        prevent_initial_call=True
    )
    def handle_frequency_parameters(resample_freq, high_pass_freq, low_pass_freq, notch_freq):
        """Retrieve frequency parameters and store them."""
        if not low_pass_freq or not high_pass_freq or not notch_freq:
            return f"⚠️ Please fill in all frequency parameters."
        
        elif high_pass_freq >= low_pass_freq:
            return f"⚠️ High-pass frequency must be less than low-pass frequency."

        return dash.no_update

def register_preprocess_meg_data():
    @callback(
        Output("preprocess-status", "children", allow_duplicate=True),
        Output("frequency-store", "data"),
        Output("annotation-store", "data"),
        Output("channel-store", "data"),
        Output("chunk-limits-store", "data"),
        Output("url", "pathname"),
        Input("preprocess-display-button", "n_clicks"),
        State("folder-store", "data"),
        State("resample-freq", "value"),
        State("high-pass-freq", "value"),
        State("low-pass-freq", "value"),
        State("notch-freq", "value"),
        State("heartbeat-channel", "value"),
        running=[
            (Output("preprocess-display-button", "disabled"), True, False),
            (Output("load-button", "disabled"), True, False),
            (Output("compute-display-psd-button", "disabled"), True, False)],
        prevent_initial_call=True
    )
    def preprocess_meg_data(n_clicks, folder_path, resample_freq, high_pass_freq, low_pass_freq, notch_freq, heartbeat_ch_name):
        """Preprocess MEG data and save it, store annotations and chunk limits in memory."""
        if n_clicks > 0:
            try:
                raw = fpu.read_raw(folder_path, preload=True, verbose=False)
                annotations_dict, max_length = au.get_annotations_dataframe(raw, heartbeat_ch_name)
                channels_dict = chu.get_grouped_channels_by_prefix(raw)
                chunk_limits = pu.update_chunk_limits(max_length)

                # Store the frequency values when the folder is valid
                freq_data = {
                    "resample_freq": resample_freq,
                    "low_pass_freq": low_pass_freq,
                    "high_pass_freq": high_pass_freq,
                    "notch_freq": notch_freq
                }

                prep_raw = pu.filter_resample(folder_path, freq_data)

                for chunk_idx in chunk_limits:
                    start_time, end_time = chunk_idx
                    raw_df = pu.get_preprocessed_dataframe_dask(folder_path, freq_data, start_time, end_time, prep_raw)

                return "Preprocessed and saved data", freq_data, annotations_dict, channels_dict, chunk_limits, "/viz/raw-signal"
            
            except Exception as e:
                return f"Error during preprocessing : {str(e)}", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update