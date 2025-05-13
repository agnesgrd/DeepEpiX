import time
import dash
from dash import html, Input, Output, State, callback
from callbacks.utils import folder_path_utils as fpu
from callbacks.utils import topomap_utils as tu
from callbacks.utils import preprocessing_utils as pu

def register_display_topomap_on_click():
    @callback(
        Output("topomap-result", "children"),
        Output("topomap-picture", "children"),
        Output("history-store", "data", allow_duplicate=True),
        Input('meg-signal-graph', 'clickData'),
        State('folder-store', 'data'),
        State('plot-topomap-button', 'outline'),
        State('page-selector', 'value'),
        State('chunk-limits-store', 'data'),
        State('frequency-store', 'data'),
        prevent_initial_call=True
    )
    def display_clicked_content(click_info, folder_path, button, page_selection, chunk_limits, freq_data):
        if button is False:
            try:
                start_time = time.time()  # Start timing
                t = click_info["points"][0]['x']
                print(f"Time to extract time from click info: {time.time() - start_time:.4f} seconds")
                
                # Load raw data (metadata only)
                load_start_time = time.time()
                raw = fpu.read_raw(folder_path, preload=False, verbose=False)
                time_range = chunk_limits[int(page_selection)]
                raw_ddf = pu.get_preprocessed_dataframe_dask(folder_path, freq_data, time_range[0], time_range[1])
                print(f"Time to load raw and preprocessed data: {time.time() - load_start_time:.4f} seconds")

                img_str_start_time = time.time()
                img_str = tu.create_topomap_from_preprocessed(raw, raw_ddf, freq_data["resample_freq"], time_range[0], t)  # Returns base64-encoded string
                print(f"Time to generate topomap image: {time.time() - img_str_start_time:.4f} seconds")

                img_src = f"data:image/png;base64,{img_str}"
                topomap_image = html.Img(src=img_src, style={
                    'height': '160px',  # Consistent height
                    'margin': '0 5px',  # Horizontal spacing between images
                })
                
                return True, topomap_image, dash.no_update

            except Exception as e:
                print(f"⚠️ Error in plot_topomap: {str(e)}")
                return None, dash.no_update, dash.no_update

        return dash.no_update, dash.no_update, dash.no_update
        
def register_activate_deactivate_topomap_button():
    @callback(
        Output('plot-topomap-button', 'outline'),
        Input('plot-topomap-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def _activate_deactivate_topomap_button(n_clicks):
        if n_clicks%2==0:
            return True
        return False     