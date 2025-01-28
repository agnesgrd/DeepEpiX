from math import *
import static.constants as c

def update_chunk_limits(total_duration):
    # Define chunk duration (in seconds) and total duration
    chunk_duration = c.CHUNK_RECORDING_DURATION

    # Calculate start and end times for the selected chunk
    chunk_limits = []
    chunk_number = ceil(total_duration/chunk_duration)
    for chunk_idx in range(chunk_number):
        start_time = chunk_idx * chunk_duration
        end_time = min(start_time + chunk_duration, total_duration)
        chunk_limits.append([start_time, end_time])
        
    # Return chunk limits in a dictionary to store in the dcc.Store
    return chunk_limits

