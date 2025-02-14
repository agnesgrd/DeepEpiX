import base64
import numpy as np
import json

def serialize_array(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

def deserialize_array(encoded, shape):
    return np.frombuffer(base64.b64decode(encoded), dtype=np.float32).reshape(shape)
