# config.py

##### App settings ##########
DEBUG = True
PORT = 8050
HOST = "127.0.0.1"
ENV = "development"

##### Cache settings ##########
CACHE_TYPE = 'FileSystemCache'
CACHE_DIR = 'cache-directory'
CACHE_DEFAULT_TIMEOUT = 84000 # seconds

##### Useful path ##########
MODEL_DIR = "models/"
TENSORFLOW_ENV = ".tfenv"
TORCH_ENV = ".torchenv"

##### Default plotting variables #####
DEFAULT_Y_AXIS_OFFSET = 40
DEFAULT_SEGMENT_SIZE = 10
CHUNK_RECORDING_DURATION = 120 # 2 minutes per chunk (120 seconds)