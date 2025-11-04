from pathlib import Path

# App settings ##########
DEBUG = True
PORT = 8050
HOST = "127.0.0.1"
ENV = "development"

# Cache settings ##########
CACHE_TYPE = "FileSystemCache"
CACHE_DEFAULT_TIMEOUT = 84000  # seconds

# Useful path ##########
# Root directory of the app (thanks to PYTHONPATH)
APP_ROOT = Path(__file__).parent

# Common paths used everywhere
MODELS_DIR = APP_ROOT / "models"
CACHE_DIR = APP_ROOT / "cache-directory"
DATA_DIR = APP_ROOT.parent / "data"  # assuming data/ is at /DeepEpiX/data
STATIC_DIR = APP_ROOT / "static"
MODEL_PIPELINE_DIR = APP_ROOT / "model_pipeline"
TENSORFLOW_ENV = APP_ROOT.parent / ".tfenv"
TORCH_ENV = APP_ROOT.parent / ".torchenv"
MONTAGES_DIR = STATIC_DIR / "montages"

# Default plotting variables #####
DEFAULT_Y_AXIS_OFFSET = 40
DEFAULT_SEGMENT_SIZE = 10
CHUNK_RECORDING_DURATION = 120  # 2 minutes per chunk (120 seconds)
