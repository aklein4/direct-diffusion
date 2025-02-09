
import os

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    print("Warning: torch_xla not found", flush=True)
    XLA_AVAILABLE = False

# get the base path of src
BASE_PATH = os.path.dirname( # src
    os.path.dirname( # utils
        __file__ # utils.constants
    )
)

# ddevice of current process
XLA_DEVICE = lambda: xm.xla_device()

# id of the current device
XLA_DEVICE_ID = lambda: xm.get_ordinal()

# number of devices
NUM_XLA_DEVICES = lambda: xm.xrt_world_size()

# whether this is the main process
XLA_MAIN = lambda: xm.is_master_ordinal(local=False)

# local data path
LOCAL_DATA_PATH = os.path.join(BASE_PATH, "local_data")

# paths to config files
MODEL_CONFIG_PATH = os.path.join(BASE_PATH, "model_configs")
TRAIN_CONFIG_PATH = os.path.join(BASE_PATH, "train_configs")

# huggingface login id
HF_ID = "aklein4"

# diffusion model info
IMAGE_SIZE = 512
PATCH_SIZE = 8
LATENT_SIZE = IMAGE_SIZE // PATCH_SIZE
LATENT_DEPTH = PATCH_SIZE * PATCH_SIZE * 3
