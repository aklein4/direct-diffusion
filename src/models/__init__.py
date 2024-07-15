
from models.dit import DiTConfig, DiT
from models.rsdit import RSDiTConfig, RSDiT


CONFIG_DICT = {
    "dit": DiTConfig,
    "rsdit": RSDiTConfig,
}

MODEL_DICT = {
    "dit": DiT,
    "rsdit": RSDiT,
}
