""" Training package """

from trainers.xla_direct_trainer import XLADirectTrainer
from trainers.xla_uncond_trainer import XLAUncondTrainer

TRAINER_DICT = {
    "XLADirectTrainer": XLADirectTrainer,
    "XLAUncondTrainer": XLAUncondTrainer,
}
