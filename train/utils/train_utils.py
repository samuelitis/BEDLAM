import os
import torch
from loguru import logger
import pytorch_lightning as pl
from collections import OrderedDict

def parse_datasets_ratios(datasets_and_ratios):
    s_ = datasets_and_ratios.split('_')
    r = [float(x) for x in s_[len(s_) // 2:]]
    d = s_[:len(s_) // 2]
    return d + r

def set_seed(seed_value):
    if seed_value >= 0:
        logger.info(f'Seed value for the experiment {seed_value}')
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        pl.trainer.seed_everything(seed_value)