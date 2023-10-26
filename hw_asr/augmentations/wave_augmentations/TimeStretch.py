from torch import Tensor
from torch import from_numpy
import torch

from numpy.random import uniform
from pyrubberband import time_stretch

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, sr, min_rate, max_rate):
        self.sr = sr
        self.min_rate = min_rate
        self.max_rate =  max_rate + 1e-5

    def __call__(self, data: Tensor):
        rate = uniform(low=self.min_rate, high=self.max_rate)
        return from_numpy(time_stretch(data.reshape(-1).numpy(), self.sr, rate)).type(torch.float32).reshape(1, -1)
