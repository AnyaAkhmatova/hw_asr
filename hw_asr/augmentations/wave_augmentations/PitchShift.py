from torch import Tensor
from torch import from_numpy
import torch

from numpy.random import uniform
from pyrubberband import pitch_shift

from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, sr, min_n_steps, max_n_steps):
        self.sr = sr
        self.min_n_steps = min_n_steps
        self.max_n_steps =  max_n_steps + 1e-5

    def __call__(self, data: Tensor):
        n_steps = uniform(low=self.min_n_steps, high=self.max_n_steps)
        return from_numpy(pitch_shift(data.reshape(-1).numpy(), self.sr, n_steps)).type(torch.float32).reshape(1, -1)
