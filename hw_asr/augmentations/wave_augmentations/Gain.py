from torch import Tensor
from numpy.random import uniform
import torch

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val =  max_val

    def __call__(self, data: Tensor):
        out = data * uniform(low=self.min_val, high=self.max_val)
        return torch.clamp(out, min=-1, max=1)

