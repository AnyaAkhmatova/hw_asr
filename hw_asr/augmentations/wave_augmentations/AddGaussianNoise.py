from torch import Tensor
from torch.distributions.normal import Normal

from hw_asr.augmentations.base import AugmentationBase


class AddGaussianNoise(AugmentationBase):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.distr = Normal(self.loc, self.scale)

    def __call__(self, data: Tensor):
        return data + self.distr.sample(data.size())
