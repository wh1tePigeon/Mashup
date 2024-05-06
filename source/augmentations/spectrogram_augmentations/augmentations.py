import random

import torchaudio.transforms as T
import torch

from ..base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, p, min_coef: float, max_coef: float, *args, **kwargs):
        super().__init__(p)
        self.min_coef = min_coef
        self.max_coef = max_coef
        self._aug = T.TimeStretch(*args, **kwargs, n_freq=128)

    def forward(self, data: torch.Tensor):
        return self._aug(data, random.uniform(self.min_coef, self.max_coef)).abs(), ['TimeStretch']
