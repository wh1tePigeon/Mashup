import torch_audiomentations
import torch

from ..base import AugmentationBase


class Audiomentation(AugmentationBase):
    """
    This class implements any augmentation from audiomentations library
    """
    CLASS_NAME = 'NONE'

    def __init__(self, p: float, *args, **kwargs):
        super().__init__(p)
        self._aug = getattr(torch_audiomentations, self.CLASS_NAME)(*args, **kwargs, p=1)

    def forward(self, data: torch.Tensor):
        return self._aug(data.unsqueeze(1)).squeeze(1), [self.CLASS_NAME]


class Gain(Audiomentation):
    CLASS_NAME = 'Gain'


class AddColoredNoise(Audiomentation):
    CLASS_NAME = 'AddColoredNoise'


class PitchShift(Audiomentation):
    CLASS_NAME = 'PitchShift'
