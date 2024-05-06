import random
from typing import Tuple, List
import torch

class AugmentationBase:
    """
    Class for applying any augmentation randomly
    """

    def __init__(self, p: float) -> None:
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, data: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        if random.random() < self.p:
            return self.forward(data)
        else:
            return data, []

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        raise NotImplementedError()
