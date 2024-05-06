from typing import List, Callable, Tuple

from torch import Tensor


class SequentialAugmentation:
    def __init__(self, augmentation_list: List[Callable]):
        self.augmentation_list = augmentation_list

    def __call__(self, data: Tensor) -> Tuple[Tensor, List[str]]:
        x = data
        aug_names = []
        for augmentation in self.augmentation_list:
            x, names = augmentation(x)
            aug_names.extend(names)
        return x, aug_names
