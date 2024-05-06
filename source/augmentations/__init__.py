from typing import List, Callable

import source.augmentations.spectrogram_augmentations
import source.augmentations.wave_augmentations
from source.augmentations.sequential import SequentialAugmentation
from omegaconf import DictConfig
from hydra.utils import instantiate


def from_configs(config: DictConfig):
    wave_augs = []
    if "augmentations" in config and "wave" in config["augmentations"]:
        for aug_dict in config["augmentations"]["wave"]:
            aug_dict = aug_dict.copy()
            aug_dict["args"]["sample_rate"] = config["preprocessing"]["sr"]
            wave_augs.append(instantiate(aug_dict))

    spec_augs = []
    if "augmentations" in config and "spectrogram" in config["augmentations"]:
        for aug_dict in config["augmentations"]["spectrogram"]:
            spec_augs.append(instantiate(aug_dict))

    return _to_function(wave_augs), _to_function(spec_augs)


def _to_function(augs_list: List[Callable]):
    if len(augs_list) == 0:
        return None
    elif len(augs_list) == 1:
        return augs_list[0]
    else:
        return SequentialAugmentation(augs_list)
