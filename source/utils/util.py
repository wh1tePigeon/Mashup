import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from omegaconf import DictConfig
import logging
import pandas as pd
import torch

SOURCE_PATH = Path(__file__).absolute().resolve().parent.parent
ROOT_PATH = SOURCE_PATH.parent
CONFIGS_PATH = SOURCE_PATH / 'configs'
CHECKPOINTS_DEFAULT_PATH = ROOT_PATH / 'checkpoints'
OUTPUT_DEFAULT_PATH = ROOT_PATH / 'output'

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()

def get_logger(name, verbosity=2):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
        verbosity, log_levels.keys()
    )
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger


def resolve_paths(cfg: DictConfig, root: str):
    for key, part in cfg.items():
        if isinstance(part, DictConfig):
            cfg[key] = resolve_paths(part, root)
        elif part is not None and type(part) == str and "$ROOT" in part:
            cfg[key] = part.replace("$ROOT", root)
    return cfg