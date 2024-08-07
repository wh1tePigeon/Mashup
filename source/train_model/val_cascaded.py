import os
import warnings
import sys
import hydra
import numpy as np
import torch
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from hydra.utils import instantiate
from omegaconf import DictConfig
from source.trainer.trainer_cascaded import Trainer
from source.utils.util import get_logger, prepare_device, CONFIGS_PATH, resolve_paths
from source.utils.object_loading import get_dataloaders

from source.datasets.cascaded.dataset import get_dataloaders


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

CONFIG_CASCADED_PATH = CONFIGS_PATH / 'cascaded'

def get_params_count(model_):
    model_parameters = filter(lambda p: p.requires_grad, model_.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

@hydra.main(config_path=str(CONFIG_CASCADED_PATH), config_name="main")
def val(cfg: DictConfig):
    cfg = resolve_paths(cfg, os.environ['ROOT'])
    model = instantiate(cfg.arch)
    logger = get_logger("train")
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg.n_gpu)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    cfg.dataset.val.offset = model.offset
    if cfg.dataset.val.filelist is None:
        val_filelist = []

    elif cfg.dataset.split_mode == "subdirs":
        raise ValueError('`val_filelist` option is not available with `subdirs` mode')
    else:
        with open(cfg.dataset.val.filelist, 'r', newline='') as f:
            val_filelist = []
            reader = csv.reader(f)
            next(reader)  # Пропускаем заголовок
            loaded_file_pairs = [row for row in reader]
            for pair in loaded_file_pairs:
                mix = pair[0].replace("$ROOT", os.environ['ROOT'])
                inst = pair[1].replace("$ROOT", os.environ['ROOT'])
                p = (mix, inst)
                val_filelist.append(p)



    train_dataloader, val_dataloader = get_dataloaders(cfg.dataset, val_filelist)

    # get function handles of loss and metrics
    loss_module = instantiate(cfg.loss).to(device)
    metrics = []

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer, trainable_params)
    scheduler = instantiate(cfg.scheduler, optimizer)
    logger.info(f"Model params count: {get_params_count(model)}")

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer=optimizer,
        config=cfg,
        device=device,
        log_step=cfg["trainer"].get("log_step", 100),
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr_scheduler=scheduler,
        len_epoch=cfg["trainer"].get("len_epoch", None),
    )
    trainer._evaluation_epoch(1, val_dataloader)

    trainer.train()


if __name__ == "__main__":
    os.environ['ROOT'] = "/home/comp/Рабочий стол/Mashup"
    val()