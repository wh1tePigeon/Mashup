import warnings
import sys
import hydra
import numpy as np
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from hydra.utils import instantiate
from omegaconf import DictConfig
from source.trainer.trainer_bsrnn import Trainer
from source.utils.util import get_logger, prepare_device, CONFIGS_PATH
from source.utils.object_loading import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

CONFIG_BSRNN_PATH = CONFIGS_PATH / 'bsrnn'

@hydra.main(config_path=str(CONFIG_BSRNN_PATH), config_name="main")
def train(cfg: DictConfig):
    dataloaders = get_dataloaders(cfg["dataset"])

    model = instantiate(cfg["arch"])
    logger = get_logger("train")
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = instantiate(cfg["loss"]).to(device)
    metrics = [
        instantiate(m) for m in cfg["metrics"]
        #config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        #for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg["optimizer"], trainable_params)
    scheduler = instantiate(cfg["scheduler"], optimizer)

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer=optimizer,
        config=cfg,
        device=device,
        log_step=cfg["trainer"].get("log_step", 100),
        dataloaders=dataloaders,
        lr_scheduler=scheduler,
        len_epoch=cfg["trainer"].get("len_epoch", None),
    )

    trainer.train()


if __name__ == "__main__":
    train()