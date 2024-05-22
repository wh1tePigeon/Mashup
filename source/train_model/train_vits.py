import warnings
import sys
import hydra
import numpy as np
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from hydra.utils import instantiate
from omegaconf import DictConfig
from source.trainer.trainer_vits import Trainer
from source.datasets.vits.vits_dataloaders import create_dataloader_train, create_dataloader_eval
from source.utils.util import get_logger, prepare_device, CONFIGS_PATH
from source.utils.object_loading import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

CONFIG_VITS_PATH = CONFIGS_PATH / 'vits'

def get_params_count(model_):
    model_parameters = filter(lambda p: p.requires_grad, model_.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

@hydra.main(config_path=str(CONFIG_VITS_PATH), config_name="main")
def train(cfg: DictConfig):
    train_dataloader = create_dataloader_train(cfg["data"], cfg["n_gpu"], 0)
    val_dataloader = create_dataloader_eval(cfg["data"])

    model = instantiate(cfg["arch"])
    logger = get_logger("train")
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # loss_module = instantiate(cfg["loss"]).to(device)
    metrics = []

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler

    gen_trainable_params = filter(lambda p: p.requires_grad, model.gen.parameters())
    gen_optimizer = instantiate(cfg["optimizer_g"], torch.optim, gen_trainable_params)
    gen_lr_scheduler = instantiate(cfg["scheduler_g"], torch.optim.lr_scheduler, gen_optimizer)
    logger.info(f"Generator params count: {get_params_count(model.gen)}")

    disc_trainable_params = list(model.msd.parameters()) + list(model.mpds.parameters())
    disc_optimizer = instantiate(cfg["optimizer_d"], torch.optim, disc_trainable_params)
    disc_lr_scheduler = instantiate(cfg["scheduler_d"], torch.optim.lr_scheduler, disc_optimizer)
    logger.info(f"MPDs params count: {get_params_count(model.mpds)}")
    logger.info(f"MSD params count: {get_params_count(model.msd)}")

    trainer = Trainer(
        model,
        None,
        metrics,
        gen_optimizer,
        disc_optimizer,
        config=cfg,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        gen_lr_scheduler=gen_lr_scheduler,
        disc_lr_scheduler=disc_lr_scheduler,
        len_epoch=cfg["trainer"].get("len_epoch", None),
    )

    trainer.train()



if __name__ == "__main__":
    train()