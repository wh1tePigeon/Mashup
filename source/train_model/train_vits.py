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
SEED = 1234
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
    train_dataloader = create_dataloader_train(cfg.dataset, cfg.n_gpu, 0)
    val_dataloader = create_dataloader_eval(cfg.dataset)

    cfg.gen.spec_channels = cfg.dataset.filter_length // 2 + 1
    cfg.gen.segment_size = cfg.dataset.segment_size // cfg.dataset.hop_length
    cfg.gen.hp.data.sampling_rate = cfg.dataset.sampling_rate
    gen = instantiate(cfg.gen)
    disc = instantiate(cfg.disc)
    logger = get_logger("train")
    logger.info(gen)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg.n_gpu)
    gen = gen.to(device)
    disc = disc.to(device)
    if len(device_ids) > 1:
        gen = torch.nn.DataParallel(gen, device_ids=device_ids)
        disc = torch.nn.DataParallel(disc, device_ids=device_ids)

    # get function handles of loss and metrics
    cfg.loss.device = device.type
    loss_module = instantiate(cfg.loss).to(device)
    metrics = []

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    gen_trainable_params = filter(lambda p: p.requires_grad, gen.parameters())
    gen_optimizer = instantiate(cfg.optimizer_g, gen_trainable_params)
    gen_lr_scheduler = instantiate(cfg.scheduler_g, gen_optimizer)
    logger.info(f"Generator params count: {get_params_count(gen)}")

    disc_trainable_params = disc.parameters()
    disc_optimizer = instantiate(cfg.optimizer_d, disc_trainable_params)
    disc_lr_scheduler = instantiate(cfg.scheduler_d, disc_optimizer)
    logger.info(f"Discriminator params count: {get_params_count(disc)}")

    trainer = Trainer(
        gen,
        disc,
        loss_module,
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