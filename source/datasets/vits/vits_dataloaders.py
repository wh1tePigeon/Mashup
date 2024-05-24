import sys
from torch.utils.data import DataLoader
from .vits_dataset import TextAudioSpeakerSet
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.batch_sampler.vits.vits_bs import DistributedBucketSampler
from source.collate_fn.vits.vits_cf import TextAudioSpeakerCollate


def create_dataloader_train(data_cfg, n_gpu, rank):
    collate_fn = TextAudioSpeakerCollate()
    train_dataset = TextAudioSpeakerSet(data_cfg["training_files"], data_cfg)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        data_cfg["batch_size"],
        [150, 300, 450],
        num_replicas=n_gpu,
        rank=rank,
        shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler)
    return train_loader


def create_dataloader_eval(data_cfg):
    collate_fn = TextAudioSpeakerCollate()
    eval_dataset = TextAudioSpeakerSet(data_cfg["validation_files"], data_cfg)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=data_cfg["batch_size"],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)
    return eval_loader