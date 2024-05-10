import torch.nn as nn
import torch
from torchmetrics.audio import SignalDistortionRatio


def compute_sdr(preds: torch.Tensor, target: torch.Tensor):
    sdr =  SignalDistortionRatio()
    return sdr(preds, target)
