import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.audio import SignalDistortionRatio


class CascadedLoss(torch.nn.Module):
    def __init__(self):
        super(CascadedLoss, self).__init__()
        self.sdr = SignalDistortionRatio()
        self.l1 = F.l1_loss

    def l1_loss(self, preds, target):
        loss = self.l1(preds, target)
        return loss

    def sdr_loss(self, preds, target):
        loss = self.sdr(preds, target)
        return loss