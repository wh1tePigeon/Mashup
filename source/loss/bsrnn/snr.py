import torch
from torch.nn.modules.loss import _Loss

class SignalNoisePNormRatio(_Loss):
    def __init__(
            self,
            p: float = 1.0,
            scale_invariant: bool = False,
            zero_mean: bool = False,
            take_log: bool = True,
            reduction: str = "mean",
            EPS: float = 1e-3,
    ) -> None:
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)
        assert not zero_mean

        self.p = p

        self.EPS = EPS
        self.take_log = take_log

        self.scale_invariant = scale_invariant

    def forward(
            self,
            est_target: torch.Tensor,
            target: torch.Tensor
            ) -> torch.Tensor:

        target_ = target
        if self.scale_invariant:
            ndim = target.ndim
            dot = torch.sum(est_target * torch.conj(target), dim=-1, keepdim=True)
            s_target_energy = (
                    torch.sum(target * torch.conj(target), dim=-1, keepdim=True)
            )

            if ndim > 2:
                dot = torch.sum(dot, dim=list(range(1, ndim)), keepdim=True)
                s_target_energy = torch.sum(s_target_energy, dim=list(range(1, ndim)), keepdim=True)

            target_scaler = (dot + 1e-8) / (s_target_energy + 1e-8)
            target = target_ * target_scaler

        if torch.is_complex(est_target):
            est_target = torch.view_as_real(est_target)
            target = torch.view_as_real(target)


        batch_size = est_target.shape[0]
        est_target = est_target.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        # target_ = target_.reshape(batch_size, -1)

        if self.p == 1:
            e_error = torch.abs(est_target-target).mean(dim=-1)
            e_target = torch.abs(target).mean(dim=-1)
        elif self.p == 2:
            e_error = torch.square(est_target-target).mean(dim=-1)
            e_target = torch.square(target).mean(dim=-1)
        else:
            raise NotImplementedError
        
        if self.take_log:
            loss = 10*(torch.log10(e_error + self.EPS) - torch.log10(e_target + self.EPS))
        else:
            loss = (e_error + self.EPS)/(e_target + self.EPS)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss