import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import sys
from pathlib import Path
from source.base import BaseTrainer
from source.utils import inf_loop, MetricTracker
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.utils.spec_utils import crop_center


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            log_step,
            train_dataloader,
            val_dataloader,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        

        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        self.step = 0
        self.eval_interval = self.config.trainer.eval_interval
        self.accum_step = self.config.trainer.accum_step
        self.train_metrics = MetricTracker("l1_loss", "grad_norm")
        self.evaluation_metrics = MetricTracker("l1_loss_val")


    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        if is_train:
            mask = self.model(X)
            pred = X * mask
            l1_loss = self.criterion.l1(pred, y)
            accum_loss = l1_loss / self.accum_step
            accum_loss.backward()

            if self.step % self.accum_step == 0:
                #self._clip_grad_norm()
                self.optimizer.zero_grad()
                self.optimizer.step()

            self.step = self.step + 1

            return l1_loss.item() * len(X)

        else:
            pred = self.model.predict(X)

            y = crop_center(y, pred)

            l1_loss = self.criterion.l1(pred, y)

            return l1_loss.item() * len(X)
    

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        sum_loss_l1 = 0
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                l1_loss = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
                sum_loss_l1 += l1_loss
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            #self.train_metrics.update("grad_norm", self.get_grad_norm())
            
            if batch_idx >= self.len_epoch:
                break

        sum_loss_l1 = sum_loss_l1 / len(self.train_dataloader.dataset)
        self.train_metrics.update("l1_loss", sum_loss_l1)
        self.writer.set_step(self.step)
        self.logger.debug(
            "Train Epoch: {} {} l1_loss: {:.6f}".format(
                epoch, self._progress(self.step), sum_loss_l1)
        )
        self.writer.add_scalar(
            "learning rate", self.lr_scheduler.get_last_lr()[0]
        )
        self._log_scalars(self.train_metrics)
        # we don't want to reset train metrics at the start of every epoch
        # because we are interested in recent train metrics
        last_train_metrics = self.train_metrics.result()
        self.train_metrics.reset()

        log = last_train_metrics

        if epoch % self.eval_interval == 0:
            val_log = self._evaluation_epoch(epoch, dataloader=self.val_dataloader)
            log.update(**{f"{name}": value for name, value in val_log.items()})  

        return log
    

    def _evaluation_epoch(self, epoch, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        sum_loss_l1 = 0
        sum_loss_sdr = 0
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader), desc="val", total=len(dataloader)):
                l1_loss, sdr_loss = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
                sum_loss_l1 += l1_loss
                sum_loss_sdr += sdr_loss

        sum_loss_l1 = sum_loss_l1 / len(dataloader.dataset)
        sum_loss_sdr = sum_loss_sdr / len(dataloader.dataset)

        self.writer.set_step(self.step)
        self.evaluation_metrics.update("l1_loss_val", sum_loss_l1)
        self.evaluation_metrics.update("sdr_loss_val", sum_loss_sdr)
        self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()
    

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        # for tensor_for_gpu in ["audio", "bonafied"]:
        #     if tensor_for_gpu in batch:
        #         batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        for tensor in batch:
            tensor = tensor.to(device)
        return batch


    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )


    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    
    def _from_pretrained(self, pretrained_path):
        """
        Start from saved checkpoints

        :param pretrained_path: Checkpoint path to be resumed
        """
        pretrained_path = str(pretrained_path)
        self.logger.info("Loading checkpoint: {} ...".format(pretrained_path))
        checkpoint = torch.load(pretrained_path, self.device)
        #self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        # if checkpoint["config"]["arch"] != self.config["arch"]:
        #     self.logger.warning(
        #         "Warning: Architecture configuration given in config file is different from that "
        #         "of checkpoint. This may yield an exception while state_dict is being loaded."
        #     )
        
        # load optimizer state (accumulated gradients)
        self.model.load_state_dict(checkpoint)

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )