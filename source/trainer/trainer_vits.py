import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import random
import PIL
from tqdm import tqdm
from source.base import BaseTrainer
from source.utils import inf_loop, MetricTracker
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.model.vits.modules.commons import slice_segments
from source.utils.spec_utils import mel_spectrogram
from source.logger.utils import plot_spectrogram_to_buf
from torchvision.transforms import ToTensor


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        gen,
        disc,
        criterion,
        metrics,
        gen_optimizer,
        disc_optimizer,
        config,
        device,
        train_dataloader,
        val_dataloader,
        gen_lr_scheduler,
        disc_lr_scheduler,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(gen, criterion, metrics, None, None, config, device)
        self.skip_oom = skip_oom
        self.cfg = config

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.disc = disc

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch

        if self.cfg.trainer.log_step is None:
            self.log_step = self.len_epoch - 1

        self.step = 0
        self.eval_interval = self.cfg.trainer.eval_interval
        self.accum_step = self.cfg.trainer.accum_step
        self.loss_names = ["disc_loss", "gen_loss", "stft_loss", "mel_loss", "loss_kl_f", "loss_kl_r", "spk_loss", "loss_g", "feat_loss"]
        self.train_metrics = MetricTracker(*self.loss_names, "Gen grad_norm", "Disc grad_norm")
        self.evaluation_metrics = MetricTracker("f1_mel_loss_val")
        #self.metrics = MetricTracker()


    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'gen': (self.model.module if self.cfg.n_gpu > 1 else self.model).state_dict(),
            'disc': (self.disc.module if self.cfg.n_gpu > 1 else self.disc).state_dict(),
            'optim_g': self.gen_optimizer.state_dict(),
            'optim_d': self.disc_optimizer.state_dict(),
            'scheduler_g' : self.gen_lr_scheduler.state_dict(),
            'scheduler_d' : self.disc_lr_scheduler.state_dict(),
            'step': self.step,
            'epoch': epoch,
            "config": self.config,
            "monitor_best": self.mnt_best
            }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = os.path.join(self.checkpoint_dir, f"checkpoint-{self.cfg.gen.hp.vocoder_name}-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _from_pretrained(self, pretrained_path):
        """
        Start from saved checkpoints

        :param pretrained_path: Checkpoint path to be resumed
        """
        pretrained_path = str(pretrained_path)
        self.logger.info("Loading checkpoint: {} ...".format(pretrained_path))
        checkpoint = torch.load(pretrained_path, self.device)
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["gen"] != self.config["gen"] or \
            checkpoint["config"]["disc"] != self.config["disc"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        
        # load optimizer state (accumulated gradients)
        self.model.load_state_dict(checkpoint["gen"]["state_dict"])
        self.disc.load_state_dict(checkpoint["disc"]["state_dict"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        self.step = checkpoint["step"] + 1

        # load architecture params from checkpoint.
        if checkpoint["config"]["gen"] != self.config["gen"] or \
            checkpoint["config"]["disc"] != self.config["disc"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["gen"]["state_dict"])
        self.disc.load_state_dict(checkpoint["disc"]["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["optimizer_d"] != self.config["optimizer_d"] or
                checkpoint["config"]["scheduler_d"] != self.config["scheduler_d"] or
                checkpoint["config"]["optimizer_g"] != self.config["optimizer_g"] or
                checkpoint["config"]["scheduler_g"] != self.config["scheduler_g"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.gen_optimizer.load_state_dict(checkpoint["optimizer_g"])
            self.disc_optimizer.load_state_dict(checkpoint["optimizer_d"])
            if self.gen_lr_scheduler is not None:
                self.gen_lr_scheduler.load_state_dict(checkpoint["scheduler_g"])
            if self.disc_lr_scheduler is not None:
                self.disc_lr_scheduler.load_state_dict(checkpoint["scheduler_g"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

        
    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["ppg", "ppg_l", "vec", "pit", "spk", "spec", "spec_l", "audio", "audio_l"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch


    @torch.no_grad()
    def _log_predictions(self, pred, target, examples_to_log=1, **kwargs):
        rows = {}
        i = 0
        for pred, target in zip(pred, target):
            if i >= examples_to_log:
                break
            rows[i] = {
                "fake_audio": self.writer.wandb.Audio(pred.cpu().squeeze().numpy(), sample_rate=self.cfg.dataset.sampling_rate),
                "real_audio": self.writer.wandb.Audio(target.cpu().squeeze().numpy(), sample_rate=self.cfg.dataset.sampling_rate),
            }
            i += 1

        self.writer.add_table("logs", pd.DataFrame.from_dict(rows, orient="index"))


    def _log_spectrogram(self, mel_fake, mel_real):
        i = random.randint(0, mel_fake.shape[0] - 1)
        mel_fake_log = mel_fake[i].to("cpu")
        mel_real_log = mel_real[i].to("cpu")
        image_fake = PIL.Image.open(plot_spectrogram_to_buf(mel_fake_log))
        image_real = PIL.Image.open(plot_spectrogram_to_buf(mel_real_log))
        self.writer.add_image("fake_mel_val", ToTensor()(image_fake))
        self.writer.add_image("real_mel_val", ToTensor()(image_real))


    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        
        if is_train:
            #self.gen_optimizer.zero_grad()

            # generator
            batch["fake_audio"], ids_slice, z_mask, \
                (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds = self.model(
                    batch["ppg"], batch["vec"], batch["pit"], batch["spec"], batch["spk"], batch["ppg_l"], batch["spec_l"])

            batch["audio"] = slice_segments(
                batch["audio"], ids_slice * self.cfg["dataset"]["hop_length"], self.cfg["dataset"]["segment_size"])  # slice
            # Spk Loss
            batch["spk_loss"] = self.criterion.spk_loss(batch["spk"], spk_preds, torch.Tensor(spk_preds.size(0))
                                .to(self.device).fill_(1.0))
            # Mel Loss
            self.cfg.mel.device = self.device.type
            mel_fake = mel_spectrogram(self.cfg.mel, batch["fake_audio"].squeeze(1))
            mel_real = mel_spectrogram(self.cfg.mel, batch["audio"].squeeze(1))
            batch["mel_loss"] = self.criterion.l1(mel_fake, mel_real) * self.cfg.trainer.c_mel

            # Multi-Resolution STFT Loss
            sc_loss, mag_loss = self.criterion.stft_loss(batch["fake_audio"].squeeze(1), batch["audio"].squeeze(1))
            batch["stft_loss"] = (sc_loss + mag_loss) *  self.cfg.trainer.c_stft

            # Generator Loss
            disc_fake = self.disc(batch["fake_audio"])
            batch["gen_loss"], _ = self.criterion.generator_loss(disc_fake)

            # Feature Loss
            disc_real = self.disc(batch["audio"])
            batch["feat_loss"] = self.criterion.feature_loss(disc_fake, disc_real)

            # Kl Loss
            batch["loss_kl_f"] = self.criterion.kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * self.cfg.trainer.c_kl
            batch["loss_kl_r"] = self.criterion.kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * self.cfg.trainer.c_kl

            # Loss
            batch["loss_g"] = batch["gen_loss"] + batch["feat_loss"] + batch["mel_loss"] + batch["stft_loss"] + \
                batch["loss_kl_f"] + batch["loss_kl_r"] * 0.5 + batch["spk_loss"] * 2
            
            batch["loss_g"].backward()
            #self.clip_grad_value_(self.model.parameters(),  None)
            #self.gen_optimizer.step()
            

            if ((self.step + 1) % self.accum_step == 0):
                # accumulate gradients for accum steps
                for param in self.model.parameters():
                    param.grad /= self.accum_step
                # update model
                self.clip_grad_value_(self.model.parameters(),  None)
                # update model
                self.gen_optimizer.step()
                self.gen_optimizer.zero_grad()
                

            # discriminator
            self.disc_optimizer.zero_grad()
            disc_fake = self.disc(batch["fake_audio"].detach())
            disc_real = self.disc(batch["audio"])
            batch["disc_loss"] = self.criterion.discriminator_loss(disc_real, disc_fake)

            batch["disc_loss"].backward()
            self.clip_grad_value_(self.disc.parameters(),  None)
            self.disc_optimizer.step()
            self.step = self.step + 1

            for loss_name in self.loss_names:
                metrics.update(loss_name, batch[loss_name].item())

            

        else:
            if hasattr(self.model, 'module'):
                batch["fake_audio"] = self.model.module.infer(batch["ppg"], batch["vec"], batch["pit"],
                                                     batch["spk"], batch["ppg_l"])[
                    :, :, :batch["audio"].size(2)]
            else:
                batch["fake_audio"] = self.model.infer(batch["ppg"], batch["vec"], batch["pit"],
                                                     batch["spk"], batch["ppg_l"])[
                    :, :, :batch["audio"].size(2)]
                
            self.cfg.mel.device = self.device.type
            batch["mel_fake"] = mel_spectrogram(self.cfg.mel, batch["fake_audio"].squeeze(1))
            batch["mel_real"] = mel_spectrogram(self.cfg.mel, batch["audio"].squeeze(1))
            batch["mel_loss"] = self.criterion.l1(batch["mel_fake"], batch["mel_real"])

        return batch

    def _evaluation_epoch(self, epoch, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            mel_loss = 0
            for _, batch in tqdm(enumerate(dataloader), desc="val", total=len(dataloader),):
                batch = self.process_batch(batch, False, metrics=self.evaluation_metrics)
                mel_loss += batch["mel_loss"].item()

            mel_loss = mel_loss / len(dataloader)

            self.writer.set_step(self.step)
            self.evaluation_metrics.update("f1_mel_loss_val", mel_loss)
            self._log_predictions(batch["fake_audio"], batch["audio"])
            self._log_spectrogram(batch["mel_fake"], batch["mel_real"])
            self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.disc.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        self.train_dataloader.batch_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            try:
                batch = self.process_batch(batch, True, metrics=self.train_metrics)
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.disc.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                
            self.train_metrics.update("Disc grad_norm", self.get_grad_norm(self.disc))
            self.train_metrics.update("Gen grad_norm", self.get_grad_norm(self.model))
            if self.step % self.log_step == 0:
                self.writer.set_step(self.step)
                self.logger.debug(
                    "Train Epoch: {} {} Gen loss: {:.6f} Disc loss: {:.6f} loss_g: {:.6f}".format(
                        epoch, self._progress(self.step), batch["gen_loss"].item(), batch["disc_loss"].item(), batch["loss_g"].item()
                    )
                )
                self.writer.add_scalar("disc learning rate", self.disc_lr_scheduler.get_last_lr()[0])
                self.writer.add_scalar("gen learning rate", self.gen_lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                self._log_predictions(batch["fake_audio"], batch["audio"])
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 >= self.len_epoch:
                break

        self.gen_lr_scheduler.step()
        self.disc_lr_scheduler.step()

        log = last_train_metrics
        # for part, dataloader in self.evaluation_dataloaders.items():
        #     val_log = self._evaluation_epoch(epoch, part, dataloader)
        #     log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
        if epoch % self.eval_interval == 0:
            val_log = self._evaluation_epoch(epoch, dataloader=self.val_dataloader)
            log.update(**{f"{name}": value for name, value in val_log.items()})        
        return log
    

    def clip_grad_value_(self, parameters, clip_value, norm_type=2):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(norm_type)
        if clip_value is not None:
            clip_value = float(clip_value)

        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            if clip_value is not None:
                p.grad.data.clamp_(min=-clip_value, max=clip_value)
        total_norm = total_norm ** (1.0 / norm_type)
        return total_norm