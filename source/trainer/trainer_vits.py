import sys
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from source.base import BaseTrainer
from source.utils import inf_loop, MetricTracker
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.model.vits.modules.commons import slice_segments
from source.utils.spec_utils import mel_spectrogram


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
        cfg,
        device,
        train_dataloader,
        val_dataloader,
        gen_lr_scheduler,
        disc_lr_scheduler,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(gen, criterion, metrics, None, None, cfg, device)
        self.skip_oom = skip_oom
        self.cfg = cfg

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

        self.loss_names = ["disc_loss", "gen_loss", "stft_loss", "mel_loss", "loss_kl_f", "loss_kl_r", "spk_loss"]
        self.train_metrics = MetricTracker(*self.loss_names, "Gen grad_norm", "MPDs grad_norm", "MSD grad_norm")
        self.evaluation_metrics = []#MetricTracker(*self.loss_names)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["ppg", "ppg_l", "vec", "pit", "spk", "spec", "spec_l", "audio", "audio_l"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    # @torch.no_grad()
    # def _log_predictions(self, pred, target, examples_to_log=3, **kwargs):
    #     rows = {}
    #     i = 0
    #     for pred, target in zip(pred, target):
    #         if i >= examples_to_log:
    #             break
    #         rows[i] = {
    #             "pred": self.writer.wandb.Audio(pred.cpu().squeeze().numpy(), sample_rate=DEFAULT_SR),
    #             "target": self.writer.wandb.Audio(target.cpu().squeeze().numpy(), sample_rate=DEFAULT_SR),
    #         }
    #         i += 1

    #     self.writer.add_table("logs", pd.DataFrame.from_dict(rows, orient="index"))

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        ppg, ppg_l, vec, pit, spk, spec, spec_l, audio, audio_l = batch
        if is_train:
            # generator
            fake_audio, ids_slice, z_mask, \
                (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds = self.model(
                    ppg, vec, pit, spec, spk, ppg_l, spec_l)

            audio = slice_segments(
                audio, ids_slice * self.cfg["data"]["hop_length"], self.cfg["data"]["segment_size"])  # slice
            # Spk Loss
            batch["spk_loss"] = self.criterion.spk_loss(spk, spk_preds, torch.Tensor(spk_preds.size(0))
                                .to(self.device).fill_(1.0))
            # Mel Loss
            mel_fake = mel_spectrogram(self.cfg["mel"], fake_audio.squeeze(1))
            mel_real = mel_spectrogram(self.cfg["mel"], audio.squeeze(1))
            batch["mel_loss"] = self.criterion.l1(mel_fake, mel_real) * self.cfg["train"]["c_mel"]

            # Multi-Resolution STFT Loss
            sc_loss, mag_loss = self.criterion.stft_loss(fake_audio.squeeze(1), audio.squeeze(1))
            batch["stft_loss"] = (sc_loss + mag_loss) *  self.cfg["train"]["c_stft"]

            # Generator Loss
            disc_fake = self.disc(fake_audio)
            batch["gen_loss"], _ = self.criterion.generator_loss(disc_fake)

            # Feature Loss
            disc_real = self.disc(audio)
            batch["feat_loss"] = self.criterion.feature_loss(disc_fake, disc_real)

            # Kl Loss
            batch["loss_kl_f"] = self.criterion.kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * self.cfg["train"]["c_kl"]
            batch["loss_kl_r"] = self.criterion.kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * self.cfg["train"]["c_kl"]

            # Loss
            batch["loss_g"] = batch["gen_loss"] + batch["feat_loss"] + batch["mel_loss"] + batch["stft_loss"] + \
                batch["loss_kl_f"] + batch["loss_kl_r"] * 0.5 + batch["spk_loss"] * 2
            batch["loss_g"].backward()
            #loss_g = gen_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + loss_kl_r * 0.5 + spk_loss * 2
            #loss_g.backward()

            if ((step + 1) % self.cfg["train"]["accum_step"] == 0) or (step + 1 == len(loader)):
                # accumulate gradients for accum steps
                for param in self.model.parameters():
                    param.grad /= self.cfg["train"]["accum_step"]
                self._clip_grad_norm(self.model)
                # update model
                self.gen_optimizer.step()
                self.gen_optimizer.zero_grad()

            # discriminator
            self.disc_optimizer.zero_grad()
            disc_fake = self.disc(fake_audio.detach())
            disc_real = self.disc(audio)

            batch["disc_loss"] = self.criterion.discriminator_loss(disc_real, disc_fake)
            batch["disc_loss"].backward()
            self._clip_grad_norm(self.disc)
            self.disc_optimizer.step()


            # # discriminator
            # self.disc_optimizer.zero_grad()
            # batch.update(self.model.disc_forward(batch["pred"].detach(), batch["target"]))
            # disc_loss = self.criterion.disc(**batch)
            # batch.update(disc_loss)
            # batch["disc_loss"].backward()
            # self._clip_grad_norm(self.model.mpds)
            # self._clip_grad_norm(self.model.msd)
            # self.disc_optimizer.step()
            # self.train_metrics.update("MPDs grad_norm", self.get_grad_norm(self.model.mpds))
            # self.train_metrics.update("MSD grad_norm", self.get_grad_norm(self.model.msd))

            # # generator
            # batch.update(self.model.disc_forward(**batch))
            # self.gen_optimizer.zero_grad()
            # gen_loss = self.criterion.gen(**batch)
            # batch.update(gen_loss)
            # batch["gen_loss"].backward()
            # self._clip_grad_norm(self.model.gen)
            # self.gen_optimizer.step()
            # self.train_metrics.update("Gen grad_norm", self.get_grad_norm(self.model.gen))

            for loss_name in self.loss_names:
                metrics.update(loss_name, batch[loss_name].item())

        for metric in self.metrics:
            metrics.update(metric.name, metric(**batch))

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, False, metrics=self.evaluation_metrics)

            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_predictions(**batch)
            # self._log_spectrogram(batch["spectrogram"])
            self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        bar = tqdm(range(self.len_epoch), desc="train")
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                batch = self.process_batch(batch, True, metrics=self.train_metrics)
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

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Gen loss: {:.6f} Disc loss: {:.6f} Mel loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["gen_loss"].item(), batch["disc_loss"].item(), batch["loss_mel"].item()
                    )
                )
                self.writer.add_scalar("disc learning rate", self.disc_lr_scheduler.get_last_lr()[0])
                self.writer.add_scalar("gen learning rate", self.gen_lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                self._log_predictions(**batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                bar.update(self.log_step)

            if batch_idx + 1 >= self.len_epoch:
                break

        self.gen_lr_scheduler.step()
        self.disc_lr_scheduler.step()

        log = last_train_metrics
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log