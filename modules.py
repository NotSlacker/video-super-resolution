import os
import shutil

from functools import partial

import logging
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import losses
import metrics

logger = logging.getLogger(__name__)


class VsrYuvModule(L.LightningModule):
    def __init__(
        self, model, *, loss_params, metric_params, optimizer_params, **kwargs
    ):
        super().__init__()

        self.model = model

        self.loss_params = loss_params
        self.metric_params = metric_params
        self.optimizer_params = optimizer_params

    def setup(self, stage):
        self.losses = [
            losses.LossWrapper(name, losses.implemented[name]().to(self.device), weight)
            for name, weight in self.loss_params.items()
        ]

        self.metrics = [
            metrics.MetricWrapper(name, metrics.implemented[name]().to(self.device))
            for name in self.metric_params
        ]

    def configure_optimizers(self):
        parameters = self.parameters()

        optimizer = torch.optim.AdamW(parameters, **self.optimizer_params)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
        }

    def forward(self, x):
        y = self.model(x)

        return y

    def calculate_loss(self, sr, hr, prefix, prog_bar=True):
        results = {}

        sr = sr.flatten(1, 2)
        hr = hr.flatten(1, 2)

        total = 0

        for loss in self.losses:
            score = loss.func(sr, hr)
            total = total + score * loss.weight

            results[f"{prefix}/{loss.name}"] = score.item()

        results[f"{prefix}/loss"] = total.item()

        self.log_dict(results, prog_bar=prog_bar, logger=True, add_dataloader_idx=False)

        return results | {"loss": total}

    def calculate_metrics(self, sr, hr, prefix):
        results = {}

        sr = sr.flatten(1, 2).clamp(0, 1)
        hr = hr.flatten(1, 2).clamp(0, 1)

        for metric in self.metrics:
            score = metric.func(sr, hr)

            results[f"{prefix}/{metric.name}"] = score.item()

        self.log_dict(results, prog_bar=False, logger=True, add_dataloader_idx=False)

        return results

    def training_step(self, batch, batch_idx):
        lr = batch["lr"]
        hr = batch["hr"]

        sr = self.model(lr)

        losses = self.calculate_loss(sr, hr, prefix="train")
        metrics = self.calculate_metrics(sr, hr, prefix="train")

        return losses

    def validation_step(self, batch, batch_idx):
        lr = batch["lr"]
        hr = batch["hr"]

        sr = self.model(lr)

        sr = sr.clamp(0, 1)
        hr = hr.clamp(0, 1)

        losses = self.calculate_loss(sr, hr, prefix="val", prog_bar=False)
        metrics = self.calculate_metrics(sr, hr, prefix="val")

        return metrics

    def test_step(self, batch, batch_idx):
        name = os.path.dirname(batch["path"][0][0])

        sr_root = os.path.join(self.logger.save_dir, "results", "sr")
        os.makedirs(sr_root, exist_ok=True)

        hr_root = os.path.join(self.logger.save_dir, "results", "hr")
        os.makedirs(hr_root, exist_ok=True)

        lr = batch["lr"]
        hr = batch["hr"]

        y = lr[..., 0, :, :]
        u = lr[..., 1, :, :]
        v = lr[..., 2, :, :]

        indices = batch["indices"][0]

        # y = y.index_select(dim=-3, index=indices)
        u = u.index_select(dim=-3, index=indices)
        v = v.index_select(dim=-3, index=indices)

        if self.is_hidden:
            if not name in self.hiddens:
                self.hiddens[name] = torch.zeros(
                    1,
                    self.model.n_hiddens,
                    y.size(-2),
                    y.size(-1),
                    device=self.device,
                )

            y, self.hiddens[name] = self.model.inference(y, self.hiddens[name])
        else:
            y = self.model.inference(y)

        # y = F.interpolate(y, scale_factor=self.model.scale_factor, mode="bicubic")
        u = F.interpolate(u, scale_factor=self.model.scale_factor, mode="bilinear")
        v = F.interpolate(v, scale_factor=self.model.scale_factor, mode="bilinear")

        sr = torch.cat([y, u, v], dim=-3)
        sr = sr.mul(255).clamp(0, 255).byte().cpu().numpy()
        hr = hr.mul(255).clamp(0, 255).byte().cpu().numpy()

        with open(os.path.join(sr_root, name + ".yuv"), "a") as f:
            sr.flatten().tofile(f)

        with open(os.path.join(hr_root, name + ".yuv"), "a") as f:
            hr.flatten().tofile(f)

    def on_test_epoch_start(self):
        outputs_root = os.path.join(self.logger.save_dir, "results")
        shutil.rmtree(outputs_root, ignore_errors=True)

        self.is_hidden = hasattr(self.model, "n_hiddens") # THIS

        self.hiddens = {}
