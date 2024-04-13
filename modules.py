import os

from functools import partial

import logging
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import utils
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

        optimizer = torch.optim.Adam(parameters, **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 75, 90], 0.5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        y = self.model(x)

        return y

    def calculate_loss(self, sr, hr):
        results = {}

        sr = sr.flatten(1, 2)
        hr = hr.flatten(1, 2)

        total = 0

        for loss in self.losses:
            score = loss.func(sr, hr)
            total = total + score * loss.weight

            results[f"loss/{loss.name}"] = score.item()

        results["loss/loss"] = total

        self.log_dict(results, prog_bar=True, logger=True, add_dataloader_idx=False)

        results["loss"] = total

        return results

    def calculate_metrics(self, sr, hr, prefix="metric", suffix=""):
        results = {}

        sr = sr.flatten(1, 2)
        hr = hr.flatten(1, 2)

        for metric in self.metrics:
            score = metric.func(sr, hr)

            results[f"{prefix}/{metric.name}/{suffix}"] = score.item()

        return results

    def training_step(self, batch, batch_idx):
        lr_y = batch["lr_y"]
        hr_y = batch["hr_y"]

        sr_y = self.model(lr_y)

        losses = self.calculate_loss(sr_y, hr_y)

        return losses

    def validation_step(self, batch, batch_idx):
        lr_y = batch["lr_y"]
        hr_y = batch["hr_y"]

        sr_y = self.model(lr_y)

        sr_y = sr_y.clamp(0, 1)
        hr_y = hr_y.clamp(0, 1)

        metrics = {}

        metrics = metrics | self.calculate_metrics(sr_y, hr_y, suffix="y")

        sr_uv = batch["sr_uv"]
        hr_uv = batch["hr_uv"]

        sr_yuv = torch.cat([sr_y, sr_uv], dim=-3)
        hr_yuv = torch.cat([hr_y, hr_uv], dim=-3)

        sr_rgb = utils.yuv_to_rgb(sr_yuv)
        hr_rgb = utils.yuv_to_rgb(hr_yuv)

        metrics = metrics | self.calculate_metrics(sr_rgb, hr_rgb, suffix="rgb")

        self.log_dict(metrics, prog_bar=False, logger=True, add_dataloader_idx=False)

        return metrics

    def test_step(self, batch, batch_idx):
        lr_y = batch["lr_y"]
        hr_y = batch["hr_y"]
        br_y = batch["br_y"]

        sr_y = self.model(lr_y)

        sr_y = sr_y.clamp(0, 1)
        hr_y = hr_y.clamp(0, 1)
        br_y = br_y.clamp(0, 1)

        metrics = {}

        metrics = metrics | self.calculate_metrics(
            sr_y, hr_y, prefix="test", suffix="y"
        )
        metrics = metrics | self.calculate_metrics(
            br_y, hr_y, prefix="test", suffix="y_br"
        )

        if sr_y.size(1) > 1:
            metrics = metrics | self.calculate_metrics(
                sr_y[:, -1:], hr_y[:, -1:], prefix="test", suffix="y_last"
            )
            metrics = metrics | self.calculate_metrics(
                br_y[:, -1:], hr_y[:, -1:], prefix="test", suffix="y_br_last"
            )

        sr_uv = batch["sr_uv"]
        hr_uv = batch["hr_uv"]
        br_uv = batch["br_uv"]

        sr_yuv = torch.cat([sr_y, sr_uv], dim=-3)
        hr_yuv = torch.cat([hr_y, hr_uv], dim=-3)
        br_yuv = torch.cat([br_y, br_uv], dim=-3)

        sr_rgb = utils.yuv_to_rgb(sr_yuv)
        hr_rgb = utils.yuv_to_rgb(hr_yuv)
        br_rgb = utils.yuv_to_rgb(br_yuv)

        metrics = metrics | self.calculate_metrics(
            sr_rgb, hr_rgb, prefix="test", suffix="rgb"
        )
        metrics = metrics | self.calculate_metrics(
            br_rgb, hr_rgb, prefix="test", suffix="rgb_br"
        )

        if sr_rgb.size(1) > 1:
            metrics = metrics | self.calculate_metrics(
                sr_rgb[:, -1:], hr_rgb[:, -1:], prefix="test", suffix="rgb_last"
            )
            metrics = metrics | self.calculate_metrics(
                br_rgb[:, -1:], hr_rgb[:, -1:], prefix="test", suffix="rgb_br_last"
            )

        self.log_dict(
            metrics,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

        # TODO save samples
        sr_yuv = sr_yuv.mul(255).clamp(0, 255).byte().cpu()
        br_yuv = br_yuv.mul(255).clamp(0, 255).byte().cpu()

        for i, sr in enumerate(sr_yuv[0]):
            sr_path = os.path.join(
                self.logger.save_dir, "results", "sr", batch["path"][i][0]
            )
            os.makedirs(os.path.dirname(sr_path), exist_ok=True)
            torchvision.io.write_file(sr_path, sr.flatten())

        # for i, br in enumerate(br_yuv[0]):
        #     br_path = os.path.join(
        #         self.logger.save_dir, "results", "br", batch["path"][i][0]
        #     )
        #     os.makedirs(os.path.dirname(br_path), exist_ok=True)
        #     torchvision.io.write_file(br_path, br.flatten())
