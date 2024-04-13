import numpy as np
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
import torchvision.transforms.v2 as T
import lightning as L

import utils

from transforms import *


class VideoYuvDataset(D.Dataset):
    def __init__(
        self,
        root,
        split,
        *,
        lr_subdir="frames/lr",
        hr_subdir="frames/hr",
        sequence_lr_indices=[0, 1],
        sequence_hr_indices=[1],
        transform=None,
        return_uv=False,
        return_path=False,
        **kwargs,
    ):
        super().__init__()

        assert split in ["train", "val", "test"]  # TODO add "all" option
        assert all(i in sequence_lr_indices for i in sequence_hr_indices)

        with open(os.path.join(root, split) + ".txt", "r") as f:
            self.video_names = f.read().splitlines()

        self.root = root
        self.lr_root = os.path.join(root, lr_subdir)
        self.hr_root = os.path.join(root, hr_subdir)

        self.sequence_length = len(sequence_lr_indices)
        self.sequence_lr_indices = sequence_lr_indices
        self.sequence_hr_indices = sequence_hr_indices

        self.frames = []
        self.sequences = []

        for video_name in self.video_names:
            video_frames = sorted(
                glob.glob(f"{video_name}/*.yuv", root_dir=self.lr_root)
            )

            sequence_start_first = len(self.frames)
            sequence_start_last = (
                sequence_start_first + len(video_frames) - self.sequence_length + 1
            )

            self.frames.extend(video_frames)
            self.sequences.extend(range(sequence_start_first, sequence_start_last))

        self.transform = transform

        self.return_uv = return_uv
        self.return_path = return_path

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        lr_indices = [self.sequences[index] + i for i in self.sequence_lr_indices]
        hr_indices = [self.sequences[index] + i for i in self.sequence_hr_indices]

        lr_frames = [self.frames[i] for i in lr_indices]
        hr_frames = [self.frames[i] for i in hr_indices]

        lr_sequence = utils.load_sequence(lr_frames, root_dir=self.lr_root)
        hr_sequence = utils.load_sequence(hr_frames, root_dir=self.hr_root)

        if self.transform:
            lr_sequence, hr_sequence = self.transform(lr_sequence, hr_sequence)

        lr_y, lr_uv = lr_sequence.split([1, 2], dim=-3)
        hr_y, hr_uv = hr_sequence.split([1, 2], dim=-3)

        br_y = torch.nn.functional.interpolate(
            lr_y.index_select(dim=-4, index=torch.tensor(self.sequence_hr_indices)),
            size=hr_y.shape[-2:],
            mode="bicubic",
        )

        result = {
            "lr_y": lr_y,
            "hr_y": hr_y,
            "br_y": br_y,
        }

        if self.return_uv:
            br_uv = torch.nn.functional.interpolate(
                lr_uv.index_select(
                    dim=-4, index=torch.tensor(self.sequence_hr_indices)
                ),
                size=hr_uv.shape[-2:],
                mode="bilinear",
            )

            result = result | {
                "sr_uv": br_uv,  # sr_uv acquired the same way as br_uv
                "hr_uv": hr_uv,
                "br_uv": br_uv,
            }

        if self.return_path:
            result = result | {
                "path": hr_frames,
            }

        return result


class VideoYuvDataModule(L.LightningDataModule):
    def __init__(self, *, batch_size, num_workers, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.kwargs = kwargs

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = VideoYuvDataset(
                split="train",
                transform=T.Compose(
                    [
                        VideoRandomCrop((120, 120)),
                        VideoRandomHorizontalFlip(),
                        VideoRandomVerticalFlip(),
                    ]
                ),
                **self.kwargs,
            )

            self.val_dataset = VideoYuvDataset(
                split="val",
                transform=T.Compose(
                    [
                        VideoRandomCrop((120, 120)),
                        VideoRandomHorizontalFlip(),
                        VideoRandomVerticalFlip(),
                    ]
                ),
                return_uv=True,
                **self.kwargs,
            )

        if stage == "test":
            self.test_dataset = VideoYuvDataset(
                split="test",
                return_uv=True,
                return_path=True,
                **self.kwargs,
            )

    def train_dataloader(self):
        return D.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return D.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return D.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


if __name__ == "__main__":
    root = "/Volumes/SAM1000EX/Datasets/Inter4K"

    vfdm = VideoYuvDataModule(
        batch_size=2,
        num_workers=2,
        root=root,
        lr_subdir="frames/ld",
        hr_subdir="frames/hd",
        sequence_lr_indices=[0, 1],
        sequence_hr_indices=[1],
    )

    vfdm.setup()
    train_loader = vfdm.train_dataloader()
    val_loader = vfdm.val_dataloader()
    test_loader = vfdm.test_dataloader()

    batch = next(iter(test_loader))

    lr_y = batch["lr_y"]
    hr_y = batch["hr_y"]
    br_y = batch["br_y"]

    sr_uv = batch["sr_uv"]
    hr_uv = batch["hr_uv"]
    br_uv = batch["br_uv"]

    path = batch["path"]

    print("lr_y", lr_y.shape)
    print("hr_y", hr_y.shape)
    print("br_y", br_y.shape)

    print("sr_uv", sr_uv.shape)
    print("hr_uv", hr_uv.shape)
    print("br_uv", br_uv.shape)

    print("path", path)

    br_yuv = torch.cat([br_y[0, 0], br_uv[0, 0]], dim=-3)
    hr_yuv = torch.cat([hr_y[0, 0], hr_uv[0, 0]], dim=-3)

    br_rgb = utils.yuv_to_rgb(br_yuv).mul(255).clamp(0, 255).byte()
    hr_rgb = utils.yuv_to_rgb(hr_yuv).mul(255).clamp(0, 255).byte()

    br_yuv = br_yuv.mul(255).clamp(0, 255).byte()
    hr_yuv = hr_yuv.mul(255).clamp(0, 255).byte()

    torchvision.io.write_file("br.yuv", br_yuv.flatten())
    torchvision.io.write_file("hr.yuv", hr_yuv.flatten())

    torchvision.io.write_png(br_rgb, "br_rgb.png")
    torchvision.io.write_png(hr_rgb, "hr_rgb.png")
