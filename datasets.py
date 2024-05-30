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


class VideoDataset(D.Dataset):
    def __init__(
        self,
        root,
        split,
        *,
        lr_subdir="lr",
        hr_subdir="hr",
        sequence_lr_indices=[0, 1],
        sequence_hr_indices=[1],
        transform=None,
        return_path=False,
        return_indices=False,
        **kwargs,
    ):
        super().__init__()

        assert split in ["train", "val", "test"]
        assert all(i in sequence_lr_indices for i in sequence_hr_indices)

        self.root = root
        self.lr_root = os.path.join(root, split, lr_subdir)
        self.hr_root = os.path.join(root, split, hr_subdir)

        video_names = glob.glob("*", root_dir=self.lr_root)

        self.sequence_length = len(sequence_lr_indices)
        self.sequence_lr_indices = sequence_lr_indices
        self.sequence_hr_indices = sequence_hr_indices

        self.frames = []
        self.sequences = []

        for video_name in video_names:
            video_frames = sorted(glob.glob(f"{video_name}/*", root_dir=self.lr_root))

            sequence_start_first = len(self.frames)
            sequence_start_last = (
                sequence_start_first + len(video_frames) - self.sequence_length + 1
            )

            self.frames.extend(video_frames)
            self.sequences.extend(range(sequence_start_first, sequence_start_last))

        self.transform = transform

        self.return_path = return_path
        self.return_indices = return_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        lr_indices = [self.sequences[index] + i for i in self.sequence_lr_indices]
        hr_indices = [self.sequences[index] + i for i in self.sequence_hr_indices]

        lr_frames = [self.frames[i] for i in lr_indices]
        hr_frames = [self.frames[i] for i in hr_indices]

        lr_sequence = utils.load_sequence(
            lr_frames, root=self.lr_root, load=utils.load_png
        )
        hr_sequence = utils.load_sequence(
            hr_frames, root=self.hr_root, load=utils.load_png
        )

        if self.transform:
            lr_sequence, hr_sequence = self.transform(lr_sequence, hr_sequence)

        result = {
            "lr": lr_sequence,
            "hr": hr_sequence,
        }

        if self.return_path:
            result = result | {
                "path": hr_frames,
            }

        if self.return_indices:
            result = result | {
                "indices": torch.tensor(self.sequence_hr_indices),
            }

        return result


class VideoDataModule(L.LightningDataModule):
    def __init__(self, *, batch_size, num_workers, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.kwargs = kwargs

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = VideoDataset(
                split="train",
                transform=T.Compose(
                    [
                        VideoRandomCrop((128, 128)),
                        VideoRandomHorizontalFlip(),
                        VideoRandomVerticalFlip(),
                        VideoToDtype(),
                        VideoRgbToYuv400(),
                    ]
                ),
                **self.kwargs,
            )

            self.val_dataset = VideoDataset(
                split="val",
                transform=T.Compose(
                    [
                        VideoRandomCrop((128, 128)),
                        VideoRandomHorizontalFlip(),
                        VideoRandomVerticalFlip(),
                        VideoToDtype(),
                        VideoRgbToYuv400(),
                    ]
                ),
                **self.kwargs,
            )

        if stage == "test":
            self.test_dataset = VideoDataset(
                split="test",
                transform=T.Compose(
                    [
                        VideoToDtype(),
                        VideoRgbToYuv444(),
                    ]
                ),
                return_path=True,
                return_indices=True,
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
    root_reds = "/Volumes/SAM1000EX/Datasets/REDS"
    root_vid4 = "/Volumes/SAM1000EX/Datasets/Vid4"

    dm_fit = VideoDataModule(
        batch_size=4,
        num_workers=4,
        root=root_reds,
        lr_subdir="lr_x2",
        hr_subdir="hr",
        sequence_lr_indices=[0, 1],
        sequence_hr_indices=[1],
    )

    dm_fit.setup(stage="fit")
    train_loader = dm_fit.train_dataloader()
    val_loader = dm_fit.val_dataloader()

    dm_test = VideoDataModule(
        batch_size=4,
        num_workers=4,
        root=root_vid4,
        lr_subdir="lr_x2",
        hr_subdir="hr",
        sequence_lr_indices=[0, 1],
        sequence_hr_indices=[1],
    )

    dm_test.setup(stage="test")
    test_loader = dm_test.test_dataloader()

    batch = next(iter(test_loader))

    lr = batch["lr"]
    hr = batch["hr"]
    path = batch["path"]

    print("lr_y", lr.shape)
    print("hr_y", hr.shape)
    print("path", path)

    # print(lr[..., :3, :3])
    # print(hr[..., :6, :6])

    lr = lr.mul(255).clamp(0, 255).byte()
    hr = hr.mul(255).clamp(0, 255).byte()

    torchvision.io.write_file("lr.yuv", lr.flatten())
    torchvision.io.write_file("hr.yuv", hr.flatten())
