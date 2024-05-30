import os
import pathlib

import torch
import tqdm

VERBOSE = False


def resample(video_path, scale, width, height):
    resampled_dir = video_path.parents[1] / scale
    resampled_path = resampled_dir / video_path.name

    resampled_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"ffmpeg -i {video_path} -vf 'scale={width}:{height}' {resampled_path} -y"

    if not VERBOSE:
        cmd += " > /dev/null 2> /dev/null"

    os.system(cmd)


def resample_extract(video_path, scale, width, height, suffix=".png", n_frames=-1):
    frames_dir = video_path.parents[2] / "frames" / scale / video_path.stem
    frames_pattern = frames_dir / "%05d"

    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [f"ffmpeg -i {video_path}"]

    if n_frames > 0:
        cmd.append(
            f"-vf 'fps=30000/1001,scale={width}:{height},select=between(n\\, 0\\, {n_frames-1})'"
        )
    else:
        cmd.append(f"-vf 'fps=30000/1001,scale={width}:{height}'")

    if suffix == ".yuv":
        cmd.append("-f segment -segment_time 0.01")

    cmd.append(f"{frames_pattern.with_suffix(suffix)} -y")

    if not VERBOSE:
        cmd.append("> /dev/null 2> /dev/null")

    cmd = " ".join(cmd)

    os.system(cmd)


def prepare_scales(videos_root, scales_dict, suffix, n_frames):
    videos_list = list(videos_root.glob("*.mp4"))[:-1]

    for video_path in tqdm.tqdm(videos_list):
        for scale in scales_dict:
            resample_extract(video_path, scale, *scales_dict[scale], suffix, n_frames)


def prepare_splits(videos_root, splits_dict):
    videos_list = list(filter(lambda v: v.name.isdigit(), videos_root.glob("*")))

    assert sum(splits_dict.values()) <= 1.0

    splits = torch.utils.data.random_split(videos_list, splits_dict.values())

    for split_t, split_list in zip(splits_dict.keys(), splits):
        with open(videos_root.parent.parent / f"{split_t}.txt", "w") as f:
            for video_path in split_list:
                f.write(f"{video_path.stem}\n")


if __name__ == "__main__":
    videos_root = pathlib.Path("/Users/notslacker/Datasets/Inter4K/videos/uhd")
    scales_dict = {"fhd": [1920, 1080], "hd": [1280, 720], "ld": [640, 360]}

    prepare_scales(videos_root, scales_dict, suffix=".yuv", n_frames=5)

    frames_root = pathlib.Path("/Users/notslacker/Datasets/Inter4K/frames/ld")
    splits_dict = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
    }

    prepare_splits(frames_root, splits_dict)
