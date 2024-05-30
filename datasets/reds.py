import os
import pathlib

import tqdm


def downscale(root, scale):
    hr_root = root / f"hr"
    lr_root = root / f"lr_x{scale}"

    sequences_list = list(
        filter(lambda p: not p.stem.startswith("._"), hr_root.glob("*"))
    )

    for frames_root in tqdm.tqdm(sequences_list):
        input_pattern = hr_root / frames_root.stem / "%08d.png"
        output_pattern = lr_root / frames_root.stem / "%08d.png"

        output_pattern.parent.mkdir(parents=True, exist_ok=True)

        cmd = f"ffmpeg -y -i {input_pattern} -vf 'scale=iw/{scale}:ih/{scale}' -start_number 0 {output_pattern} 2> /dev/null"

        os.system(cmd)


def downscale_compress(root, scale, crf=25):
    hr_root = root / f"hr"
    lr_root = root / f"lr_x{scale}_crf{crf}"

    sequences_list = list(
        filter(lambda p: not p.stem.startswith("._"), hr_root.glob("*"))
    )

    for frames_root in tqdm.tqdm(sequences_list):
        input_pattern = hr_root / frames_root.stem / "%08d.png"
        output_pattern = lr_root / frames_root.stem / "%08d.png"

        output_pattern.parent.mkdir(parents=True, exist_ok=True)

        cmd = f"ffmpeg -y -i {input_pattern} -vf 'scale=iw/{scale}:ih/{scale}' -c:v libx264rgb -crf {crf} -f h264 pipe:1 2> /dev/null | ffmpeg -i pipe:0 -start_number 0 {output_pattern} 2> /dev/null"

        os.system(cmd)


if __name__ == "__main__":
    train_root = pathlib.Path("/Volumes/SAM1000EX/Datasets/REDS/train")

    # downscale(train_root, scale=2)
    # downscale(train_root, scale=3)
    downscale_compress(train_root, scale=2, crf=25)
    downscale_compress(train_root, scale=3, crf=25)

    val_root = pathlib.Path("/Volumes/SAM1000EX/Datasets/REDS/val")

    # downscale(val_root, scale=2)
    # downscale(val_root, scale=3)
    downscale_compress(val_root, scale=2, crf=25)
    downscale_compress(val_root, scale=3, crf=25)
