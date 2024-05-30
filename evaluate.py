import argparse
import os
import re
import subprocess

import tqdm

video_meta = {
    "calendar": {"height": 576, "width": 720, "format": "yuv444p"},
    "city": {"height": 576, "width": 702, "format": "yuv444p"},
    "foliage": {"height": 480, "width": 720, "format": "yuv444p"},
    "walk": {"height": 480, "width": 720, "format": "yuv444p"},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("distorted", type=str, help="path to distorted yuv video")
    parser.add_argument("reference", type=str, help="path to reference yuv video")
    parser.add_argument("output", type=str, help="path to output log")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for name, meta in tqdm.tqdm(video_meta.items()):
        path_to_distorted = os.path.join(args.distorted, name + ".yuv")
        path_to_reference = os.path.join(args.reference, name + ".yuv")
        path_to_log = os.path.join(args.output, name)

        cmd = [
            "ffmpeg",
            "-video_size",
            f"{meta['width']}x{meta['height']}",
            "-pix_fmt",
            f"{meta['format']}",
            "-i",
            f"{path_to_distorted}",
            "-video_size",
            f"{meta['width']}x{meta['height']}",
            "-pix_fmt",
            f"{meta['format']}",
            "-i",
            f"{path_to_reference}",
            "-lavfi",
            f"[0:v][1:v]libvmaf=log_fmt=csv:log_path={path_to_log}.vmaf;[0:v][1:v]psnr=stats_file={path_to_log}.psnr;[0:v][1:v]ssim=stats_file={path_to_log}.ssim",
            "-f",
            "null",
            "-",
        ]

        res = subprocess.run(cmd, capture_output=True, text=True).stderr

        vmaf = re.findall("VMAF score: (\d+\.\d+)", res)[0]
        psnr = re.findall("PSNR y:(\d+\.\d+)", res)[0]
        ssim = re.findall("SSIM Y:(\d+\.\d+)", res)[0]

        with open(path_to_log + ".vmaf", "a") as f:
            f.write("vmaf (global): " + vmaf + "\n")

        with open(path_to_log + ".psnr", "a") as f:
            f.write("psnr (global): " + psnr + "\n")

        with open(path_to_log + ".ssim", "a") as f:
            f.write("ssim (global): " + ssim + "\n")
