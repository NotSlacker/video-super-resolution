import argparse
import json
import os

import tqdm

from ffmpeg_quality_metrics import FfmpegQualityMetrics

VERBOSE = True

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
    parser.add_argument("output", type=str, help="path to output file")
    args = parser.parse_args()

    for name, meta in tqdm.tqdm(video_meta.items()):
        path_to_reference = os.path.join(args.reference, name + ".yuv")
        path_to_reference_tmp = os.path.join(args.reference, name + ".y4m")

        cmd = f"ffmpeg -y -r 30 -video_size {meta['width']}x{meta['height']} -pix_fmt {meta['format']} -i {path_to_reference} {path_to_reference_tmp}"
        if not VERBOSE:
            cmd += "> /dev/null 2> /dev/null"

        os.system(cmd)

        path_to_distorted = os.path.join(args.distorted, name + ".yuv")
        path_to_distorted_tmp = os.path.join(args.distorted, name + ".y4m")

        cmd = f"ffmpeg -y -r 30 -video_size {meta['width']}x{meta['height']} -pix_fmt {meta['format']} -i {path_to_distorted} {path_to_distorted_tmp}"
        if not VERBOSE:
            cmd += "> /dev/null 2> /dev/null"

        os.system(cmd)

        ffqm = FfmpegQualityMetrics(ref=path_to_reference_tmp, dist=path_to_distorted_tmp, verbose=True)

        ffqm.calculate(
            ["ssim", "psnr", "vmaf"]
            # ["ssim", "psnr", "vmaf"], {"model_params": ["enable_transform=true"]}
        )

        path_to_output = os.path.join(args.output, name + ".json")
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

        with open(path_to_output, "w") as f:
            f.writelines(ffqm.get_results_json())

        # os.remove(path_to_reference_tmp)
        # os.remove(path_to_distorted_tmp)
