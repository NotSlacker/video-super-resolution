import os

import torch
import torchvision

YUV_SHAPE = {
    3840*2160 * 3//2: (2160, 3840),
    1920*1080 * 3//2: (1080, 1920),
    1280* 720 * 3//2: (720, 1280),
     640* 360 * 3//2: (360, 640),
}

def yuv_to_rgb(yuv, color_matrix="709", color_range="limited"):
    y = yuv[..., 0, :, :]
    u = yuv[..., 1, :, :]
    v = yuv[..., 2, :, :]

    if color_range == "limited":
        y = y - 16 / 255
        u = u - 128 / 255
        v = v - 128 / 255

        if color_matrix == "601":
            r = 1.164 * y             + 1.596 * v
            g = 1.164 * y - 0.392 * u - 0.813 * v
            b = 1.164 * y + 2.017 * u
        if color_matrix == "709":
            r = 1.164 * y             + 1.793 * v
            g = 1.164 * y - 0.213 * u - 0.533 * v
            b = 1.164 * y + 2.112 * u

    if color_range == "full":
        y = y
        u = u - 128 / 255
        v = v - 128 / 255

        if color_matrix == "601":
            r = y             + 1.402 * v
            g = y - 0.344 * u - 0.714 * v
            b = y + 1.772 * u
        if color_matrix == "709":
            r = y              + 1.5748 * v
            g = y - 0.1873 * u - 0.4681 * v
            b = y + 1.8556 * u

    return torch.stack([r, g, b], dim=-3).clamp(0, 1)

def load_yuv(path, format="420"):
    yuv = torchvision.io.read_file(path).to(torch.float32)

    h, w = YUV_SHAPE[yuv.numel()]

    if format == "420":
        y, uv = yuv.split([h*w, h*w // 2])

        y = y.reshape(1, h, w)
        uv = uv.reshape(2, h // 2, w // 2)

    if format == "444":
        y, uv = yuv.split([h*w, h*w * 2])

        y = y.reshape(1, h, w)
        uv = uv.reshape(2, h, w)

    y = y / 255
    uv = uv / 255

    uv = torch.nn.functional.interpolate(uv.unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0)

    yuv = torch.cat([y, uv], dim=-3)

    return yuv

def load_sequence(frames, root_dir):
    sequence = []
    for frame_path in frames:
        frame = load_yuv(os.path.join(root_dir, frame_path))
        sequence.append(frame)
    sequence = torch.stack(sequence)

    return sequence
