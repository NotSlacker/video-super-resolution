import os

import torch
import torchvision


def yuv_to_rgb(yuv, color_matrix="709", color_range="full"):
    y = yuv[..., 0, :, :]
    u = yuv[..., 1, :, :]
    v = yuv[..., 2, :, :]

    if color_range == "limited":
        y = y - 16 / 255
        u = u - 128 / 255
        v = v - 128 / 255

        if color_matrix == "601":
            r = 1.164 * y + 1.596 * v
            g = 1.164 * y - 0.392 * u - 0.813 * v
            b = 1.164 * y + 2.017 * u

        if color_matrix == "709":
            r = 1.164 * y + 1.793 * v
            g = 1.164 * y - 0.213 * u - 0.533 * v
            b = 1.164 * y + 2.112 * u

    if color_range == "full":
        y = y
        u = u - 128 / 255
        v = v - 128 / 255

        if color_matrix == "601":
            r = y + 1.402 * v
            g = y - 0.344 * u - 0.714 * v
            b = y + 1.772 * u

        if color_matrix == "709":
            r = y + 1.5748 * v
            g = y - 0.1873 * u - 0.4681 * v
            b = y + 1.8556 * u

    return torch.stack([r, g, b], dim=-3).clamp(0, 1)


def rgb_to_yuv(rgb, color_matrix="709", color_range="full"):
    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]

    if color_range == "limited":
        if color_matrix == "601":
            y = 0.257 * r + 0.504 * g + 0.098 * b
            u = -0.148 * r - 0.291 * g + 0.439 * b
            v = 0.439 * r - 0.368 * g - 0.071 * b

        if color_matrix == "709":
            y = 0.183 * r + 0.614 * g + 0.062 * b
            u = -0.1 * r - 0.338 * g + 0.439 * b
            v = 0.439 * r - 0.399 * g - 0.04 * b

        y = y + 16 / 255
        u = u + 128 / 255
        v = v + 128 / 255

    if color_range == "full":
        if color_matrix == "601":
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.169 * r - 0.331 * g + 0.5 * b
            v = 0.5 * r - 0.419 * g - 0.081 * b

        if color_matrix == "709":
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
            u = -0.1146 * r - 0.3854 * g + 0.5 * b
            v = 0.5 * r - 0.4542 * g - 0.0458 * b

        y = y
        u = u + 128 / 255
        v = v + 128 / 255

    return torch.stack([y, u, v], dim=-3).clamp(0, 1)


def rgb_to_y(rgb, color_matrix="709", color_range="full"):
    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]

    if color_range == "limited":
        if color_matrix == "601":
            y = 0.257 * r + 0.504 * g + 0.098 * b

        if color_matrix == "709":
            y = 0.183 * r + 0.614 * g + 0.062 * b

        y = y + 16 / 255

    if color_range == "full":
        if color_matrix == "601":
            y = 0.299 * r + 0.587 * g + 0.114 * b

        if color_matrix == "709":
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b

        y = y

    return y.unsqueeze(-3).clamp(0, 1)


def load_raw(path, height, width, format="yuv420"):
    data = torchvision.io.read_file(path)

    if format == "yuv400":
        return data.reshape(1, height, width)

    if format == "yuv420":
        return data.reshape(1, height * 3 // 2, width)

    if format == "yuv444":
        return data.reshape(3, height, width)


def load_png(path):
    data = torchvision.io.read_image(path)

    return data


def load_sequence(frames, root, load, **kwargs):
    sequence = []

    for frame_path in frames:
        frame = load(os.path.join(root, frame_path), **kwargs)
        sequence.append(frame)

    return torch.stack(sequence)
