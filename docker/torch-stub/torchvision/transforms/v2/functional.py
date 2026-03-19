"""torchvision.transforms.v2.functional stub."""

import torch


def to_tensor(pic):
    return torch.Tensor()


def normalize(tensor, mean, std, inplace=False):
    return tensor


def resize(img, size, **kwargs):
    return img


def center_crop(img, output_size):
    return img


def crop(img, top, left, height, width):
    return img


def hflip(img):
    return img


def vflip(img):
    return img


def rotate(img, angle, **kwargs):
    return img


def adjust_brightness(img, brightness_factor):
    return img


def adjust_contrast(img, contrast_factor):
    return img


def adjust_saturation(img, saturation_factor):
    return img


def adjust_hue(img, hue_factor):
    return img


def to_pil_image(pic, mode=None):
    from PIL import Image
    return Image.new("RGB", (1, 1))


def clamp(input, min=None, max=None):
    return input
