"""torchvision.transforms stub — ToTensor, Compose for lerobot utils.py."""

import torch
from torchvision.transforms import v2


class ToTensor:
    """Stub: converts PIL/numpy to torch.Tensor."""

    def __call__(self, pic):
        return torch.Tensor()

    def __repr__(self):
        return "ToTensor()"


class ToPILImage:
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        from PIL import Image
        return Image.new("RGB", (1, 1))


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor


class Resize:
    def __init__(self, size, **kwargs):
        self.size = size

    def __call__(self, img):
        return img


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img


class RandomCrop:
    def __init__(self, size, **kwargs):
        self.size = size

    def __call__(self, img):
        return img


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        pass

    def __call__(self, img):
        return img


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        pass

    def __call__(self, img):
        return img


class Lambda:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)
