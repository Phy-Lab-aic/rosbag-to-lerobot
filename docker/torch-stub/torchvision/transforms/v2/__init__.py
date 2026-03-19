"""torchvision.transforms.v2 stub — Transform base class for lerobot transforms.py."""

from torchvision.transforms.v2 import functional


class Transform:
    """Base transform class (stub for lerobot custom transforms)."""

    def __call__(self, *args, **kwargs):
        return args[0] if args else None

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        result = args[0] if args else None
        for t in self.transforms:
            result = t(result)
        return result


class ToTensor(Transform):
    pass


class Normalize(Transform):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std


class Resize(Transform):
    def __init__(self, size, **kwargs):
        self.size = size


class CenterCrop(Transform):
    def __init__(self, size):
        self.size = size


class RandomCrop(Transform):
    def __init__(self, size, **kwargs):
        self.size = size


class ColorJitter(Transform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        pass


class RandomHorizontalFlip(Transform):
    def __init__(self, p=0.5):
        pass


class ToDtype(Transform):
    def __init__(self, dtype, scale=False):
        pass


class RandomChoice(Transform):
    def __init__(self, transforms, p=None):
        self.transforms = transforms


class Identity(Transform):
    pass
