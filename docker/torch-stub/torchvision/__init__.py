"""Minimal torchvision stub."""

__version__ = "0.0.0+stub"

from torchvision import io, transforms


def set_video_backend(backend):
    """No-op stub for torchvision.set_video_backend()."""
    pass


def get_video_backend():
    return "pyav"
