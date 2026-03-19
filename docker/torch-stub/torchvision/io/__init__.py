"""torchvision.io stub — VideoReader for video_utils.py."""

import torch


class VideoReader:
    """Stub VideoReader that raises if actually used for decoding."""

    def __init__(self, path, stream="video", **kwargs):
        self._path = path
        self._timestamps = []
        self._frames = []

    def seek(self, timestamp):
        pass

    def __next__(self):
        raise StopIteration

    def __iter__(self):
        return self

    def get_metadata(self):
        return {"video": {"fps": [30.0], "duration": [0.0]}}


def read_video(filename, start_pts=0, end_pts=None, pts_unit="pts", **kwargs):
    return torch.Tensor(), None, None


def write_video(filename, video_array, fps, **kwargs):
    pass
