"""Minimal accelerate stub — Accelerator class for lerobot utils."""

__version__ = "0.0.0+stub"


class Accelerator:
    """Stub Accelerator (never used in dataset_manager's code paths)."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def device(self):
        import torch
        return torch.device("cpu")

    @property
    def is_main_process(self):
        return True

    @property
    def num_processes(self):
        return 1

    def prepare(self, *args):
        if len(args) == 1:
            return args[0]
        return args

    def backward(self, loss):
        pass

    def wait_for_everyone(self):
        pass

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def log(self, values, step=None):
        pass

    def save_state(self, output_dir=None):
        pass

    def unwrap_model(self, model):
        return model
