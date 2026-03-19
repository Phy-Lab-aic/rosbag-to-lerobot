"""torch.utils.data stub — Dataset and DataLoader for LeRobotDataset inheritance."""


class Dataset:
    """Base class for datasets (stub for LeRobotDataset to inherit from)."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError


class IterableDataset:
    def __iter__(self):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


class Sampler:
    def __init__(self, data_source=None):
        pass

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


class RandomSampler(Sampler):
    pass


class SequentialSampler(Sampler):
    pass


class BatchSampler(Sampler):
    pass


class ConcatDataset(Dataset):
    pass


class Subset(Dataset):
    pass
