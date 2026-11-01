"""Epoch sampling policies, with independent seeded RNGs."""

import torch
from torch.utils.data import Sampler


class RandomReshufflingSampler(Sampler[int]):
    """Visit every item once per epoch, with a new permutation each epoch."""

    def __init__(self, data_source, seed: int = 42):
        self.length = len(data_source)
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        return iter(torch.randperm(self.length, generator=self.generator).tolist())

    def __len__(self):
        return self.length


class ShuffleOnceSampler(Sampler[int]):
    """Visit the same seeded random permutation on every epoch."""

    def __init__(self, data_source, seed: int = 42):
        self.indices = torch.randperm(
            len(data_source), generator=torch.Generator().manual_seed(seed)
        ).tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SequentialSampler(Sampler[int]):
    """Fixed source order after splitting, included as a no-shuffling control."""

    def __init__(self, data_source):
        self.length = len(data_source)

    def __iter__(self):
        return iter(range(self.length))

    def __len__(self):
        return self.length


SAMPLERS = {
    "reshuffle": RandomReshufflingSampler,
    "once": ShuffleOnceSampler,
    "sequential": SequentialSampler,
}
