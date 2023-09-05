from typing import Sized

import numpy as np


class Dataset:
    def __getitem__(self, idx: int):
        raise NotImplemented("Subclasses of Dataset should implement __getitem__")

    def __len__(self):
        raise NotImplemented("Subclasses of Dataset should implement __len__")


class Sampler:
    def __iter__(self):
        NotImplemented("Subclasses of Sampler should implement __iter__")


class RandomSampler(Sampler):
    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self):
        data_len = len(self.data_source)

        yield from np.random.permutation(data_len)

    def __len__(self):
        return len(self.data_source)


class SequentialSampler(Sampler):
    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, sampler: Sampler = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler if sampler is not None else SequentialSampler(list(range(len(dataset))))

        self._curr_index = 0

    def __iter__(self) -> "DataLoader":
        return self

    def __next__(self):
        if self._curr_index >= len(self.dataset):
            raise StopIteration

        batch = self.dataset[self._curr_index:self._curr_index + self.batch_size]
        self._curr_index += self.batch_size

        return batch
