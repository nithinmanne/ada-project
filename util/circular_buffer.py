from collections.abc import Sequence

import numpy as np


class CircularBuffer(Sequence):
    def __init__(self, max_len, shape, **kwargs):
        self.buffer = np.empty((max_len, *shape), **kwargs)
        self.length = 0
        self.full = False
        self.offset = 0

    def __getitem__(self, indices):
        assert np.max(indices) < self.length
        return self.buffer[(indices + self.offset) % len(self.buffer)]

    def append(self, value):
        if not self.full:
            self.buffer[self.length] = value
            self.length += 1
            self.full = self.length == len(self.buffer)
        else:
            self.buffer[self.offset] = value
            self.offset = (self.offset + 1) % len(self.buffer)

    def __len__(self):
        return self.length
