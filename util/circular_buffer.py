"""This is my implementation of a very fast Circular Buffer in Python using Numpy.
   I used the Sequence abstract class which just needs the length and get."""
from collections.abc import Sequence

import numpy as np


class CircularBuffer(Sequence):
    """This is the circular buffer, its backed by a fixed size Numpy array."""
    def __init__(self, max_len, shape, **kwargs):
        self.buffer = np.empty((max_len, *shape), **kwargs)
        self.length = 0
        self.full = False
        self.offset = 0

    def __getitem__(self, indices):
        """Use Numpy slicing to get the sample concisely"""
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

    def mean(self):
        return np.mean(self.buffer[:self.length], axis=0)

    def __len__(self):
        return self.length
