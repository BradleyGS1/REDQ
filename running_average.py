import numpy as np
import tensorflow as tf

# Create running average metric class
# Used for keeping track of any mid-training metric via a running average
class RunningAverage:
    def __init__(self, sample_size: int, non_zero: bool = False):
        self.sample = np.zeros(shape=sample_size, dtype=np.float32)
        self.sample_size = sample_size
        self.index = 0
        self.size = 0
        self.non_zero = non_zero

    def _update(self, new_value: np.float32):
        self.sample[self.index] = new_value

        self.size = min(self.size + 1, self.sample_size)
        self.index = (self.index + 1) % self.sample_size

        self.avg = np.sum(self.sample) / self.size

    def _get_running_average(self):
        if self.size != self.sample_size:
            return np.median(self.sample[np.abs(self.sample) > 1e-6]) + self.non_zero * 1e-6
        else:
            return self.avg + self.non_zero * 1e-6