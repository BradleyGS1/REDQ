import numpy as np
import tensorflow as tf

# Create running average metric class
# Used for keeping track of any mid-training metric via a running average
class RunningAverageMetric:
    def __init__(self, sample_size: int, non_zero: bool):
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
        
        return np.array(0, dtype=np.int8)

    def _tf_update(self, new_value: tf.Tensor):
        tf.numpy_function(self._update, [new_value], tf.int8)

    def _get_running_average(self):
        if self.size != self.sample_size:
            return tf.constant(self.non_zero, dtype=tf.float32)
        else:
            return tf.constant(self.avg + self.non_zero * 1e-6, dtype=tf.float32)
        
# Create running average return class
# Used for keeping track of the mean return evaluations via a running average
class RunningAverageReturn:
    def __init__(self, sample_size: int):
        self.sample = np.zeros(shape=sample_size, dtype=np.float32)
        self.sample_size = sample_size
        self.index = 0
        self.size = 0

    def _update(self, new_value: np.float32):
        self.sample[self.index] = new_value

        self.size = min(self.size + 1, self.sample_size)
        self.index = (self.index + 1) % self.sample_size

        self.avg = np.sum(self.sample) / self.size        

    def _get_running_average(self):
        return self.avg