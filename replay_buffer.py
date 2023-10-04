import numpy as np
import tensorflow as tf

# Create replay buffer class

class ReplayBuffer:
    def __init__(self, replay_size: int):
        self.states_buffer = tf.TensorArray(dtype=tf.float32, size=replay_size, clear_after_read=False)
        self.actions_buffer = tf.TensorArray(dtype=tf.float32, size=replay_size, clear_after_read=False)
        self.rewards_buffer = tf.TensorArray(dtype=tf.float32, size=replay_size, clear_after_read=False)
        self.new_states_buffer = tf.TensorArray(dtype=tf.float32, size=replay_size, clear_after_read=False)
        self.done_flags_buffer = tf.TensorArray(dtype=tf.float32, size=replay_size, clear_after_read=False)

        self.buffers = [
            self.states_buffer,
            self.actions_buffer,
            self.rewards_buffer,
            self.new_states_buffer,
            self.done_flags_buffer
        ]

        #self.buffer_priorities = np.full(shape=replay_size, fill_value=0)
        self.replay_size = replay_size
        self.index = 0
        self.size = 0

    def _append(self, transition: tuple):
        for i in range(5):
            self.buffers[i] = self.buffers[i].write(self.index, transition[i])
            
        self.size = min(self.size + 1, self.replay_size)
        self.index = (self.index + 1) % self.replay_size

    # Edits sample in place
    def _sample(self, sample: list, batch_size: int):
        sample_indices = tf.convert_to_tensor(np.random.choice(self.size, size=batch_size), dtype=tf.int32)
        for i in range(5):
            sample[i] = tf.squeeze(self.buffers[i].gather(sample_indices))