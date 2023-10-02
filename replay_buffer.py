import numpy as np
import tensorflow as tf

# Create replay buffer class

class ReplayBuffer:
    def __init__(self, replay_size: int):
        self.states_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.actions_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.rewards_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.new_states_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.done_flags_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)

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

    @tf.function
    def _sample(self, batch_size: int):
        sample_indices = tf.random.categorical(tf.zeros(shape=(batch_size, self.size), dtype=tf.float32), num_samples=1, dtype=tf.int32)

        states = tf.squeeze(tf.gather(self.states_buffer.stack(), sample_indices, axis=0, batch_dims=True))
        actions = tf.squeeze(tf.gather(self.actions_buffer.stack(), sample_indices, axis=0, batch_dims=True))
        rewards = tf.squeeze(tf.gather(self.rewards_buffer.stack(), sample_indices, axis=0, batch_dims=True))
        new_states = tf.squeeze(tf.gather(self.new_states_buffer.stack(), sample_indices, axis=0, batch_dims=True))
        done_flags = tf.squeeze(tf.gather(self.done_flags_buffer.stack(), sample_indices, axis=0, batch_dims=True))

        sample = (states, actions, rewards, new_states, done_flags)

        return sample