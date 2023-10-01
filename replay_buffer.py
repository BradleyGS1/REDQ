import numpy as np
import tensorflow as tf

# Create replay buffer class

class ReplayBuffer:
    def __init__(self, replay_size: int):
        self.buffer = [None] * replay_size
        #self.buffer_priorities = np.full(shape=replay_size, fill_value=0)
        self.replay_size = replay_size
        self.index = 0
        self.size = 0

    def _append(self, transition: tuple):
        self.buffer[self.index] = transition
        self.size = min(self.size + 1, self.replay_size)
        self.index = (self.index + 1) % self.replay_size

    def _sample(self, batch_size: int):
        sample_indices = np.random.choice(range(self.size), size=batch_size)

        states = tf.TensorArray(dtype=tf.float32, size=batch_size)
        actions = tf.TensorArray(dtype=tf.float32, size=batch_size)
        rewards = tf.TensorArray(dtype=tf.float32, size=batch_size)
        new_states = tf.TensorArray(dtype=tf.float32, size=batch_size)
        done_flags = tf.TensorArray(dtype=tf.float32, size=batch_size)

        for i, sample_index in enumerate(sample_indices):
            state, action, reward, new_state, done = self.buffer[sample_index]

            states = states.write(i, tf.squeeze(state))
            actions = actions.write(i, tf.squeeze(action))
            rewards = rewards.write(i, tf.squeeze(reward))
            new_states = new_states.write(i, tf.squeeze(new_state))
            done_flags = done_flags.write(i, tf.squeeze(done))

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()
        new_states = new_states.stack()
        done_flags = done_flags.stack()

        sample = (states, actions, rewards, new_states, done_flags)

        return sample