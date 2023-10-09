import numpy as np
import tensorflow as tf

from running_average import RunningAverage

# Create replay buffer classes
    
class ReplayBufferUniform:
    def __init__(self, replay_size: int = 10 ** 5):
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


class ReplayBufferPrioritised:
    def __init__(self, replay_size: int = 10 ** 5, prioritised_replay_size: int = 10 ** 3):
        self.uniform_buffer = ReplayBufferUniform(replay_size)
        self.prioritised_buffer = ReplayBufferUniform(prioritised_replay_size)

        self.episodic_reward = RunningAverage(sample_size=10)

        self.visited_indices = []
        self.cumulative_reward = 0

        self.performance_threshold = -1e8


    def _append(self, transition: tuple):
        self.visited_indices.append(self.uniform_buffer.index)
        self.uniform_buffer._append(transition)

        reward = transition[2]
        done = transition[4]

        self.cumulative_reward += reward

        if done:
            if self.cumulative_reward >= self.performance_threshold:
                for i in self.visited_indices:
                    transition = tuple(self.uniform_buffer.buffers[j].read(i) for j in range(5))
                    self.prioritised_buffer._append(transition)

            else:
                self.episodic_reward._update(self.cumulative_reward)

                mean_episodic_reward = self.episodic_reward._get_running_average(type="mean")
                std_episodic_reward = self.episodic_reward._get_running_average(type="std")

                self.performance_threshold = mean_episodic_reward + std_episodic_reward

            self.visited_indices = []
            self.cumulative_reward = 0


    # Edits sample in place
    def _sample(self, sample: list, batch_size: int):
        transition_batch = [[] for _ in range(5)]
        for _ in range(batch_size):
            if np.random.uniform() < 0.9:
                buffer_instance = self.uniform_buffer
            else:
                buffer_instance = self.prioritised_buffer
            
            sample_index = tf.convert_to_tensor(np.random.choice(buffer_instance.size), dtype=tf.int32)
            for j in range(5):
                transition_sample = buffer_instance.buffers[j].read(sample_index)
                transition_batch[j].append(transition_sample)
                
        for j in range(5):
            sample[j] = tf.concat(transition_batch[j], axis=0)





