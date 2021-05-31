import gym
import numpy as np
from tianshou.data import Batch, ReplayBuffer, Collector


class HERReplayBuffer(ReplayBuffer):
    def __init__(self, size, env, **kwargs):
        super().__init__(size, **kwargs)
        self.env = env

    def add(self, batch, buffer_ids=None):
        current_index, episode_reward, episode_length, episode_start_index = super().add(batch, buffer_ids)
        if episode_length[0] > 0:
            episode = self[episode_start_index[0]:episode_start_index[0] + episode_length[0]]
            episode.obs, episode.obs_next, episode.rew, episode.done = self.env.her(
                episode.obs, episode.obs_next
            )
            for b in episode:
                super().add(b)
        return current_index, episode_reward, episode_length, episode_start_index

    # def sample_index(self, batch_size):
    #     num_her = int(np.ceil(self.her_sample_proportion * batch_size))
    #     num_reg = batch_size - num_her
    #     return super().sample_index(num_reg), self.her_buffer.sample_index(num_her)
    #
    # def sample(self, batch_size):
    #     indices = self.sample_index(batch_size)
    #     batch = self[indices]
    #     print(batch)
    #     return self[indices], indices
