import numpy as np
from numba import njit
from tianshou.data import Batch, ReplayBuffer, Collector, VectorReplayBuffer
from tianshou.data.buffer.manager import _next_index


class HERReplayBuffer(VectorReplayBuffer):
    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.env = env

    def add(self, batch, buffer_ids=None):
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)
        current_index, episode_reward, episode_length, episode_start_index = super().add(batch, buffer_ids)
        episode_dones = np.argwhere(episode_length > 0)[:, 0]
        for i in episode_dones:
            indices = _get_episode_indices(episode_length[i], episode_start_index[i],
                                           self._extend_offset, self.done,
                                           self.last_index, self._lengths)
            episode = self[indices]
            episode.obs, episode.obs_next, episode.rew, episode.done, episode.info = self.env.her(
                episode.obs, episode.obs_next
            )
            for b in episode:
                super().add(b[None, ...], buffer_ids=[buffer_ids[i]])

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


@njit
def _get_episode_indices(length, start, offset, done, last_index, lengths):
    indices = np.empty(length, dtype=np.int64)
    indices[0] = start
    index = np.array([start])
    for i in range(length - 1):
        index = _next_index(index, offset, done, last_index, lengths)
        indices[i + 1] = index[0]
    return indices
