import numpy as np


class ReplayBuffer:

    def __init__(self, size, obs_size, n_nodes):
        self.size = size
        self.obs = np.zeros([self.size, obs_size], dtype=np.float32)
        self.adj = np.zeros([self.size, n_nodes, n_nodes], dtype=np.float32)
        self.weight_matrix = np.zeros([self.size, n_nodes, n_nodes], dtype=np.float32)
        self.next_obs = np.zeros([self.size, obs_size], dtype=np.float32)
        self.actions = np.zeros([self.size], dtype=np.int32)
        self.rewards = np.zeros([self.size], dtype=np.float32)
        self.done = np.zeros([self.size], dtype=np.bool)
        self.transition_length = np.zeros([self.size], dtype=np.int32)
        self.num_in_buffer = 0
        self.next_idx = 0


    def store_transition(self, obs, adj, weight_matrix, action, reward, next_obs, done, transition_length):
        self.obs[self.next_idx] = obs
        self.adj[self.next_idx] = adj
        self.weight_matrix[self.next_idx] = weight_matrix
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = reward
        self.next_obs[self.next_idx] = next_obs
        self.done[self.next_idx] = done
        self.transition_length[self.next_idx] = transition_length

        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        self.next_idx = (self.next_idx + 1) % self.size

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions
         can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = np.random.choice(self.num_in_buffer, batch_size)
        return self.obs[idxes], \
               self.adj[idxes], \
               self.weight_matrix[idxes], \
               self.actions[idxes], \
               self.rewards[idxes], \
               self.next_obs[idxes], \
               1 - np.array(self.done[idxes], dtype=np.int32), \
               self.transition_length[idxes]
