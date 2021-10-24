import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, sampling_size, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.con_state = np.zeros((max_size, 1, sampling_size, state_dim))
        self.con_next_state = np.zeros((max_size, 1, sampling_size, state_dim))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, con_state, con_next_state):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.con_state[self.ptr] = con_state
        self.con_next_state[self.ptr] = con_next_state
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size < 55000:
            ind = np.random.randint(0, self.size, size=batch_size)
        else:
            ind1 = np.random.randint(0, self.size - 50000, size=int(batch_size * 0.1))
            ind2 = np.random.randint(self.size - 50000, self.size-20000, size=int(batch_size * 0.15))
            ind3 = np.random.randint(self.size - 20000, self.size, size=int(batch_size * 0.75))
            ind = np.hstack((ind1, ind2, ind3))
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.con_state[ind]).to(self.device),
            torch.FloatTensor(self.con_next_state[ind]).to(self.device)
        )
