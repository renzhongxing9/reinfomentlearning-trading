from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    def push(self, state, action, reward, next_state):
        # placeholder
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # 向memory中添加经验回放
        self.memory[self.index] = Transition(state, action, reward, next_state)
        self.index = (self.index + 1) % self.capacity

