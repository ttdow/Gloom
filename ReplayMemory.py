import random
import numpy as np

import torch

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priority = np.empty(self.capacity)
        self.position = 0

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, state, action, next_state, reward, done, agent):
        
        # Make more room in memory if needed
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # Not sure why this happens        
        if type(state) != np.ndarray:
            state = state[0]

        # Convert data from ndarray to tensor for ease of use
        state = torch.from_numpy(state).float().to(torch.device("cpu"))
        next_state = torch.from_numpy(next_state).float().to(torch.device("cpu"))

        # Calculate importance/priority of this memory (i.e. td error)
        #print(state.shape, action, next_state.shape, reward, done)
        
        priority = agent.prioritize(state, action, next_state, reward, done)
        self.priority[self.position] = priority

        # Save a new memory to circular buffer
        self.memory[self.position] = (state, action, next_state, reward, done)

        # Cycle through circular buffer
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        # Grab <batch_size> random samples of memories
        batch = random.sample(self.memory, batch_size)

        # Zip the unpacked sample of memories
        states, actions, next_states, rewards, dones = zip(*batch)

        return states, actions, next_states, rewards, dones