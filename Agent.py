import torch
import numpy as np

from DNN import DNN

class Agent():
    def __init__(self, n_states, n_actions):

        self.n_states = n_states
        self.n_actions = n_actions

        self.device = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")
        self.model = DNN().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.gamma = 0.99

        self.losses = []
    
    def act(self, state, epsilon):

        action = 0

        # Convert state data to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Explore
        if torch.rand(1)[0] < epsilon:  
            action = torch.tensor([np.random.choice(range(self.n_actions))]).item()

        # Exploit
        else:
            # Check for weird edge case       
            if type(state) != np.ndarray:
                state = state[0]

            # Calculate Q-values of actions given state
            q_values = self.model(state)

            # Get the best action as determined by the Q-values
            best_q_value = torch.argmax(q_values)
            action = best_q_value.item()

        # Print Q-values for testing
        # 2% chance to log Q-values (hacky version of periodic logging)
        test_q_values = self.model(state).squeeze(0)
        if torch.rand(1)[0] > 0.98:
            print("Q(s, a) = " + str(test_q_values[action].item()))
            print("maxQ(s, a) = " + str(test_q_values.max().item()))

        return action
    
    def learn(self, memory, batch_size):

        # Ensure their are enough memories for a batch
        if len(memory) < batch_size:
            return
        
        # Sample a random batch of memories
        states, actions, next_states, rewards, dones = memory.sample(batch_size)

        # Convert states from tuples of tensors to multi-dimensional tensors
        states = torch.stack(states, dim=0)
        next_states = torch.stack(next_states, dim=0)

        # Convert tuples to 2D tensors to work with batched states data
        actions = torch.tensor(list(actions)).unsqueeze(1)
        rewards = torch.tensor(list(rewards)).unsqueeze(1)
        dones = torch.tensor(list(dones)).unsqueeze(1)

        # Give the DNN the batch of states to generate Q-values, then trim to the actions that were selected
        q_values = self.model(states).gather(1, actions)

        # Give the DNN the batch of next states to generate Q-values, then select the optimal action choice Q-value
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)

        # Use Bellman equation to determine optimal action values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones.long()))

        # Calculate loss from optimal actions and taken actions
        loss = self.loss_fn(q_values, expected_q_values.float())
        
        # 2% chance to log loss (hacky version of periodic logging)
        if torch.rand(1)[0] > 0.98:
            self.losses.append(loss.item())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, file, epsilon):
        checkpoint = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'epsilon': epsilon}
        torch.save(checkpoint, file)

    def load(self, file):
        checkpoint = torch.load(file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['epsilon']