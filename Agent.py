import torch
import numpy as np
import time
import copy

from torch.optim.lr_scheduler import CosineAnnealingLR

from DNN import DNN

class Agent():
    def __init__(self, n_states, n_actions, lr, gamma, tau, alpha, n_layers):

        self.n_states = n_states
        self.n_actions = n_actions

        self.device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DNN(n_states, n_actions, n_layers).to(self.device) # Used for calculating policy Q-values
        self.target = copy.deepcopy(self.model) # Used for calculating target Q-values

        # Freeze parameters in target network - we update the target network manually
        for p in self.target.parameters():
            p.requires_grad = False

        # Hyperparameters
        self.learning_rate = lr # Learning rate used for gradient descent
        self.gamma = gamma      # Discount rate for future Q-value estimates
        self.tau = tau          # Soft update coefficient for target network
        self.alpha = alpha      # Controls degree of prioritization

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0.00001)
        self.loss_fn = torch.nn.MSELoss()

        self.losses = []

    def soft_update_target_network(self):

        # Iterate through each weight in both the current and target DNNs (they are identically structured)
        for current_parameter, target_parameter in zip(self.model.parameters(), self.target.parameters()):
            
            # Update the data value of the weight by interpolating between the current weight and target weight
            #   This smooths the update of the target network and *should* result in more consistency and stability
            target_parameter.data.copy_(self.tau * current_parameter + (1 - self.tau) * target_parameter)
    
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

            # Calculate Q-values of actions given current state using the current model
            q_values = self.model(state)

            # Get the best action as determined by the Q-values
            best_q_value = torch.argmax(q_values)
            action = best_q_value.item()

        return action
    
    def prioritize(self, state, action, next_state, reward, done):

        # Calculate Q-value of state, action transition
        q_value = self.model(state)[action]

        # Calculate Q-value of next state, best action transition
        next_q_value = self.target(next_state).max(0)[0]

        # Calculate expected Q-value using Bellman equation
        expected_q_value = reward + (self.gamma * next_q_value * (1 - int(done)))

         # Calculate TD error for prioritized replay
        TDError = expected_q_value.float() - q_value

        return abs(TDError)
    
    def learn(self, memory, batch_size, round_end):

        # Ensure their are enough memories for a multi_step batch (i.e. batch_size * 2)
        mem_size = len(memory)
        double_batch = batch_size * 2
        if mem_size <= double_batch:
            return

        # If memory not full, only take up to the current memory size of priorities - buffer
        if mem_size < memory.capacity:
            priorities = memory.priority[:mem_size - double_batch]
        else:
            priorities = memory.priority[:double_batch]

        # Calculate a probability using the priority value
        priorities_sum = priorities.sum()
        probs = priorities / priorities_sum

        # Grab a random memory using the priority probability
        idx = np.random.choice(mem_size - double_batch, 1, p=probs, replace=False)[0]

        # Stack the selected memory and the next batch_size * 2 memories in time in to a batch
        transitions = [memory.memory[idx + i] for i in range(0, double_batch)]

        # Update priorities of the selected batch_size memories to decrease future priority
        for i in range(0, batch_size):
            memory.priority[idx + i] = (memory.priority[idx + i] + 1e-5) ** self.alpha

        # Iterate through the first half of the list
        for i in range(0, batch_size):
           
            # Compound the reward of each memory using the next time delayed reward
            # of the next batch_size memories
            state, action, next_state, reward, done = transitions[i]
            for j in range(i+1, i+batch_size):
                _, _, next_state, r, _ = transitions[j]
                reward += (self.gamma ** (j-i)) * r

            # Resave the transitions with the new reward and the final state
            transitions[i] = state, action, next_state, reward, done

        # Reorganize batch data for processing
        states, actions, next_states, rewards, dones = zip(*transitions[:batch_size-1])

        # Convert states from tuples of tensors to multi-dimensional tensors
        states = torch.stack(states, dim=0).to(self.device)
        next_states = torch.stack(next_states, dim=0).to(self.device)

        # Convert tuples to 2D tensors to work with batched states data
        actions = torch.tensor(list(actions)).unsqueeze(1).to(self.device)
        rewards = torch.tensor(list(rewards)).unsqueeze(1).to(self.device)
        dones = torch.tensor(list(dones)).unsqueeze(1).to(self.device)

        # Give the CURRENT DNN the batch of states to generate Q-values, then trim to the actions that were selected
        q_values = self.model(states).gather(1, actions)

        # Give the TARGET DNN the batch of next states to generate Q-values, then select the optimal action choice Q-value
        next_q_values = self.target(next_states).max(1)[0].unsqueeze(1)

        # Use Bellman equation to determine optimal action values using the TARGET DNN
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones.long()))

        # Calculate loss from optimal actions and taken actions
        loss = self.loss_fn(q_values, expected_q_values.float())

        # Log loss
        if round_end:
            self.losses.append(loss.item())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the learning rate
        self.scheduler.step()

        return

    def save(self, file, epsilon, rewards, wins, damage_done, damage_taken):

        checkpoint = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'epsilon': epsilon,
                      'rewards': rewards,
                      'losses': self.losses,
                      'wins': wins,
                      'damage_done': damage_done,
                      'damage_taken': damage_taken}
        
        torch.save(checkpoint, file)

    def load(self, file):

        checkpoint = torch.load(file, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])         # Load current DNN weights
        self.target =  copy.deepcopy(self.model)                # Update target DNN weights
        self.optimizer.load_state_dict(checkpoint['optimizer']) # Update optimizer weights

        if "losses" in checkpoint:
            self.losses = checkpoint['losses']

        if "rewards" in checkpoint:
            return checkpoint['epsilon'], checkpoint['rewards']
        else:
            return checkpoint['epsilon'], list()