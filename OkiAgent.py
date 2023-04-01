from DNN import DNN
import torch 
import numpy as np
import copy

class OkiAgent():

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        self.device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DNN().to(self.device)       # Used for calculating current Q-values during training 
        self.target =  copy.deepcopy(self.model) # Used for calculating target Q-values during training

        # Freeze parameters in target network - we update the target network manually
        for p in self.target.parameters():
            p.requires_grad = False

        # Hyperparameters
        self.learning_rate = 1e-3   # Learning rate used for gradient descent
        self.gamma = 0.99           # Discount rate for future Q-value estimates
        self.tau = 0.01             # Soft update coefficient for target network

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

        self.losses = []

    def soft_update_target_network(self):

        # Iterate through each weight in both the current and target DNNs (they are identically structured)
        for current_parameter, target_parameter in zip(self.model.parameters(), self.target.parameters()):
            
            # Update the data value of the weight by interpolating between the current weight and target weight
            #   This smooths the update of the target network and *should* result in more consistency and stability
            target_parameter.data.copy_(self.tau * current_parameter + (1 - self.tau) * target_parameter)

    def act_not_training(self, state, epsilon):
        action = 29
        if torch.rand(1)[0] < 0.2:
            action = 25
        return action

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

        # Print Q-values for testing
        # 2% chance to log Q-values (hacky version of periodic logging)
        #test_q_values = self.model(state).squeeze(0)
        #if torch.rand(1)[0] > 0.98:
            #print("Q(s, a) = " + str(test_q_values[action].item()))
            #print("maxQ(s, a) = " + str(test_q_values.max().item()))

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

        actions = torch.tensor(list(actions)).unsqueeze(1)
        rewards = torch.tensor(list(rewards)).unsqueeze(1)
        dones = torch.tensor(list(dones)).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)

        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)

        # Use Bellman equation to determine optimal action values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones.long()))

        # Calculate loss from optimal actions and taken actions
        loss = self.loss_fn(q_values, expected_q_values.float())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, file, epsilon, rewards):
        checkpoint = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'epsilon': epsilon,
                      'rewards': rewards,
                      'losses': self.losses}
                      
        torch.save(checkpoint, file)

    def load(self, file):
        checkpoint = torch.load(file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint['epsilon']