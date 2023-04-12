import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, n_states, n_actions, n_layers):
        super(DNN, self).__init__()

        self.n_obs = n_states
        self.n_act = n_actions
        self.hidden_size = 128
        self.n_layers = n_layers # TODO this doesn't change anything yet

        # Input layer
        self.fc1 = nn.Linear(self.n_obs, self.hidden_size)
        
        # Advantage (actions) network branch
        self.fc2_adv = nn.Linear(self.hidden_size, self.hidden_size)
        self.advantage = nn.Linear(self.hidden_size, self.n_act)

        # Value (states) network branch
        self.fc2_val = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, 1)

    def forward(self, x):

        # Input layer
        x = F.relu(self.fc1(x))

        # Advantage (actions) network branch
        adv = F.relu(self.fc2_adv(x))
        adv = self.advantage(adv)

        # Value (states) network branch
        val = F.relu(self.fc2_val(x))
        val = self.value(val)#.expand(x.size(0), self.advantage.out_features)

        # Normalize advantage values about 0 and add to state value for each possible 
        # action to calculate the Q-values
        if x.ndim > 1: # For batches (i.e. learning)
            x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.advantage.out_features)
        else: # For single values (i.e. action selection and priority calculation)
            x = val + adv - adv.mean(0)

        # Return Q-values of each possible action given state x
        return x