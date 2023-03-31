# CPSC601Project #

## TODO ##
<s>
----- DOUBLE DEEP Q NETWORK -----
1. Create a second DNN (identical to first) called target
2. When loading model weights also copy them to the target network (copy.deepcopy)
3. Every training iteration, use the policy network to calculate Q values and the target network to calculate future/estimate/target Q values.
4. Update the policy network every iteration (or round) using batched experience replay, loss function, optimization (i.e. the normal way)
5. Update the target network less frequently (hyperparameter) using a "soft update"
		for current_params, target_params in zip(self.current_net.parameters(), self.target_net.parameters()):
            target_params.data.copy_(self.tau * current_params.data + (1 - self.tau) * target_params.data)

	5.a zip policy network parameters (weights) and target network parameters (a[i] and b[i] become (a[i], b[i])
	5.b For each weight in the two networks, interpolate between the current network weight and the target network weight using the
		hyperparameter tau (soft update coefficient) i.e. 0.01</s>

---- SKIP ACT/CACHE IF CANNOT ACT? -----
- Framerate fix likely causes agent to select actions and update q-values even when those actions have no impact on the state

---- REVIEW DNN ARCHITECTURE -----

---- FIND A WAY TO GET MORE OPPONENT AIS ----
- Train the RL agents described in the Halina paper
- Dig through competition AIs for ones we can replicate

---- FIX GPU IMPLEMENTATION -----
- Track down tensors not being updated on GPU (I think this is the issue?)

---- UPDATE LOGGABLE METRICS -----
- Add more metrics to be logged
- Time spent in each model in the combined
- Round win rate in combined
- Damage done/taken
- Actions taken

----- PRIORITY REPLAY BUFFER -----
1. Modify or overload the learn function so it can take 1 or batch_size experiences
2. Call learn on each (state, action) transition during the training loop to calculate the expected Q-value every time
2. Save the temporal difference for each transition as the priority in the experience replay buffer as well
3. At the end of the round, sample a batch of transitions from the buffer using the priority to determine the probability
	of sampling that particular experience
4. Adjust the priority of the sampled transitons so they are less likely to be sampled in the future
	priority = (TD_error + epsilon) ** alpha, where epsilon is some small positive constant and alpha is a hyperparameter
		controlling the degree of prioritization

----- DUELING DEEP Q NETWORK -----
1. When calculating the Q-value for a (state, action) pair we calculate two outputs: 1.) advantage and 2.) state
	1.a. Advantage: this is the same as the normal output for a DQN (the action space Q-values given the state)
	1.b. State: this is the value of being in the current state (1 normalized value)
2. Then both the state value and advantage values are used to calculate the final Q-value
	q_values = state_value + advantage_values - advantage.mean() 
		This way being in a certain state can provide a reward of its own 