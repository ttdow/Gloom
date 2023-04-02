# CPSC601Project #

## TODO ##
----- DOUBLE DEEP Q NETWORK -----
<s>
<ol>
<li>1. Create a second DNN (identical to first) called target</li>
<li>2. When loading model weights also copy them to the target network (copy.deepcopy)</li>
<li>3. Every training iteration, use the policy network to calculate Q values and the target network to calculate future/estimate/target Q values.</li>
<li>4. Update the policy network every iteration (or round) using batched experience replay, loss function, optimization (i.e. the normal way)</li>
<li>5. Update the target network less frequently (hyperparameter) using a "soft update"
		for current_params, target_params in zip(self.current_net.parameters(), self.target_net.parameters()):
            target_params.data.copy_(self.tau * current_params.data + (1 - self.tau) * target_params.data)</li>
<ol>
<li>5.a zip policy network parameters (weights) and target network parameters (a[i] and b[i] become (a[i], b[i])</li>
<li>5.b For each weight in the two networks, interpolate between the current network weight and the target network weight using the
		hyperparameter tau (soft update coefficient) i.e. 0.01</li>
</ol>
</ol>
</s>

---- SKIP ACT/CACHE IF CANNOT ACT? -----
<ul><li>- Framerate fix likely causes agent to select actions and update Q-values even when those actions have no impact on the state</li></ul>

---- REVIEW DNN ARCHITECTURE -----

---- FIND A WAY TO GET MORE OPPONENT AIs ----
<ul>
<li>- Replicate and train the RL agents described in the Halina paper</li>
<li>- Dig through competition AIs for ones we can replicate</li>
</ul>

---- FIX GPU IMPLEMENTATION -----
<ul><li>- Track down tensors not being updated on GPU (I think this is the issue?)</li></ul>

---- UPDATE LOGGABLE METRICS -----
<ul>
<li>- Add more metrics to be logged</li>
<li>- Time spent in each model in the combined</li>
<li>- Round win rate in combined</li>
<li>- Damage done/taken</li>
<li>- Actions taken</li>
</ul>

----- PRIORITY REPLAY BUFFER -----
<ol>
<li>1. Modify or overload the learn function so it can take 1 or batch_size experiences</li>
<li>2. Call learn on each (state, action) transition during the training loop to calculate the expected Q-value every time</li>
<li>2. Save the temporal difference for each transition as the priority in the experience replay buffer as well</li>
<li>3. At the end of the round, sample a batch of transitions from the buffer using the priority to determine the probability
	of sampling that particular experience</li>
<li>4. Adjust the priority of the sampled transitons so they are less likely to be sampled in the future
	priority = (TD_error + epsilon) ** alpha, where epsilon is some small positive constant and alpha is a hyperparameter
		controlling the degree of prioritization</li>
</ol>

----- DUELING DEEP Q NETWORK -----
<ol>
<li>1. When calculating the Q-value for a (state, action) pair we calculate two outputs: 1.) advantage and 2.) state</li>
	<ol><li>1.a. Advantage: this is the same as the normal output for a DQN (the action space Q-values given the state)</li>
	<li>1.b. State: this is the value of being in the current state (1 normalized value)</li></ol>
<li>2. Then both the state value and advantage values are used to calculate the final Q-value
	q_values = state_value + advantage_values - advantage.mean() 
		This way being in a certain state can provide a reward of its own</li>
</ol>
