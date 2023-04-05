# CPSC601Project #

## TODO ##

---- REVIEW DNN ARCHITECTURE -----
<ul>
<li>Research?</li>
</ul>

---- FIND A WAY TO GET MORE OPPONENT AIs ----
<ul>
<li><s>Replicate and train the RL agents described in the Halina paper</s></li>
<li>Dig through competition AIs for ones we can replicate</li>
</ul>

---- FIX GPU IMPLEMENTATION -----
<ul><li>Track down tensors not being updated on GPU (I think this is the issue?)</li></ul>

---- UPDATE LOGGABLE METRICS -----
<ul>
<li><s>Add more metrics to be logged</s></li>
<li>Time spent in each model in the combined</li>
<li><s>Round win rate</s></li>
<li><s>Damage done/taken</s></li>
<li>Actions taken</li>
</ul>

----- DUELING DEEP Q NETWORK -----
<ol>
<li>When calculating the Q-value for a (state, action) pair we calculate two outputs: 1.) advantage and 2.) state</li>
	<ol><li>Advantage: this is the same as the normal output for a DQN (the action space Q-values given the state)</li>
	<li>State: this is the value of being in the current state (1 normalized value)</li></ol>
<li>Then both the state value and advantage values are used to calculate the final Q-value<br>
	q_values = state_value + advantage_values - advantage.mean() 
		This way being in a certain state can provide a reward of its own</li>
</ol>

----- DISTRIBUTIONAL Q NETWORK -----

----- NOISY NETS -----

----- SOFT Q-LEARNING -----

----- EVOLUTIONARY HYPERPARAMETER OPTIMIZATION -----