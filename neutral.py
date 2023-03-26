import numpy as np
import random
from random import randint
import sys
import os
from uuid import uuid4
from collections import namedtuple
import math

import gym
import gym_fightingice
from gym_fightingice.envs.Machete import Machete
from gym_fightingice.envs.KickAI import KickAI
from gym_fightingice.envs.WakeUp import WakeUp

import torch

from classifier import Classifier
from DNN import DNN

class Agent():
    def __init__(self, n_obs, n_act):

        self.n_obs = n_obs
        self.n_act = n_act

        self.device = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")
        self.model = DNN().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.gamma = 0.99
    
    def act(self, state, epsilon):

        action = 0

        # Explore
        if torch.rand(1)[0] < epsilon:  
            action = torch.tensor([np.random.choice(range(self.n_act))]).item()
        # Exploit
        else:
            # Check for weird edge case       
            if type(state) != np.ndarray:
                state = state[0]

            # Convert state data to tensor
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Calculate Q-values of actions given state
            q_values = self.model(state)

            # Get the best action as determined by the Q-values
            best_q_value = torch.argmax(q_values)
            action = best_q_value.item()

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

        # Give the DNN the batch of states to generate Q-values
        q_values = self.model(states) 
        next_q_values = self.model(next_states)

        # Convert tuples to 2D tensors to work with batched states data
        rewards = torch.tensor(list(rewards)).unsqueeze(1)
        dones = torch.tensor(list(dones)).unsqueeze(1)

        # Use Bellman equation to determine optimal action values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones.long()))

        # Calculate loss from optimal actions and taken actions
        loss = self.loss_fn(q_values, expected_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, file):
        checkpoint = {'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, file)

    def load(self, file):
        checkpoint = torch.load(file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
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
        state = torch.from_numpy(state).float().to(agent.device)
        next_state = torch.from_numpy(next_state).float().to(agent.device)

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
    
def calc_reward(env, env_state, action, next_env_state, prev_opp_state, opp_state):

    reward = 0

    if type(env_state) != np.ndarray:
        env_state = env_state[0]

    #print(env_state[92:97])

    #print(type(env_state))
    #print(env_state.shape)

    playerX = env_state[2]
    opponentX = env_state[67]

    #print(env_state[65])
    #print(env_state[65] - prev_state[65])

    #print("Player X: " + str(playerX))
    #print("Opponent X: " + str(opponentX))
    dist = abs(playerX - opponentX) * 960
    #print("Distance: " + str(int(dist)))

    if type(opp_state) != str:
        if opp_state.equals(env.getP2().gateway.jvm.enumerate.State.DOWN) and opp_state != prev_opp_state:
            #print(opp_state)
            #print('---')
            reward += 1000

    return reward
   
def main():

    # Check for checkpoint to load - CLI syntax: py neutral.py <filepath>
    # Model saves automatically at the end of n_episodes (hyperparameter below)
    # Can change file output name at the bottom of this function
    file = ""
    if (len(sys.argv) > 1):
        file = str(sys.argv[1])

    # Setup action space
    _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
    action_strs = _actions.split(" ")
    action_vecs = []

    # Onehot encoding for actions
    for i in range(len(action_strs)):
        v = np.zeros(len(action_strs), dtype=np.float32)
        v[i] = 1
        action_vecs.append(v)

    # Setup observation space
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="", port=4242, freq_restart_java=100000)
    state = env.reset(p2=WakeUp)

    # Setup epsilon values for explore/exploit calcs
    EPSILON_MAX = 0.95
    EPSILON_DECAY = 0.99999975
    EPSILON_MIN = 0.05
    epsilon = EPSILON_MAX

    # Initialize agent and experience replay memory
    agent = Agent(state.shape[0], len(action_vecs))
    memory = ReplayMemory(50000)

    # Load model if it exists
    if file != "":
        agent.load(file)
        print("Model: " + file + " loaded.")

    # Hyperparameters
    batch_size = 128
    n_episodes = 100
    n_rounds = 3

    # Flag for round finished
    done = False

    # Training loop
    for episode in range(n_episodes):

        state = env.reset(p2=WakeUp)
        round = 0
        total_reward = 0

        prev_opp_state = -1
        prev_state = state
        while round < n_rounds:
            
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            #print(state[65] - prev_state[65])
            # Get opponent's current state from env (STAND, CROUCH, AIR, DOWN)
            opp_state = env.getP2().state

            if len(state) == 143 and len(prev_state) == 143:
                print(prev_state[65] - state[65])

            calc_reward(env, state, action, next_state, prev_opp_state, opp_state)

            # Update opponent's last state
            prev_state = state
            prev_opp_state = opp_state

            total_reward += reward
            memory.push(state, action, next_state, reward, done, agent)
            state = next_state

            agent.learn(memory, batch_size)

            epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

            if done:
                round += 1
                state = env.reset(p2=KickAI)

        print("Total reward: " + str(total_reward))

    # Save this model
    agent.save('./checkpoint.pt')

    env.close()
    exit()

if __name__ == "__main__":
    main()