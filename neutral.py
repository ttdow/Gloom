import numpy as np
import random
from random import randint
import sys
import os
from uuid import uuid4
from collections import namedtuple

import gym
import gym_fightingice
from gym_fightingice.envs.Machete import Machete

import torch

from classifier import Classifier
from DNN import DNN

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Agent():
    def __init__(self, n_obs, n_act):

        self.n_obs = n_obs
        self.n_act = n_act

        self.device = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")
        self.model = DNN().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.gamma = 0.99
    
    def act(self, obs, epsilon):

        max_q_value = 0

        if torch.rand(1)[0] < epsilon:
            # Explore
            max_q_value = torch.tensor([np.random.choice(range(self.n_act))]).item()
        else:
            # Exploit
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.model(obs)
            max_q_value = torch.argmax(q_values).item()

        return max_q_value
    
    def learn(self, memory, batch_size):

        #if len(memory) < batch_size:
        #    return
        
        #transitions = memory.sample(batch_size) # (states, actions, next_states, rewards, dones)
        #batch = Transition(*zip(*transitions))

        states, actions, rewards, next_states, dones = memory.sample(1)

        print(type(states))
        if type(states) == np.ndarray:
            states = torch.tensor(states)
        else:
            return

        print("New state: " + str(type(states)))

        """
        state_batch = torch.tensor(states).to(self.device)
        action_batch = torch.tensor(actions).to(self.device)
        reward_batch = torch.tensor(rewards).to(self.device)
        next_state_batch = torch.tensor(next_states).to(self.device)
        done_batch = torch.tensor(dones).to(self.device)

        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.model(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        """
    
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        #batch = random.sample(self.memory, 1)#, batch_size)
        #states, actions, next_states, rewards, dones = zip(*batch)

        idx = random.randint(0, len(self)-1)
        states, actions, next_states, rewards, dones = self.memory[idx]

        return states, actions, next_states, rewards, dones
   
def main():

    # Setup action space
    _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
    action_strs = _actions.split(" ")
    action_vecs = []

    # Onehot encoding for actions
    for i in range(len(action_strs)):
        v = np.zeros(len(action_strs), dtype=np.float32)
        v[i] = 1
        action_vecs.append(v)

    x = torch.tensor(action_vecs)

    print("Action space length: ", len(action_vecs)) # 56

    # Setup observation space
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="", port=4242, freq_restart_java=100000)
    state = env.reset(p2=Machete)

    print(state.shape)    # (143, )
    print("Observation space length: ", state.shape[0]) # 143
    print(type(state[0])) # numpy.float64

    EPSILON_MAX = 0.95
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.05
    epsilon = EPSILON_MAX

    # Test model
    agent = Agent(state.shape[0], len(action_vecs))
    memory = ReplayMemory(10000)
    action = agent.act(state, epsilon) # int

    print(type(action))
    print("y = ", action)

    batch_size = 128

    done = False

    n_episodes = 100
    n_rounds = 3

    for episode in range(n_episodes):

        state = env.reset(p2=Machete)
        round = 0
        total_reward = 0

        while round < n_rounds:

            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            memory.push(state, action, next_state, reward, done)
            state = next_state

            agent.learn(memory, batch_size)

            epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

            if done:
                round += 1
                state = env.reset(p2=Machete)

        print("Total reward: " + str(total_reward))
        print("Memory size: " + str(len(memory)))

    env.close()
    exit()

if __name__ == "__main__":
    main()