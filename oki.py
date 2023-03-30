import numpy as np
import random
from random import randint
import sys
import os
from uuid import uuid4

import gym
import gym_fightingice
from gym_fightingice.envs.Machete import Machete
from gym_fightingice.envs.KickAI import KickAI
from gym_fightingice.envs.WakeUp import WakeUp

import torch 

from classifier import Classifier
from DNN import DNN

from OkiAgent import OkiAgent

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
    agent = OkiAgent(state.shape[0], len(action_vecs))
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

    #Training loop
    player_hp_weight = 1.0
    opp_hp_weight = 0.25
    for episode in range(n_episodes):
        state = env.reset(p2 = WakeUp)
        round = 0
        total_reward = 0

        prev_opp_state = -1
        prev_player_state = -1
        opp_state = -1

        training = False
        action_count = 0
        while round < n_rounds:

            #print(dir(env.getP2()))
            #exit()
            opp_state = env.getP2().state
            if type(opp_state) != str and type(prev_opp_state) != str and type(opp_state) != int and type(prev_opp_state) != int:
                if opp_state.equals(env.getP2().gateway.jvm.enumerate.State.DOWN) and training == False;
                    #print('TRAINING START')
                    training = True
            #print(state [0], state[1], state[2]) 
            if training == True:
                action_count += 1
                #print("TRAINING")
                action = agent.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                reward = 0
                if len(prev_state) == 143 and len(state) == 143:
                    reward = (opp_hp_weight * (prev_state[65] - state[65])) - (player_hp_weight * (prev_state[0] - state[0]))
                print('reward: ', reward)
                total_reward += reward
                memory.push(state, action, next_state, reward, done, agent)
                agent.learn(memory, batch_size)

                epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
                if action_count == 60:
                    #print('TRAINING STOP')
                    training = False
                    action_count = 0
            elif training == False:
                action = agent.act_not_training(state, epsilon)
                next_state, reward, done, _ = env.step(action)
            # Update opponent's last state
            prev_state = state
            prev_opp_state = opp_state

            state = next_state

            if done:
                round += 1
                state = env.reset(p2=WakeUp)

        print("Total reward: " + str(total_reward))
        if episode > 0 and episode % 50 == 0:
            agent.save('./oki_checkpiont.pt', epsilon)

    #agent.save('./oki_checkpoint.pt')

    env.close()
    exit()

if __name__ == "__main__":
    main()