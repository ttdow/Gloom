import numpy as np
import math
import random
from random import randint
import sys
import os
from uuid import uuid4

import gym
import gym_fightingice
from gym_fightingice.envs.Machete import Machete
from gym_fightingice.envs.WakeUp import WakeUp
from gym_fightingice.envs.RL_TEST import RLTEST

import matplotlib.pyplot as plt

import torch 

from classifier import Classifier
from DNN import DNN

from OkiAgent import OkiAgent
from ReplayMemory import ReplayMemory

def main():

    # Check for checkpoint to load - CLI syntax: py neutral.py <filepath>
    # Model saves automatically at the end of n_episodes (hyperparameter below)
    # Can change file output name at the bottom of this function
    file = ""
    if (len(sys.argv) > 1):
        file = str(sys.argv[1])

    # Setup action space
    _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
    #print(_actions.index("STAND_D_DF_FA"))
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
    EPSILON_MAX = 0.99
    EPSILON_MIN = 0.10
    epsilon = EPSILON_MAX

    # Hyperparameters
    batch_size = 16                # Experience replay batch size per round
    targetDNN_soft_update_freq = 2 # Target network soft update frequency
    learning_rate = 0.0000625      # Optimizer learning rate - NOT USED CURRENTLY
    gamma = 0.99                   # Discount rate
    tau = 0.01                     # Target network rate
    alpha = 0.6                    # Priority decay
    n_layers = 1                   # Hidden layers

    # Initialize agent and experience replay memory
    agent = OkiAgent(state.shape[0], len(action_vecs), learning_rate, gamma, tau, alpha, n_layers)
    memory = ReplayMemory(50000)

    # Load model if it exists

    rewards = []
    actions = []

    if False:
        epsilon, rewards = agent.load("oki_019.pt")
        print("Model: " + file + " loaded.")

    # Hyperparameters
    #batch_size = 128
    n_episodes = 50
    n_rounds = 3


    # Flag for round finished
    done = False

    damage_done = []
    damage_taken = []
    wins = 0

    #Training loop
    player_hp_weight = 10
    opp_hp_weight = 10
    for episode in range(n_episodes):
        state = env.reset(p2 = WakeUp)
        round = 0
        total_reward = 0


        prev_opp_state = -1
        prev_player_state = -1
        opp_state = -1

        training = False
        sweep = False
        action_count = 0
        episode_actions = []
        while round < n_rounds:

            #print(dir(env.getP2()))
            #exit()
            opp_state = env.getP2().state
            if type(opp_state) != str and type(prev_opp_state) != str and type(opp_state) != int and type(prev_opp_state) != int:
                if opp_state.equals(env.getP2().gateway.jvm.enumerate.State.DOWN) and training == False:
                    #print('TRAINING START')
                    sweep = False
                    training = True

            #print(state [0], state[1], state[2]) 
            if training == True:
                action_count += 1
                #print("TRAINING")
                action = agent.act(state, epsilon)
                episode_actions.append(action)
                #env.getP2().setAction(action)
                next_state, reward, done, _ = env.step(action)
                reward = 0
                if len(prev_state) == 143 and len(state) == 143:
                    reward = (opp_hp_weight * (state[65] - next_state[65])) - (player_hp_weight * (state[0] - next_state[0])) - (1/900)
                #print('reward: ', reward)
                total_reward += reward
                memory.push(state, action, next_state, reward, done, agent)
                agent.learn(memory, batch_size, done)

                #epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
                if action_count == 90 or (state[0] - next_state[0] > 0):
                    print('TRAINING STOP')
                    training = False
                    action_count = 0
            elif training == False:
                action = agent.act_not_training(state, epsilon)
                if action == 25:
                    sweep = True
                #env.getP2().setAction(action)
                next_state, reward, done, _ = env.step(action)
                if len(state) == 143:
                    reward = (opp_hp_weight * (state[65] - next_state[65])) - (player_hp_weight * (state[0] - next_state[0]))
                    if reward != 0.0 and sweep == False:
                        print(reward)
                        total_reward += reward
                        memory.push(state, action, next_state, reward, done, agent)
                        agent.learn(memory, batch_size, done)

            # Update opponent's last state
            prev_state = state
            prev_opp_state = opp_state
            state = next_state
            if done:
                playerHP = state[0] * 100
                damage_taken.append(100 - playerHP)
                opponentHP = state[65] * 100
                damage_done.append(100 - opponentHP)

                # Log winner
                if playerHP > opponentHP:
                    wins += 1
                round += 1
                state = env.reset(p2=WakeUp)

        print("Total reward: " + str(total_reward))
        print("Epsilon: " + str(epsilon))
        epsilon = EPSILON_MIN + 0.5 * (EPSILON_MAX - EPSILON_MIN) * (1 + math.cos((episode / n_episodes) * math.pi))
        rewards.append(total_reward)
        if episode > 0 and episode % 1 == 0:
            agent.save('./oki_020.pt', epsilon, rewards, wins, damage_done, damage_taken)

    env.close()
    exit()

if __name__ == "__main__":
    main()