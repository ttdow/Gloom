import numpy as np
import random
import sys
import pandas as pd
import gc
import time

import torch
import gym

from Agent import Agent
from OkiAgent import OkiAgent

import json

from gym_fightingice.envs.Machete import Machete
from gym_fightingice.envs.RL_TEST import RLTEST

from neutral import calc_reward

from ReplayMemory import ReplayMemory

def main():
    neutral_file = "neutral.pt"
    oki_file = "oki_020.pt"

    opponent_file_list = ["neutral.pt"]

    gc.collect()
    gc.disable()


    # Hyperparameters
    batch_size = 16                # Experience replay batch size per round
    targetDNN_soft_update_freq = 2 # Target network soft update frequency
    learning_rate = 0.0000625      # Optimizer learning rate - NOT USED CURRENTLY
    gamma = 0.99                   # Discount rate
    tau = 0.01                     # Target network rate
    alpha = 0.6                    # Priority decay
    n_layers = 1                   # Hidden layers

    _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
    action_strs = _actions.split(" ")
    action_vecs = []

    # Onehot encoding for actions
    for i in range(len(action_strs)):
        v = np.zeros(len(action_strs), dtype=np.float32)
        v[i] = 1
        action_vecs.append(v)

    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="", port=4242, freq_restart_java=100000)
    state = env.reset(p2=RLTEST)
    agent = Agent(state.shape[0], len(action_vecs), learning_rate, gamma, tau, alpha, n_layers)

    #okiAgent = OkiAgent(state.shape[0], len(action_vecs), learning_rate, gamma, tau, alpha, n_layers)
    okiAgent = Agent(state.shape[0], len(action_vecs), learning_rate, gamma, tau, alpha, n_layers)

    opponentAgent = Agent(state.shape[0], len(action_vecs), learning_rate, gamma, tau, alpha, n_layers)
    neutral_memory = ReplayMemory(100000)
    oki_memory = ReplayMemory(100000)

    _, _ = agent.load(neutral_file)
    _, _ = okiAgent.load(oki_file)

    epsilon = 0.10

    batch_size = 256
    n_episodes = 1
    n_rounds = 3

    done = False
    frame_counter = 0
    old_time = time.time()

    rewards = []
    wnners = []

    rewards = []
    winners = []

    results_dict = {}

    player_hp_weight = 10
    opp_hp_weight = 10
    for opponent_file in opponent_file_list:
        results_dict[opponent_file] = {}
        results_dict[opponent_file]['damage_taken'] = []
        results_dict[opponent_file]['damage_done'] = []
        results_dict[opponent_file]['oki_frames'] = []
        results_dict[opponent_file]['wins'] = 0
        results_dict[opponent_file]['actions'] = []
        _, _ = opponentAgent.load(opponent_file)
        for episode in range(n_episodes):
            state = env.reset(p2 = RLTEST)
            round = 0
            total_reward = 0
            prev_opp_state = -1
            opp_state = -1
            prev_state = -1
            oki = False
            action_count = 0

            old_time = time.time()

            while round < n_rounds:
                frame_counter += 1
                oki_count = 0
                if type(state) != np.ndarray:
                    state = state[0]
                opponent_action = opponentAgent.act(state, epsilon)
                env.getP2().setAction(opponent_action)

                opp_state = env.getP2().state
                #print(opp_state)
                if type(opp_state) != str and type(prev_opp_state) != str and type(opp_state) != int and type(prev_opp_state) != int:
                    if opp_state.equals(env.getP2().gateway.jvm.enumerate.State.DOWN) and oki == False:
                        print('START OKI')
                        oki = True
                if oki == True:
                    if type(state) != np.ndarray:
                        state = state[0]
                    oki_count += 1
                    action_count += 1
                    action = okiAgent.act(state, epsilon)
                    results_dict[opponent_file]['actions'].append(action)
                    next_state, _, done, _ = env.step(action)
                    if len(next_state) == 143 and len(state) == 143:
                        reward = (opp_hp_weight * (state[65] - next_state[65])) - (player_hp_weight * (state[0] - next_state[0]))# - (1/900)
                    total_reward += reward
                    oki_memory.push(state, action, next_state, reward, done, OkiAgent)

                    if action_count == 90 or (state[0] - next_state[0] > 0):
                        print('END OKI')
                        oki = False
                        action_count = 0
                elif oki == False:
                    if type(state) != np.ndarray:
                        state = state[0]
                    action = agent.act(state, epsilon)
                    results_dict[opponent_file]['actions'].append(action)

                    next_state, reward, done, _ = env.step(action)
                    opp_state = env.getP2().state
                    reward = calc_reward(env, state, action, next_state, prev_opp_state, opp_state, done)

                    neutral_memory.push(state, action, next_state, reward, done, agent)

                prev_state = state
                prev_opp_state = opp_state
                state = next_state
                if done:
                    okiAgent.learn(oki_memory, batch_size, done)
                    agent.learn(neutral_memory, batch_size, done)
                    new_time = time.time()
                    dt = new_time - old_time
                    old_time = new_time
                    player_hp = state[0] * 100
                    results_dict[opponent_file]['damage_taken'].append(100 - player_hp)
                    opponent_hp = state[65] * 100
                    results_dict[opponent_file]['damage_done'].append(100 - opponent_hp)
                    results_dict[opponent_file]['oki_frames'].append(oki_count)

                    if player_hp > opponent_hp:
                        results_dict[opponent_file]['wins'] += 1
                    frame_counter = 0
                    round += 1

                    print("Wins: " + str(results_dict[opponent_file]['wins']))
                    state = env.reset(p2 = RLTEST)

            _, _ = agent.load(neutral_file)
            _, _ = okiAgent.load(oki_file)
            #print("Total reward: " + str(total_reward))
            rewards.append(total_reward)
            #if episode > 0 and episode % 1 == 0:

    with open('results.json', 'w') as outfile:
        json.dump(results_dict, outfile)

    env.close()
    exit()

if __name__ == "__main__":
    main()