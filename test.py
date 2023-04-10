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
        state = torch.from_numpy(state).float().to(torch.device("cpu"))
        next_state = torch.from_numpy(next_state).float().to(torch.device("cpu"))

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

def main():
    neutral_file = "neutral4.pt"
    oki_file = "oki_019.pt"

    opponent_file_list = ["aggressive.pt"]

    gc.collect()
    gc.disable()


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
    agent = Agent(state.shape[0], len(action_vecs))
    okiAgent = OkiAgent(state.shape[0], len(action_vecs))

    opponentAgent = Agent(state.shape[0], len(action_vecs))
    memory = ReplayMemory(100000)

    epsilon, _ = agent.load(neutral_file)
    _, _ = okiAgent.load(oki_file)

    EPSILON_MAX = 0.99
    EPSILON_DECAY = 0.9999995
    EPSILON_MIN = 0.00

    batch_size = 256
    n_episodes = 100
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
                    oki_count += 1
                    action_count += 1
                    action = okiAgent.act(state, epsilon)
                    next_state, _, done, _ = env.step(action)
                    if len(next_state) == 143 and len(state) == 143:
                        reward = (opp_hp_weight * (state[65] - next_state[65])) - (player_hp_weight * (state[0] - next_state[0])) - (1/900)
                    total_reward += reward
                    memory.push(state, action, next_state, reward, done, OkiAgent)
                    okiAgent.learn(memory, batch_size)

                    if action_count == 90 or (state[0] - next_state[0] > 0):
                        print('END OKI')
                        oki = False
                        action_count = 0
                elif oki == False:
                    if type(state) != np.ndarray:
                        state = state[0]
                    action = agent.act(state, epsilon)

                    next_state, reward, done, _ = env.step(action)
                    opp_state = env.getP2().state
                    reward = calc_reward(env, state, action, next_state, prev_opp_state, opp_state, done)

                    memory.push(state, action, next_state, reward, done, agent)
                    agent.learn(memory, batch_size)

                epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
                prev_state = state
                prev_opp_state = opp_state
                state = next_state
                if done:
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

            print("Total reward: " + str(total_reward))
            rewards.append(total_reward)
            #if episode > 0 and episode % 1 == 0:

    with open('results.json', 'w') as outfile:
        json.dump(results_dict, outfile)

    env.close()
    exit()

if __name__ == "__main__":
    main()