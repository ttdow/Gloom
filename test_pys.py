import numpy as np
import random
import sys
import pandas as pd
import gc
import time
import json

import torch
import gym
from gym_fightingice.envs.RL_TEST import RLTEST

from neutral import calc_reward
from Agent import Agent
from ReplayMemory import ReplayMemory

def main():

    # Force garbage collection and disable to reduce frame rate effects
    gc.collect()
    gc.disable()

    # Define pre-trained model files
    neutral_file = "neutral_training4.pt"
    oki_file = "oki_training4.pt"

    # Hyperparameters
    batch_size = 16 # Experience replay batch size per round
    gamma = 0.99    # Discount rate
    tau = 0.01      # Target network rate
    alpha = 0.6     # Priority decay
    n_layers = 1    # Hidden layers

    # Constant epsilon during testing
    epsilon = 0.10

    # Fight parameters
    batch_size = 16
    n_episodes = 100
    n_rounds = 3

    # Prepare vector of actions
    _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
    action_strs = _actions.split(" ")
    action_vecs = []

    # Onehot encoding for actions
    for i in range(len(action_strs)):
        v = np.zeros(len(action_strs), dtype=np.float32)
        v[i] = 1
        action_vecs.append(v)
    action_space = len(action_vecs)

    # Generate gym environment
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="", port=4242, freq_restart_java=100000)

    # Define pre-trained opponent model files
    opponent_file_list = ["aggressive.pt", "balanced.pt"]

    # Test against each AI opponent in the list
    for opponent_file in opponent_file_list:

        # Setup testing environment
        state = env.reset(p2=RLTEST)
        if type(state) == list:
            state_space = len(state[0])
        else:
            state_space = state.shape[0]

        # Create our test models
        neutral_agent = Agent(state_space, action_space, 0.0, gamma, tau, alpha, n_layers, n_episodes)
        oki_agent = Agent(state_space, action_space, 0.0, gamma, tau, alpha, n_layers, n_episodes)

        # Create opponent AI model
        opponent_agent = Agent(state_space, action_space, 0.0, gamma, tau, alpha, n_layers, n_episodes)

        # Set memory cache size
        neutral_memory = ReplayMemory(100000)
        oki_memory = ReplayMemory(100000)

        # Load pre-trained models
        neutral_agent.load(neutral_file)
        oki_agent.load(oki_file)
        opponent_agent.load(opponent_file)

        # Containers for loggable metrics
        rewards = []
        results_dict = {}
        results_dict[opponent_file] = {}
        results_dict[opponent_file]['damage_taken'] = []
        results_dict[opponent_file]['damage_done'] = []
        results_dict[opponent_file]['oki_frames'] = []
        results_dict[opponent_file]['wins'] = 0
        results_dict[opponent_file]['actions'] = []

        # HP weights for Oki model
        player_hp_weight = 10
        opp_hp_weight = 10

        # Initalize Oki model frame counter
        oki_count = 0

        # --------------------------- TESTING LOOP ----------------------------
        for episode in range(n_episodes):

            # Reset environment state at the start of each episode
            state = env.reset(p2=RLTEST)

            # Reset round counter
            round = 0

            # Reward tracking variable
            total_reward = 0

            # State tracking variables
            prev_opp_state = -1
            opp_state = -1
            oki = False
            action_count = 0

            # ------------------------- FIGHT LOOP ----------------------------
            while round < n_rounds:

                # Ensure state data is in the correct format
                if type(state) != np.ndarray:
                    state = state[0]

                # Get opponent AI's action
                opponent_action = opponent_agent.act(state, 0.0)
                env.getP2().setAction(opponent_action)

                # Determine if opponent is knocked down so we can switch to the oki model
                opp_state = env.getP1().getOpponentState()
                if type(opp_state) != str and type(prev_opp_state) != str and type(opp_state) != int and type(prev_opp_state) != int:
                    if opp_state.equals(env.getP1().gateway.jvm.enumerate.State.DOWN) and oki == False:
                        print('START OKI')
                        oki = True

                # Our agent actions:
                # Oki model
                if oki == True:

                    # Track oki state frames
                    oki_count += 1
                    action_count += 1

                    # Select action using the Oki model
                    action = oki_agent.act(state, epsilon)

                    # Log action selection
                    results_dict[opponent_file]['actions'].append(action)

                    # Update the environment using the selected action
                    next_state, _, done, _ = env.step(action)

                    # Calculate bespoke Oki model reward
                    reward = (opp_hp_weight * (state[65] - next_state[65])) - (player_hp_weight * (state[0] - next_state[0])) - (1/900)
                    total_reward += reward

                    # Cache this memory to the experience replay
                    oki_memory.push(state, action, next_state, reward, done, oki_agent)

                    # Learn from the model's experience replay
                    oki_agent.learn(oki_memory, batch_size, done)

                    # Check if Oki model timer is finished
                    if action_count == 90 or (state[0] - next_state[0] > 0):
                        print('END OKI')
                        oki = False
                        action_count = 0

                # Neutral model
                elif oki == False:
                    
                    # Select action using the Neutral model
                    action = neutral_agent.act(state, epsilon)

                    # Log action selection
                    results_dict[opponent_file]['actions'].append(action)

                    # Update the environment using the selected action
                    next_state, reward, done, _ = env.step(action)

                    # Calculate the bespoke Neutral model reward
                    opp_state = env.getP1().getOpponentState()
                    reward = calc_reward(env, state, action, next_state, prev_opp_state, opp_state, done)

                    # Cache this memory to the experience replay
                    neutral_memory.push(state, action, next_state, reward, done, neutral_agent)

                    # Learn from the model's experience replay
                    neutral_agent.learn(neutral_memory, batch_size, done)

                # Update state data for next frame
                prev_opp_state = opp_state
                state = next_state

                # Check for round end
                if done:

                    # Log damage done, damage taken, and number of frames spent in oki mode
                    player_hp = state[0] * 100
                    results_dict[opponent_file]['damage_taken'].append(100 - player_hp)
                    opponent_hp = state[65] * 100
                    results_dict[opponent_file]['damage_done'].append(100 - opponent_hp)
                    results_dict[opponent_file]['oki_frames'].append(oki_count)

                    # Reset oki frame count
                    oki_count = 0

                    # Log wins
                    if player_hp > opponent_hp:
                        results_dict[opponent_file]['wins'] += 1

                    # Increment round counter
                    round += 1

                    # Report the current win rate
                    print("Wins: " + str(results_dict[opponent_file]['wins']))
                    
                    # Reset state for next round
                    state = env.reset(p2=RLTEST)

                    # Force garbage collection at the end of each round
                    gc.collect()

            # Reload Oki and Neutral checkpoints at the end of each episode to 'forget' testing fights
            neutral_agent.load(neutral_file)
            oki_agent.load(oki_file)

            # Log total reward attained for episode
            rewards.append(total_reward)

        # Write testing results to file
        file_name = "results-" + opponent_file + ".json"
        with open(file_name, 'w') as outfile:
            json.dump(results_dict, outfile)

        # Force garbage collection at the end of the test
        gc.collect()

    # Re-enable garbage collector
    gc.enable()

    # Shut 'er down
    env.close()
    exit()

if __name__ == "__main__":
    main()