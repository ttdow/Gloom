import numpy as np
import time
import random
import sys
import gc
from tqdm import tqdm
import copy
import math

import torch
import gym
from gym_fightingice.envs.Machete import Machete

from Agent import Agent

import neutral
from ReplayMemory import ReplayMemory
    
def main():

    # Check for checkpoint to load - CLI syntax: py neutral.py <filepath>
    # Model saves automatically at the end of n_episodes (hyperparameter below)
    # Can change file output name at the bottom of this function
    neutral_file = "./neutral_best.pt"
    oki_file = "./oki_best.pt"
    if (len(sys.argv) > 1):
        file = str(sys.argv[1])

    # Disable the garbage collector for more consistent frame times
    gc.collect()
    gc.disable()

    # Setup action spac
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
    state = env.reset(p2=Machete)

    # Define validation opponent
    opponent = "Thunder2021"

    # Setup epsilon values for explore/exploit dilemma during training
    EPSILON_MAX = 1.0
    EPSILON_MIN = 0.0
    epsilon = EPSILON_MAX

    # Training parameters
    n_episodes = 1000 # Number of training episodes
    n_rounds = 3      # Rounds per episode

    # Validation parameters
    n_valid_episodes = 5
    validation_rate = 50
    best_win_rate = 0.0

    # Hyperparameters
    batch_size = 16                # Experience replay batch size per round
    targetDNN_soft_update_freq = 2 # Target network soft update frequency
    learning_rate = 0.0000625      # Optimizer learning rate - NOT USED CURRENTLY
    gamma = 0.99                   # Discount rate
    tau = 0.01                     # Target network rate
    alpha = 0.6                    # Priority decay
    n_layers = 1                   # Hidden layers

    # Initialize agents and experience replays
    neutral_memory = ReplayMemory(100000)
    neutral_agent = Agent(state.shape[0], len(action_vecs), learning_rate, gamma, tau, alpha, n_layers, n_episodes)

    oki_memory = ReplayMemory(100000)
    oki_agent = Agent(state.shape[0], len(action_vecs), learning_rate, gamma, tau, alpha, n_layers, n_episodes)

    # Initialize loggables
    rewards = []
    damage_done = []
    damage_taken = []
    wins = 0

    # Load checkpoint models if they exist
    if neutral_file != "":
        _, rewards, wins, damage_done, damage_taken = neutral_agent.load(neutral_file)
    if oki_file != "":
        _, _, _, _, _ = oki_agent.load(oki_file)

    # Flag for round finished
    done = False

    # Initialize timing data
    frame_counter = 0
    old_time = time.time()

    # Oki model reward weights
    player_hp_weight = 10
    opp_hp_weight = 10

    # ------------------------- TRAINING LOOP ---------------------------------
    for episode in range(551, n_episodes):

        print("Training Episode: " + str(episode))

        # Reset env for next episode
        state = env.reset(p2=Machete)
        round = 0
        total_reward = 0

        # Reset opponent's state for next episode
        prev_opp_state = -1

        # Initialize Oki state variables
        oki = False
        action_count = 0

        # Round timing data
        old_time = time.time()

        # -------------------------- ROUND LOOP -------------------------------
        while round < n_rounds:
            
            # Track frame rate
            frame_counter += 1

            # Get opponent state from env (STAND, AIR, CROUCH, DOWN)
            opp_state = env.getP1().getOpponentState()

            # If the opponent state is DOWN switch to Oki model
            if type(opp_state) != str and type(prev_opp_state) != str and type(opp_state) != int and type(prev_opp_state) != int:
                if opp_state.equals(env.getP2().gateway.jvm.enumerate.State.DOWN) and oki == False:
                    oki = True

            # Ensure the environment state is in the correct format
            if type(state) != np.ndarray:
                state = state[0]

            # Oki model
            if oki == True:
                action_count += 1

                # Get the next action from the Oki model
                action = oki_agent.act(state, epsilon)

                # Step environment
                next_state, _, done, _ = env.step(action)

                # Calculate reward
                if len(prev_state) == 143 and len(state) == 143:
                    reward = (opp_hp_weight * (state[65] - next_state[65])) - (player_hp_weight * (state[0] - next_state[0])) - (1/900)

                # Update experience replay
                oki_memory.push(state, action, next_state, reward, done, oki_agent)
                oki_agent.learn(oki_memory, batch_size, done)

                # Watch for end of Oki mode
                if action_count == 60:
                    oki = False
                    action_count = 0
            
            # Neutral model
            elif oki == False:

                # Get the next action from the neutral model
                action = neutral_agent.act(state, epsilon)

                # Step the environment with the selected action
                next_state, _, done, _ = env.step(action)

                # Calculate reward function based on states and action
                reward = neutral.calc_reward(env, state, action, next_state, prev_opp_state, opp_state, done)

                # Add the last state, action transition to the agent's memory cache
                neutral_memory.push(state, action, next_state, reward, done, neutral_agent)

                # Update Q-values in a batch
                neutral_agent.learn(neutral_memory, batch_size, done)

            # Save total reward for the episode for logging
            total_reward += reward

            # Update the state data
            prev_state = state
            state = next_state
            prev_opp_state = opp_state

            # Check if round is complete
            if done:
                
                # Calculate average frame rate of round
                new_time = time.time()
                dt = new_time - old_time
                #print(str(frame_counter) + " frames / " + str(dt) + " (FPS: " + str(frame_counter / dt) + ")")
                old_time = new_time
                frame_counter = 0

                # Log player and opponent health
                playerHP = state[0] * 100
                damage_taken.append(100 - playerHP)
                opponentHP = state[65] * 100
                damage_done.append(100 - opponentHP)

                # Log winner
                if playerHP > opponentHP:
                    wins += 1

                # Setup for the next round
                round += 1
                state = env.reset(p2=Machete)
        # ------------------------ END ROUND LOOP------------------------------

        # Only update target network at the end of an episode
        if episode > 0 and episode % targetDNN_soft_update_freq == 0:
            neutral_agent.soft_update_target_network()
            oki_agent.soft_update_target_network()

        epsilon = EPSILON_MIN + 0.5 * (EPSILON_MAX - EPSILON_MIN) * (1 + math.cos((episode / n_episodes) * math.pi))

        # Log total reward of episode
        rewards.append(total_reward)
        print("  Total Reward: " + str(total_reward))
        print("  Win Rate: " + str(wins / (episode * 3)))
        print("--------------------")

        # Save the models every episode
        neutral_agent.save('./neutral_training3.pt', epsilon, rewards, wins, damage_done, damage_taken)
        oki_agent.save('./oki_training3.pt', epsilon, rewards, wins, damage_done, damage_taken)

        # Force garbage collection between episodes
        gc.collect()

        # Validate training every 30 episodes
        if episode % validation_rate == 0 and episode > 0:

            valid_wins = 0
        
            # ----------------------- VALIDATION LOOP -----------------------------
            for episode in range(n_valid_episodes):

                print("Validation Episode: " + str(episode))

                # Reset env for next episode
                state = env.reset(p2=opponent)
                round = 0

                # Reset opponent's state for next episode
                prev_opp_state = -1

                # Initialize Oki state variables
                oki = False
                action_count = 0

                # -------------------------- ROUND LOOP -------------------------------
                while round < n_rounds:
                    
                    # Track frame rate
                    frame_counter += 1

                    # Get update state to determine if they are downed
                    opp_state = env.getP1().getOpponentState()
                    if type(opp_state) != str and type(prev_opp_state) != str and type(opp_state) != int and type(prev_opp_state) != int:
                        if opp_state.equals(env.getP2().gateway.jvm.enumerate.State.DOWN) and oki == False:
                            oki = True

                    # Ensure the environment state is in the correct format
                    if type(state) != np.ndarray:
                        state = state[0]

                    # Oki model
                    if oki == True:
                        action_count += 1

                        # Get the next action from the Oki model - no exploration in validation
                        action = oki_agent.act(state, 0.0)

                        # Step environment
                        next_state, _, done, _ = env.step(action)

                        # Watch for end of Oki mode
                        if action_count == 60:
                            oki = False
                            action_count = 0
                    
                    # Neutral model
                    elif oki == False:

                        # Get the next action from the neutral model
                        action = neutral_agent.act(state, 0.0)

                        # Step the environment with the selected action
                        next_state, reward, done, _ = env.step(action)

                    # Update the state data
                    prev_state = state
                    state = next_state
                    prev_opp_state = opp_state

                    # Check if round is complete
                    if done:

                        # Log winner
                        if playerHP > opponentHP:
                            valid_wins += 1

                        # Setup for the next round
                        round += 1
                        state = env.reset(p2=opponent)
                # -------------------- END ROUND LOOP--------------------------

                # Force garbage collection between episodes
                gc.collect()

            # Report validation results - win rate
            win_rate = valid_wins / (n_valid_episodes * n_rounds)
            print("  Win rate: " + str(win_rate))
            print("--------------------")

            # If this validation is the best so far, save it
            if win_rate >= best_win_rate:
                print("Saving new best checkpoints.")
                neutral_agent.save('./neutral_best.pt', epsilon, rewards, wins, damage_done, damage_taken)
                oki_agent.save('./oki_best.pt', epsilon, rewards, wins, damage_done, damage_taken)
                best_win_rate = win_rate

            # ---------------------- END VALIDATION LOOP ----------------------

    # ------------------------- END TRAINING LOOP -----------------------------

    # Re-enable garbage collection
    gc.enable()

    # Shut down the gym environment
    env.close()
    exit()

if __name__ == "__main__":
    main()