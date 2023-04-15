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
from ReplayMemory import ReplayMemory

def calc_reward(env, env_state, action, next_env_state, prev_opp_state, opp_state, done):

    reward = -1

    # Ensure the environment state is in the correct format
    if type(env_state) != np.ndarray:
        env_state = env_state[0]

    # ---------------------- Incentive for downs ------------------------------
    if type(opp_state) != str and opp_state != None:
        if str(opp_state) == "DOWN" and str(prev_opp_state):
            reward += 100

    player_old_HP = env_state[0]
    player_new_HP = next_env_state[0]

    opponent_old_HP = env_state[65]
    opponent_new_HP = next_env_state[65]

    # ------------------ Incentivize dealing damage to opponent ---------------
    damage_done = (opponent_old_HP - opponent_new_HP) * 100
    if damage_done > 0:
        reward += damage_done    # Reward proportional to damage done

    # ------------------ Incentivize taking less damage -----------------------
    damage_taken = (player_old_HP - player_new_HP) * 100
    if damage_taken > 0:
        reward -= damage_taken  # Penalize proportional to damage taken

    # ------------------ Big incentive for winning a round --------------------
    if done:
        if player_new_HP > opponent_new_HP:
            reward += 500

    # TODO Customize reward based on spacing state?
    # -------------------- Determine spacing state ----------------------------
    #dist = GetDistance(env_state)
    #if dist <= 135:
        # Close range / Shimmy
        # At close range, all attack options (pokes, normals, and specials) can connect
        # Generally want to be on offensive, but once your turn is over you either try to extend
        #  your turn or return to neutral
        #pass

    #elif dist <= 250:
        # Mid range / Footsie range
        # Just beyond the reach of your opponent's pokes and normals, but within jump-in range
        # Purposefully move in and out of your opponent's attack range to bait
        #pass

    #elif dist <= 500:
        # Far range
        # Only have to worry about projectiles
        #pass

    # Get player and opponent x-coords
    playerX = env_state[2]
    opponentX = env_state[67]
    
    # Full screen
    # Be wary of corner
    if playerX < 100:
        if opponentX < playerX:
            reward += 1    # Bonus for pinning opponent in corner
        else:
            reward -= 1    # Malus for being pinned
    elif playerX > 860:
        if opponentX > playerX:
            reward += 1    # Bonus for pinning opponent in corner
        else:
            reward -= 1    # Malus for being pinned

    return reward

def main():

    # Check for checkpoint to load - CLI syntax: py neutral.py <filepath>
    # Model saves automatically at the end of n_episodes (hyperparameter below)
    # Can change file output name at the bottom of this function
    file = ""
    if (len(sys.argv) > 1):
        file = str(sys.argv[1])

    # Disable the garbage collector for more consistent frame times
    gc.collect()
    gc.disable()

    # Setup action space
    _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
    action_strs = _actions.split(" ")
    action_vecs = []

    # One-hot encoding for actions
    for i in range(len(action_strs)):
        v = np.zeros(len(action_strs), dtype=np.float32)
        v[i] = 1
        action_vecs.append(v)

    # Setup observation space
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="", port=4242, freq_restart_java=100000)
    state = env.reset(p2=Machete)

    # Setup epsilon values for explore/exploit calcs
    EPSILON_MAX = 1.0
    EPSILON_MIN = 0.1
    epsilon = EPSILON_MAX

    # Training parameters
    n_episodes = 50       # Number of training episodes
    n_rounds = 3           # Round per episode

    # Hyperparameters
    batch_size = 16                # Experience replay batch size per round
    targetDNN_soft_update_freq = 2 # Target network soft update frequency
    learning_rate = 0.0000625      # Optimizer learning rate
    gamma = 0.99                   # Discount rate
    tau = 0.01                     # Target network rate
    alpha = 0.6                    # Priority decay
    n_layers = 1                   # Hidden layers

    # Load model if it exists
    #if file != "":
        #_, rewards = agent.load(file)
        #print("Model: " + file + " loaded.")

    # Flag for round finished initialy false
    done = False

    # Initialize timing data
    frame_counter = 0
    old_time = time.time()

    # Initialize agent and experience replay memory
    agent = Agent(state.shape[0], len(action_vecs), learning_rate, gamma, tau, alpha, n_layers, n_episodes)
    memory = ReplayMemory(100000)

    # Initialize logs
    rewards = []
    damage_done = []
    damage_taken = []
    wins = 0

    # ------------------------- TRAINING LOOP -------------------------
    for episode in range(n_episodes):

        # Reset env for next episode
        state = env.reset(p2=Machete)
        round = 0
        total_reward = 0

        # Reset opponent's state for next episode
        prev_opp_state = -1

        # Round timing data
        old_time = time.time()

        print("Episode: " + str(episode))

        # Loop until n_rounds are complete
        while round < n_rounds:

            # Track frame rate
            frame_counter += 1

            # Ensure the environment state is in the correct format
            if type(state) != np.ndarray:
                state = state[0]
    
            # Get the next action
            action = agent.act(state, epsilon)

            # Step the environment with the selected action
            next_state, reward, done, _ = env.step(action)

            # Get opponent state from env (STAND, AIR, CROUCH, DOWN)
            opp_state = env.getP1().getOpponentState()

            # Calculate reward function based on states and action
            reward = calc_reward(env, state, action, next_state, prev_opp_state, opp_state, done)

            # Update opponent's last state
            prev_opp_state = opp_state

            # Save total reward for the episode for logging
            total_reward += reward

            # Add the last state, action transition to the agent's memory cache
            memory.push(state, action, next_state, reward, done, agent)

            # Update Q-values
            agent.learn(memory, batch_size, done)

            # Update the state for next frame
            state = next_state

            # Check if round is complete
            if done:
                
                # Calculate average frame rate of round
                new_time = time.time()
                dt = new_time - old_time
                print(str(frame_counter) + " frames / " + str(dt) + " (FPS: " + str(frame_counter / dt) + ")")
                old_time = new_time
                frame_counter = 0

                # Log play and opponent health
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

        # Only update target network at the end of an episode
        if episode > 0 and episode % targetDNN_soft_update_freq == 0:
            agent.soft_update_target_network()

        print("Epsilon: " + str(epsilon))
        print("Total Reward: " + str(total_reward))
        print("-------------------------------")

        # Decrease epsilon for next epsiode - cosine annealing
        epsilon = EPSILON_MIN + 0.5 * (EPSILON_MAX - EPSILON_MIN) * (1 + math.cos((episode / n_episodes) * math.pi))

        # Log total reward of episode for
        rewards.append(total_reward)

        #Save the model checkpoint
        agent.save('./test.pt', epsilon, rewards, wins, damage_done, damage_taken)

        # Force garbage collection between episodes
        gc.collect()

    # Re-enable garbage collection
    gc.enable()

    # Terminate
    env.close()
    exit()

if __name__ == "__main__":
    main()