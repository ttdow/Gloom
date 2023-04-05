import numpy as np
import time
import random
import sys
import gc
from tqdm import tqdm

import torch
import gym

from Agent import Agent
from gym_fightingice.envs.Machete import Machete

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priority = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, state, action, next_state, reward, done, agent):
        
        # Make more room in memory if needed
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priority.append(None)

        # Not sure why this happens        
        if type(state) != np.ndarray:
                state = state[0]

        # Convert data from ndarray to tensor for ease of use
        state = torch.from_numpy(state).float().to(agent.device)
        next_state = torch.from_numpy(next_state).float().to(agent.device)

        # Learn from last state, action transition
        priority = agent.prioritize(state, action, next_state, reward, done)
        self.priority[self.position] = priority

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
    
def GetDistance(env_state):

    # Get normalized player positions in stage
    playerX = env_state[2]
    opponentX = env_state[67]

    # Calculate distance between players in pixels
    dist = abs(playerX - opponentX) * 960

    return int(dist)
    
def calc_reward(env, env_state, action, next_env_state, prev_opp_state, opp_state, done):

    reward = -1

    # Ensure the environment state is in the correct format
    if type(env_state) != np.ndarray:
        env_state = env_state[0]

    # ------------------- Huge incentive for downs ----------------------------
    if type(opp_state) != str:
        if opp_state.equals(env.getP2().gateway.jvm.enumerate.State.DOWN) and opp_state != prev_opp_state:
            reward += 1000

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

    # TODO Customize reward based on spacing state
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
    EPSILON_DECAY = 0.9847666521
    EPSILON_MIN = 0.01
    epsilon = EPSILON_MAX

    # Initialize agent and experience replay memory
    agent = Agent(state.shape[0], len(action_vecs))
    memory = ReplayMemory(100000)

    # Initialize logs
    rewards = []
    damage_done = []
    damage_taken = []
    wins = 0

    # Load model if it exists
    if file != "":
        epsilon, rewards = agent.load(file)
        print("Model: " + file + " loaded.")

    # Hyperparameters
    batch_size = 32                # Experience replay batch size per round
    n_episodes = 150               # Number of training episodes
    n_rounds = 3                   # Round per episode
    targetDNN_soft_update_freq = 1 # Target network soft update frequency

    # Flag for round finished
    done = False

    # Initialize timing data
    frame_counter = 0
    old_time = time.time()

    # Training loop - loop until n_episodes are complete
    for episode in tqdm(range(n_episodes+1)):

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

            # Get opponent's current state from env (STAND, CROUCH, AIR, DOWN)
            opp_state = env.getP2().state

            # Calculate reward function based on states and action
            reward = calc_reward(env, state, action, next_state, prev_opp_state, opp_state, done)

            # Update opponent's last state
            prev_opp_state = opp_state

            # Save total reward for the episode for logging
            total_reward += reward

            # Add the last state, action transition to the agent's memory cache
            memory.push(state, action, next_state, reward, done, agent)

            # Update Q-values
            agent.learn(memory, batch_size)

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

                # Update epsilon for next round
                epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

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

        # Log total reward of episode for
        rewards.append(total_reward)

        # Save the model every 50 episodes
        if episode > 0 and episode % 50 == 0:
            print("Saving checkpoint at episode " + str(episode))
            agent.save('./test.pt', epsilon, rewards, wins, damage_done, damage_taken)

        print("------------------------------")

        # Force garbage collection between episodes
        gc.collect()

    # Re-enable garbage collection
    gc.enable()

if __name__ == "__main__":
    main()