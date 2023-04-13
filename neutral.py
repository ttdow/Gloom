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

class Genotype():
    def __init__(self, batch_size, update_freq, lr, gamma, tau, alpha, n_layers):
        self.batch_size = batch_size    # Experience replay batch size per round
        self.update_freq = update_freq  # Target network soft update frequency
        self.lr = 0.001               # Optimizer learning rate
        self.gamma = gamma              # Discount rate
        self.tau = tau                  # Target network update rate
        self.alpha = alpha              # Priority decay rate
        self.n_layers = n_layers        # Hidden layers

    def __lt__(self, other):
        return self.gamma < other.gamma

    def __str__(self):

        out = ""
        out += "Batch Size: " + str(self.batch_size) + "\n"
        out += "Update Freq: " + str(self.update_freq) + "\n"
        out += "Gamma: " + str(self.gamma) + "\n"
        out += "Tau: " + str(self.tau) + "\n"
        out += "Alpha: " + str(self.alpha) + "\n"
        out += "Hidden Layers: " + str(self.n_layers) + "\n"

        return out
    
class EHO():
    def __init__(self, batch_size, update_freq, lr, gamma, tau, alpha, n_layers):
        
        # Create progenitor using the user defined values
        self.progenitor = Genotype(batch_size, update_freq, lr, gamma, tau, alpha, n_layers)
        self.genotypes = []
        self.genotypes.append(self.progenitor)
        
        # Used to track which genotype is currently being validated
        self.index = 0

        # Used to track the validation results
        self.phenotypes = [None] * 10

        # Create a full set of 10 genotypes via mutation
        self.populate()

    def update_phenotype(self, reward, win_rate):

        # Update the currently selected phenotype and move to next
        self.phenotypes[self.index] = (reward, win_rate)
        self.index += 1

        # Circular list
        if self.index > 9:
            self.index = 0

    def populate(self):

        # Create 9 more genotypes by mutating the progenitor
        for i in range(0, 9):
            self.genotypes.append(self.mutate(self.progenitor))

    def mutate(self, genotype):

        # Make a deep copy of the provided genotype
        genotype = copy.deepcopy(genotype)

        # Mutate batch size
        m_batch_size = random.randint(-5, 5)
        if genotype.batch_size + m_batch_size < 1:
            genotype.batch_size = 1
        else:
            genotype.batch_size += m_batch_size

        # Mutate target DNN soft update frequency
        m_update_freq = random.randint(-2, 2)
        if genotype.update_freq + m_update_freq < 1:
            genotype.update_freq = 1
        else:
            genotype.update_freq += m_update_freq

        # Mutate Q-learning gamma (discount rate)
        m_gamma = random.uniform(-0.25, 0.25)
        if genotype.gamma + m_gamma < 0.01:
            genotype.gamma = 0.01
        elif genotype.gamma + m_gamma > 1.0:
            genotype.gamma = 1.0
        else:
            genotype.gamma += m_gamma

        # Mutate target DNN update rate
        m_tau = random.uniform(-0.05, 0.05)
        if genotype.tau + m_tau < 0.00001:
            genotype.tau = 0.00001
        else:
            genotype.tau += m_tau

        # Mutate priority decay rate
        m_alpha = random.uniform(-0.1, 0.1)
        if genotype.alpha + m_alpha < 0.01:
            genotype.alpha = 0.01
        else:
            genotype.alpha += m_alpha

        # Mutate number of hidden layers in DNN
        m_n_layers = random.randint(-1, 1)
        if genotype.n_layers + m_n_layers < 1:
            genotype.n_layers = 1
        else:
            genotype.n_layers += m_n_layers

        # Return the mutated copy of the original genotype
        return genotype
    
    def crossover(self, g0, g1):

        child_genotype = Genotype(0, 0, 0, 0, 0, 0, 0)
        coin_tosses = []
        for i in range(7):
            coin_tosses.append(random.randint(0, 1))

        if coin_tosses[0] == 0:
            child_genotype.batch_size = g0.batch_size
        else:
            child_genotype.batch_size = g1.batch_size

        if coin_tosses[1] == 0:
            child_genotype.update_freq = g0.update_freq
        else:
            child_genotype.update_freq = g1.update_freq

        if coin_tosses[3] == 0:
            child_genotype.gamma = g0.gamma
        else:
            child_genotype.gamma = g1.gamma

        if coin_tosses[4] == 0:
            child_genotype.tau = g0.tau
        else:
            child_genotype.tau = g1.tau

        if coin_tosses[5] == 0:
            child_genotype.alpha = g0.alpha
        else:
            child_genotype.alpha = g1.alpha

        if coin_tosses[6] == 0:
            child_genotype.n_layers = g0.n_layers
        else:
            child_genotype.n_layers = g1.n_layers

        return child_genotype
    
    def selection(self):

        # Determine the fitness of each phenotype in the current generation
        fitness = []
        for phenotype in self.phenotypes:
            fitness.append(phenotype[1])

        # Sort the genotypes by fitness values
        genotypes = self.genotypes
        sorted_genotypes = [x for _, x in sorted(zip(fitness, genotypes), reverse=True)]

        # Output results of last generation before making a new one
        print("Top 3 Selections: ")
        print(str(sorted_genotypes[0]))
        print("Fitness value: " + str(fitness[0]))
        print("")
        print(str(sorted_genotypes[1]))
        print("Fitness value: " + str(fitness[1]))
        print("")
        print(str(sorted_genotypes[2]))
        print("Fitness value: " + str(fitness[2]))
        print("------------------------------")

        next_generation = []

        # Create 6 new genotypes by crossing-over the top 4 in every combination
        next_generation.append(self.crossover(sorted_genotypes[0], sorted_genotypes[1]))
        next_generation.append(self.crossover(sorted_genotypes[0], sorted_genotypes[2]))
        next_generation.append(self.crossover(sorted_genotypes[0], sorted_genotypes[3]))
        next_generation.append(self.crossover(sorted_genotypes[1], sorted_genotypes[2]))
        next_generation.append(self.crossover(sorted_genotypes[1], sorted_genotypes[2]))
        next_generation.append(self.crossover(sorted_genotypes[2], sorted_genotypes[3]))

        # Create 2 new genotypes by mutating the top 2
        next_generation.append(self.mutate(sorted_genotypes[0]))
        next_generation.append(self.mutate(sorted_genotypes[1]))

        # Carry over the top 2
        next_generation.append(sorted_genotypes[0])
        next_generation.append(sorted_genotypes[1])

        # Out with the old, in with new
        self.genotypes = next_generation

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priority = np.empty(self.capacity)
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

        # Calculate importance/priority of this memory (i.e. td error)
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

    # ---------------------- Incentive for downs ------------------------------
    if type(opp_state) != str and opp_state != None:
        if opp_state.equals(env.getP2().gateway.jvm.enumerate.State.DOWN) and opp_state != prev_opp_state:
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
    opponent = "Thunder2021"
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="", port=4242, freq_restart_java=100000)
    state = env.reset(p2=Machete)

    # Setup epsilon values for explore/exploit calcs
    EPSILON_MAX = 1.0
    EPSILON_DECAY = 0.9772372210
    EPSILON_MIN = 0.1
    epsilon = EPSILON_MAX

    # Training parameters
    n_episodes = 150       # Number of training episodes
    n_rounds = 3           # Round per episode

    # Validation parameters
    n_valid_episodes = 20

    # Hyperparameters
    batch_size = 16                # Experience replay batch size per round
    targetDNN_soft_update_freq = 2 # Target network soft update frequency
    learning_rate = 0.0000625      # Optimizer learning rate
    gamma = 0.99                   # Discount rate
    tau = 0.01                     # Target network rate
    alpha = 0.6                    # Priority decay
    n_layers = 1                   # Hidden layers

    # Setup evolutionary hyperparameter optimizer
    eho = EHO(batch_size, targetDNN_soft_update_freq, learning_rate, gamma, tau, alpha, n_layers)
    n_generations = 3

    # Load model if it exists
    #if file != "":
        #_, rewards = agent.load(file)
        #print("Model: " + file + " loaded.")

    # Flag for round finished initialy false
    done = False

    # Initialize timing data
    frame_counter = 0
    old_time = time.time()

    # Generational loop - evolutionary hyperparameter optimization
    for generation in range(n_generations):

        # Train multiple hyperparameter genotypes in this generation
        for genotype in eho.genotypes:

            print(str(genotype))

            # Initialize agent and experience replay memory
            agent = Agent(state.shape[0], len(action_vecs), genotype.lr, genotype.gamma, genotype.tau, genotype.alpha, genotype.n_layers)
            memory = ReplayMemory(100000)

            # Initialize logs
            rewards = []
            damage_done = []
            damage_taken = []
            wins = 0

            # ------------------------- TRAINING LOOP -------------------------
            for episode in tqdm(range(n_episodes)):

                # Reset env for next episode
                state = env.reset(p2=Machete)
                round = 0
                total_reward = 0

                # Reset opponent's state for next episode
                prev_opp_state = -1

                # Round timing data
                old_time = time.time()

                #print("Episode: " + str(episode))

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
                    opp_state = env.getP2().state # TODO Can't be used with Java AI agent - need to find workaround

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
                        #print(str(frame_counter) + " frames / " + str(dt) + " (FPS: " + str(frame_counter / dt) + ")")
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

                #print("Epsilon: " + str(epsilon))
                #print("Total Reward: " + str(total_reward))

                # Decrease epsilon for next epsiode - cosine annealing
                epsilon = EPSILON_MIN + 0.5 * (EPSILON_MAX - EPSILON_MIN) * (1 + math.cos((episode / n_episodes) * math.pi))

                # Log total reward of episode for
                rewards.append(total_reward)

                # Save the model every 25 episodes
                #if episode % 25 == 0 and episode > 0:
                    #print("Saving checkpoint at episode " + str(episode))
                    #agent.save('./test3.pt', epsilon, rewards, wins, damage_done, damage_taken)

                #print("------------------------------")

                # Force garbage collection between episodes
                gc.collect()

            # Track wins for validation
            wins = 0

            # --------------------- VALIDATION LOOP -----------------------
            for episode in tqdm(range(n_valid_episodes)):

                # Reset env for next episode
                state = env.reset(p2=opponent)
                round = 0

                # Reset opponent's state for next episode
                prev_opp_state = -1

                # Loop until n_rounds are complete
                while round < n_rounds:

                    # Ensure the environment state is in the correct format
                    if type(state) != np.ndarray:
                        state = state[0]
            
                    # Get the next action - set epsilon to 0 for validation
                    action = agent.act(state, 0.0)

                    # Step the environment with the selected action
                    next_state, reward, done, _ = env.step(action)

                    # Update the state for next frame
                    state = next_state

                    # Check if round is complete
                    if done:

                        # Log winner
                        if playerHP > opponentHP:
                            wins += 1

                        # Setup for the next round
                        round += 1
                        state = env.reset(p2=opponent)

            # Save fitness value (reward, win_rate) for this genotype
            eho.update_phenotype(None, wins / (n_valid_episodes * n_rounds))
            print("Win rate: " + str(wins / (n_valid_episodes * n_rounds)))
            print("------------------------------")
            wins = 0

            # Force garbage collection between genotypes
            gc.collect()

        # Create a new generation using the outcomes of the previous generation
        eho.selection()

    # Save final checkpoint
    #print("Saving checkpoint at episode " + str(episode))
    #agent.save('./test3.pt', epsilon, rewards, wins, damage_done, damage_taken)

    # Re-enable garbage collection
    gc.enable()

    # Terminate
    env.close()
    exit()

if __name__ == "__main__":
    main()