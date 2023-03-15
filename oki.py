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

import torch 

from classifier import Classifier
from DNN import DNN

def save_training_data(trainX, trainY):
    # Save the training data collected
    ID = str(uuid4())
    #print("Round ended. Saving data with ID: ", ID)

    # Set up the arrays to be saved
    final_train_X = np.array(trainX, dtype=np.float64)

    final_train_Y = np.array(trainY, dtype=np.int32)

    #print("Skill: ", final_train_Y[0])
    #print("Train X Shape: ", final_train_X.shape)
    #print("Train Y Shape: ", final_train_Y.shape)

    #print(final_train_Y[1])

    X_file = "trainX-" + ID
    Y_file = "trainY-" + ID

    # Save the files
    #cwd = os.getcwd()
    if not os.path.exists('training_data'):
        os.makedirs('training_data')
    #os.chdir(cwd + "\\training_data")

    # Save training data
    np.save(os.path.join('training_data', X_file), final_train_X, allow_pickle=True)
    np.save(os.path.join('training_data', Y_file), final_train_Y, allow_pickle=True)

    #os.chdir(cwd)

def training_episodes(model, env, action_strs, action_vecs, epsilon):
    NUM_ROUNDS = 3
    NUM_STEPS = 500
    round = 1

    DRIVER_ONEHOTS = [[1,0,0], [0,1,0], [0,0,1]]

    # First one casues issues for some reason
    obs = env.reset()
    obs = obs[0]

    while True:

        # Initialize round
        obs = env.reset()
        done = False

        # Training data
        trainX = []
        rewards = []

        steps = 0

        driver = randint(0, 2)

        while not done:
            if type(obs) != np.ndarray:
                print("Oopsie")
                action = 0
                obs, reward, done, _ = env.step(action)

            steps += 1

            # Get next action
            action = 0
            if random.random() < epsilon: # Explore
                action = random.randint(0, 55)
            else:                         # Exploit
                action = 0
                best_val = float("-inf")
                for a in range(len(action_strs)):
                    act_vector = action_vecs[action_strs[a]]
                    t_obs = torch.from_numpy(obs).float()
                    t_act = torch.from_numpy(act_vector).float()
                    input = torch.cat((t_obs, t_act))
                    pred = model.forward(input)

                    if pred > best_val:
                        action = a
                        best_val = pred

                #print("Best action = ", action)

            # Collect training data
            trainX.append(np.concatenate((obs, action_vecs[action_strs[action]])))
            rewards.append(DRIVER_ONEHOTS[driver])

            # Do action
            new_obs, reward, done, _ = env.step(action)

            # Update observation
            obs = new_obs

            if done:
                save_training_data(trainX, rewards)
                round += 1

        if round > NUM_ROUNDS:
            break

    return env

def classifier_training_data(clear=False):
    # Extract the training data files in the train_data folder.
    # If clear is set to true, delete the data files after
    # extracting info (helpful for running a LOT of games).
    
    trainX_files = []
    trainY_files = []

    # Save initial working directory
    #py_directory = os.getcwd()

    #os.chdir(py_directory + "\\training_data")
    if not os.path.exists('training_data'):
        os.makesirs('training_data')
    files = os.listdir('training_data')

    for f in files:
        if f.startswith("trainX"):
            trainX_files.append(f)
        elif f.startswith("trainY"):
            trainY_files.append(f)

    trainX_files.sort()
    trainY_files.sort()

    x_init = True
    y_init = True
    trainX = None
    trainY = None

    for xfile in trainX_files:
        if x_init:
            trainX = np.load(os.path.join('training_data', xfile))
            x_init = False
        else:
            x = np.load(os.path.join('training_data', xfile))
            trainX = np.concatenate((trainX, x))

    #print(trainX.shape)
    #print(trainX[0][0])

    for yfile in trainY_files:
        if y_init:
            trainY = np.load(os.path.join('training_data', yfile))
            y_init = False
        else:
            y = np.load(os.path.join('training_data', yfile))
            trainY = np.concatenate((trainY, y))

    #print(trainY)
    #print(trainY.shape)

    trainX = np.split(trainX, np.size(trainX, axis=0), axis=0)

    for i in range(len(trainX)):
        trainX[i] = np.squeeze(trainX[i])

    #print(trainX[0].shape, "trainX shape")

    trainY = np.split(trainY, np.size(trainY, axis=0), axis=0)

    for i in range(len(trainY)):
        trainY[i] = np.squeeze(trainY[i])

    #print(len(trainY))
    #print(trainY[0].shape)

    # Cleanup the replays if requested
    if clear:
        for f in files:
            os.remove(os.path.join('training_data', f))

    # Revert back to initial working diretory
    #os.chdir(py_directory)

    return trainX, trainY

def generate_reward(classifier, skill, trainX, num_skills):
    trainY = []

    # Pass state observations to discriminator to classify
    trainX = torch.FloatTensor(trainX)
    pred = classifier.forward(trainX)

    for p in pred:
        trainY.append(np.log(p[skill].detach().numpy()) - 
                      np.log(1/num_skills))

    # Data shuffling
    trainY = np.array(trainY)
    trainY = np.expand_dims(trainY, 1)

    return trainY

def train_model(model, name, trainX, trainY):
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, 'models', name)):
        os.makedirs(os.path.join(cwd, 'models', name))
    #os.chdir(cwd + "\\models\\" + name)
    
    print(type(trainX)) # list

    #model.forward(trainX)

def train_DIAYN(name, num_skills, clear=False):

    # Extract training data
    trainX, class_trainY = classifier_training_data(True)

    # For each skill, we generate our trainY then train
    for skill in range(num_skills):

        # Load classifier
        classifier = Classifier("DIAYN\\Classifier " + str(num_skills), num_skills)

        # Generate reward
        trainY = generate_reward(classifier, skill, trainX, num_skills)

        # Load skill and train
        n = "DIAYN\\Skill " + str(skill + 1)
        model = DNN()
        train_model(model, n, trainX, trainY)

def main():
    print('start oki training')

    num_skills = 1

    name = 'DIAYN\\Skill'

    _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
    action_strs = _actions.split(" ")
    action_vecs = {}

    for i in range(len(action_strs)):
        v = np.zeros(len(action_strs), dtype=np.float32)
        v[i] = 1
        action_vecs[action_strs[i]] = v
    
    print("Action space length", len(action_vecs))

    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path="", port=4242, freq_restart_java=100000)
    obs = env.reset(p2 = KickAI)
    obs = torch.from_numpy(obs).float()
    actions = torch.from_numpy(action_vecs['AIR']).float()
    x = torch.cat((obs, actions))

    model = DNN()
    y = model(x) #Input: state, action space, Output: action
    print("y = ", y)
    EPSILON_START = 0.95
    EPSILON_END = 0.05
    epsilon = EPSILON_START

    n_episodes = 1

    for episode in range(1, n_episodes + 1):
        anneal = n_episodes - episode + 1
        epsilon = max((anneal / n_episodes) - 0.05, EPSILON_END)

        env = training_episodes(model, env, action_strs, action_vecs, epsilon)
        train_DIAYN(name, num_skills, clear=True)

    env.close()
    exit()

if __name__ == "__main__":
    main()