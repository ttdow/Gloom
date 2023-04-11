import numpy as np
import random
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import gc

import torch
import gym

from Agent import Agent
from OkiAgent import OkiAgent

from scipy.interpolate import interp1d


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    neutral_file = "neutral.pt"
    oki_file = "oki_019.pt"

    neutral_checkpoint = torch.load(neutral_file, map_location = device)
    oki_checkpoint = torch.load(oki_file, map_location = device)

    oki_rewards = oki_checkpoint['rewards']
    oki_losses = oki_checkpoint['losses']

    print(len(oki_rewards))
    print(len(oki_losses))

    oki_n_episodes = len(oki_rewards)

    x = np.array(range(oki_n_episodes))
    #cubic_interpolation_model = interp1d(range(oki_n_episodes), oki_rewards)
    #X_ = np.linspace(x.min(), x.max(), 500)
    #Y_ = cubic_interpolation_model(X_)
    y = oki_rewards

    a, b = np.polyfit(x, y, 1)
    print(a)
    plt.plot(x, y) 
    plt.plot(x, a * x + b)
    plt.title("oki model rewards")
    plt.xlabel("Training Episodes")
    plt.ylabel("Rewards")
    plt.show()

    

if __name__ == "__main__":
    main()