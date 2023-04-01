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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    neutral_file = "neutral.pt"
    oki_file = "oki.pt"

    neutral_checkpoint = torch.load(neutral_file, map_location = device)
    oki_checkpoint = torch.load(oki_file, map_location = device)

    oki_rewards = oki_checkpoint['rewards']
    oki_losses = oki_checkpoint['losses']

    print(len(oki_rewards))
    print(len(oki_losses))

    #oki_n_episodes = len(oki_rewards)
    
    #plt.plot(range(oki_n_episodes), oki_rewards) 
    #plt.title("oki model rewards")
    #plt.xlabel("Training Episodes")
    #plt.ylabel("Rewards")
    #plt.show()

    

if __name__ == "__main__":
    main()