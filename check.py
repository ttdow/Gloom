import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

def load(file):
        
        checkpoint = torch.load(file, map_location=torch.device("cpu"))

        rewards = checkpoint['rewards']
        losses = checkpoint['losses']
        wins = checkpoint['wins']
        damage_done = checkpoint['damage_done']
        damage_taken = checkpoint['damage_taken']

        # Calc line of best fit for rewards and plot
        a, b = np.polyfit(range(0, len(rewards)), rewards, 1)
        print("y = " + str(a) + " * x + " + str(b))
        plt.plot(rewards)
        plt.show()

        # Calc line of best fit for losses and plot
        a, b = np.polyfit(range(0, len(losses)), losses, 1)
        print("y = " + str(a) + " * x + " + str(b))
        plt.plot(losses)
        plt.show()

        # Report win ratio
        print("Wins: " + str(wins) + " (" + str(wins/450) + ")")

        # Plot damage done
        plt.plot(damage_done)
        plt.show()

        # Plot damage taken
        plt.plot(damage_taken)
        plt.show()

        print(checkpoint['epsilon'])

        return

file = "./test2.pt"

load(file)