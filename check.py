import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

def load(file):
<<<<<<< HEAD

=======
        
>>>>>>> PriorityReplay
        checkpoint = torch.load(file, map_location=torch.device("cpu"))

        rewards = checkpoint['rewards']
        losses = checkpoint['losses']
<<<<<<< HEAD
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

        return

file = "./test.pt"
=======

        # Create best fit line for rewards
        x = np.array(range(0, len(rewards)))
        y = np.array(rewards)
        a, b = np.polyfit(x, y, 1)
        print("y = " + str(a) + "x + " + str(b))

        # Plot rewards
        plt.plot(rewards)
        plt.title("Reward Per Episode")
        plt.show()

        # Create best fit line for losses
        x = np.array(range(0, len(losses)))
        y = np.array(losses)
        a, b = np.polyfit(x, y, 1)
        print("y = " + str(a) + "x + " + str(b))

        # Plot losses
        plt.plot(checkpoint['losses'])
        plt.title("Loss Per Round")
        plt.show()

        # Print epsilon
        print("Final epsilon during training: " + str(checkpoint['epsilon']))

        return

# Get file name from argument
file = ""
if (len(sys.argv) > 1):
        file = str(sys.argv[1])
else:
        print("Please enter a file name.")
        exit()
>>>>>>> PriorityReplay

load(file)