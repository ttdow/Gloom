import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

def load(file):
        checkpoint = torch.load(file, map_location=torch.device("cpu"))

        #print(checkpoint['model']) # Load current DNN weights
        print(checkpoint['rewards'])
        #print(len(checkpoint['rewards']))

        x = np.array(range(0, len(checkpoint['rewards'])))
        y = np.array(checkpoint['rewards'])

        a, b = np.polyfit(x, y, 1)
        print(a, b)

        #plt.scatter(x, y)
        plt.plot(y)
        plt.title("Reward Per Episode")
        plt.show()


        print(checkpoint['losses'])

        plt.plot(checkpoint['losses'])
        plt.title("Loss Per Round")
        plt.show()

        #self.optimizer.load_state_dict(checkpoint['optimizer']) # Update optimizer weights

        return checkpoint['epsilon']

file = "./neutral3.pt"

print("Epsilon: " + str(load(file)))