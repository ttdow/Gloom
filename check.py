import torch
import copy
import matplotlib.pyplot as plt
import numpy as np

def load(file):
        checkpoint = torch.load(file, map_location=torch.device("cpu"))

        #print(checkpoint['model']) # Load current DNN weights
        #print(checkpoint['rewards'])
        #print(len(checkpoint['rewards']))

        a, b = np.polyfit(range(0, len(checkpoint['rewards'])), checkpoint['rewards'], 1)
        print("y = " + str(a) + " * x + " + str(b))
        plt.plot(checkpoint['rewards'])
        plt.show()

        #print(checkpoint['losses'])

        a, b = np.polyfit(range(0, len(checkpoint['losses'])), checkpoint['losses'], 1)
        print("y = " + str(a) + " * x + " + str(b))
        plt.plot(checkpoint['losses'])
        plt.show()

        #self.optimizer.load_state_dict(checkpoint['optimizer']) # Update optimizer weights

        return checkpoint['epsilon']

file = "./test.pt"

print("Epsilon: " + str(load(file)))