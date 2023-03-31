import torch
import copy
import matplotlib.pyplot as plt

def load(file):
        checkpoint = torch.load(file, map_location=torch.device("cpu"))

        #print(checkpoint['model']) # Load current DNN weights
        print(checkpoint['reward'])
        print(len(checkpoint['reward']))

        plt.plot(checkpoint['reward'])
        plt.show()


        print(checkpoint['loss'])

        plt.plot(checkpoint['loss'])
        plt.show()

        #self.optimizer.load_state_dict(checkpoint['optimizer']) # Update optimizer weights

        return checkpoint['epsilon']

file = "./counter.pt"

print("Epsilon: " + str(load(file)))