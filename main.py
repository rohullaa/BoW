import torch 
import numpy as np

if __name__ == "__main__":
    X_train = torch.load("data/X_train.pt")
    Y_train = torch.load("data/X_test.pt")

    Y_train = np.loadtxt(fname = "data/Y_train.txt")
    Y_test = np.loadtxt(fname = "data/Y_test.txt")