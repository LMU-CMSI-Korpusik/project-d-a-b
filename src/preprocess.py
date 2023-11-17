import pandas as pd
import numpy as np
import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class Net(nn.Module):
    def __init__(self, num_input, num_out):
        super().__init__()
        # only a single input layer then a sigmoid output layer
        # can create attribute for each hidden layer
        self.input = nn.Linear(num_input, num_out)
        self.h_1 = nn.Linear(num_input, num_out)
        self.h_2 = nn.Linear(num_input, num_out)
        self.h_3 = nn.Linear(num_input, num_out)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        return  self.sigmoid(self.h_3(self.sigmoid(self.h_2(self.sigmoid(self.h_1(self.input(input)))))))

# import train data and prompts as pandas dataframes
train_essays = pd.read_csv("data/train_essays.csv")
prompts = pd.read_csv("data/train_prompts.csv")

# create numpy feature vectors

# convert numpy feature vectors to torch tensors
    

def main():
    return 
    

if __name__ == "__main__":
    main()