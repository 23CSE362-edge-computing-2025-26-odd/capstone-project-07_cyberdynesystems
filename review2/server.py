"""
Preamble:
class Server(): The class that implements the server side of the workflow
    self.CLIENTS: ([client]) Stores the clients connected to the server

    choose_c(): calls the compatibility check function of the client and appends the client objects to the C list

    send_parameters(): sends the global parameters to the chosen C clients for training and for inferencing to K - C clients

    fedAvg(updated_parameters): updates the self.GLOBAL_PARAMETERS with the aggregate
"""
import MNIST
import client
import random
import numpy as np

class Server:
    def _init_(self, input_size=784, hidden_size=64, output_size=10):
        self.GLOBAL_PARAMETERS = {self.GLOBAL_PARAMETERS = {
            "w1": np.random.randn(hidden_size, input_size) * 0.1,
            "b1": np.zeros((hidden_size, 1)),
            "w2": np.random.randn(output_size, hidden_size) * 0.1,
            "b2": np.zeros((output_size, 1))
        } # Parameter that stores the weights and bias of the current model
        self.CLIENTS = [] # Store client objects (K)
        self.CHOSEN = [] # Store the chosen clients (C)
        self.UPDATES = {} # Buffer to store the local updates
        self.TOTAL_DATASET_SIZE = 0

    '''
    choose_c: Calls the compatibility check function of the client and appends the client objects to the Chosen list
    '''
    def choose_c(self): # Chooses clients based where they are fit for training the model
        self.CHOSEN = []
        for i in self.CLIENTS:
            if i.compatibility_check():
                self.TOTAL_DATASET_SIZE += i.LOCAL_DATASET_SIZE
                self.CHOSEN.append(i)
    
    '''
    send_parameters: 
    This method has to send the global parameters to the server during every communication round
    '''
    def send_parameters(self):
        CHOSEN_SUBSET = random.sample(self.CHOSEN, 3)
        for i in self.CLIENTS: # Choosing a subset of the chosen clients to train
            i.TRAINING = True if i in CHOSEN_SUBSET else False
            i.recv_parameters(self.GLOBAL_PARAMETERS) # Sends the global parameters to the client

    '''
    fedAVG:
    Compute the weight nk / n where nk is the size of the dataset in the client and n is the total size of the dataset
    across the chosen clients. Multiply it with the local update of the client. Sum all of it and set it as the global parameter.
    '''
    def fedAvg(self):
        pass
