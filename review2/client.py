'''
Preamble: 
class Client(): The class that simulates an actual edge device. 
    self.ID: (int) Assigned uniquely. Used to distinguish between clients.
    self.CPU: (float) Simulates the CPU usage. Represents percentage of CPU.
    self.RAM: (float) Simulates the RAM usage. Represents percentage of CPU.
    self.TOTALRAM: (int) Total RAM 
    self.CHARGE: (int) The battery level of the client
    self.IDLE: (bool) to check if the device is idle
    self.PLUGGED: (bool) to check is the device is connected to a power source

    local_training(model_parameters: global model parameters i.e to be trained,
                   epochs: number of epochs to train) -> updates to model parameters and accuracy/loss: Trains the received model by applying SGD/Adam and sends the updates
    
    inference(model_parameters: global model parameters in case it isnt part of training) -> output of the model: Run the model in inference mode

    compatibility_checker(): returns a boolean based on the resource parameters of the client

    (FINAL) comm_interface(): Uses UDP socket to connect to the server. Sends the local model updates, parameters of the client. Receives the global model parameters.
'''
from inspect import FullArgSpec

import MNIST

class Client():
    def __init__(self, client_id: int, cpu_usage: int, ram_usage: int, charge: int, idle: bool, plugged: bool, local_dataset):
        self.ID = client_id
        self.CLIENT_RESOURCES =  {"CPU": cpu_usage, "RAM": ram_usage, "CHARGE": charge, "IDLE": idle, "PLUGGED": plugged}
        self.TRAINING = False
        self.NET = MNIST.SimpleNN()
        self.LOCAL_DATASET_SIZE = len(local_dataset)
        self.LOCAL_DATASET = local_dataset

    def local_training(self, pso_epochs):
        print(f"[*] Training on CLIENT{self.ID}")
        optim = MNIST.ParticleSwarmOptimizer(self.NET, client_data=self.LOCAL_DATASET)
        for i in range(pso_epochs):
            print(f"\t[*] Epoch {i + 1}")
            optim.step()
        # After training, update the client's model with the best solution found by PSO
        self.NET.unflatten_weights(optim.gbest_pos)
        self.TRAINING = False
        # Return the updated parameters and the fitness
        final_fitness = optim.calculate_fitness(optim.gbest_pos)
        return self.NET.get_params(), final_fitness

    def inference(self, x):
        return self.NET.forward(x)

    # Function to check the compatibility of the client with the server process
    def compatibility_check(self) -> bool:
        if (self.CLIENT_RESOURCES["IDLE"] and self.CLIENT_RESOURCES["CHARGE"] >= 30) or (self.CLIENT_RESOURCES["PLUGGED"]) or (not self.CLIENT_RESOURCES["IDLE"] and self.CLIENT_RESOURCES["CHARGE"] >= 50):
            if self.CLIENT_RESOURCES["CPU"] <= 80 and self.CLIENT_RESOURCES["RAM"] <= 80:
                return True
            else:
                return False
        else:
            return False

    def recv_parameters(self, params_dict):
        self.NET.set_params(params_dict)
