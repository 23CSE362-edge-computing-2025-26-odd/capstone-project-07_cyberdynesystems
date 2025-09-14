"""
Preamble:
class Client(): The class that simulates an actual edge device.
    self.ID: (int) Assigned uniquely. Used to distinguish between clients.
    self.CLIENT_RESOURCES: (dict)
        self.CLIENT_RESOURCES["CPU"]: Simulates the CPU usage of the client
        self.CLIENT_RESOURCES["RAM"]: Simulates the RAM usage of the client
        self.CLIENT_RESOURCES["CHARGE"]: Simulates the CHARGE of the client
        self.CLIENT_RESOURCES["IDLE"]: Simulates the IDLE state of the client
        self.CLIENT_RESOURCES["PLUGGED"]: Simulates the PLUGGED state of the client
    self.TRAINING: (bool) Used to check if the client is chosen for training or inferencing
    self.NET: (MNIST.SimpleNN()) Used to set up the neural network in the client-end
    self.LOCAL_DATASET_SIZE: (int) The size of the local dataset on the client
    self.LOCAL_DATASET: (np.array(uint8)) This is the local dataset stored in the client object

    local_training(epochs: number of epochs to train):
        updates to model parameters and accuracy/loss: Trains the received model by applying SGD/Adam and sends the updates

    inference(): forward pass the input and return the output

    compatibility_checker(): returns a boolean based on the resource parameters of the client

    recv_parameters(params_dict: weights and bias of the global model): Set the global model parameters to the local model

    (FINAL) comm_interface():
    Uses TCP socket to connect to the server. Sends the local model updates, parameters of the client. Receives the global model parameters.
"""
import MNIST

class Client:
    def __init__(self, client_id: int, cpu_usage: int, ram_usage: int, charge: int, idle: bool, plugged: bool, local_dataset):
        self.ID = client_id
        self.CLIENT_RESOURCES =  {"CPU": cpu_usage, "RAM": ram_usage, "CHARGE": charge, "IDLE": idle, "PLUGGED": plugged}
        self.TRAINING = False
        self.NET = MNIST.SimpleNN()
        self.LOCAL_DATASET_SIZE = len(local_dataset)
        self.LOCAL_DATASET = local_dataset

    # Function to check the compatibility of the client with the server process
    def compatibility_check(self) -> bool:
        pass

    def local_training(self, pso_epochs):
        pass

    def inference(self, x):
        pass

    def recv_parameters(self, params_dict):
	pass
