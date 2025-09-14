'''
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
    self.NET: Used to set up the neural network in the client-end
    self.LOCAL_DATASET_SIZE: (int) The size of the local dataset on the client
    
    local_training(epochs: number of epochs to train):
        updates to model parameters and accuracy/loss: Trains the received model by applying SGD/Adam and sends the updates

    inference(): forward pass the input and return the output

    threshold_calc(): Computes the *score based on the client parameters

    recv_parameters(params_dict: weights and bias of the global model): Set the global model parameters to the local model

    (FINAL) comm_interface():
    Uses TCP socket to connect to the server. Sends the local model updates, parameters of the client. Receives the global model parameters.

'''

class Client():
    def _init_(self, cpu_usage: int, total_cores: int, ram_usage: int, total_ram: int, charge: int, idle: bool, plugged: bool):
        self.ID = None
        self.CLIENT_RESOURCES =  {"CORESUSAGE": 0, "TOTALCORES": 0, "RAM": 0, "TOTALRAM": 0, "CHARGE": 0, "IDLE": None, "PLUGGED": None}
        self.TRAINING = None
        self.NET = None
        self.LOCAL_DATASET_SIZE = 0
    
    def local_training(self, epochs):
	pass

    def inference(self, x):
	pass

    def threshold_calc(self):
        pass

    def recv_parameters(self, params_dict):
	pass
