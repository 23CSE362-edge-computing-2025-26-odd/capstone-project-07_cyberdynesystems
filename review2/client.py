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

    threshold_calc(): Computes the *score based on the client parameters

    comm_interface(): Uses UDP socket to connect to the server. Sends the local model updates, parameters of the client. Receives the global model parameters.
'''


class Client():
    def init(self, cpu_usage: int, total_cores: int, ram_usage: int, total_ram: int, charge: int, idle: bool, plugged: bool):
        self.ID = None
        self.CPU = 0
	self.RAM = 0
	self.TOTALRAM = 0
	self.CHARGE = 0
	self.IDLE = 0
    
    def local_training(self, epochs):
        pass

    def inference(self, x):
	pass

    def threshold_calc(self):
        pass

    def recv_parameters(self, params_dict):
	pass
    
    def comm_interface(self, place_holder): # Try socket programming
        pass