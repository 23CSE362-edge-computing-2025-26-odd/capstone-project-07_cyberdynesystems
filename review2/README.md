# Edge Computing
## Simulation of the vanilla FL paper "Communication Efficient Learning of deep networks using decentralized data" 

Consists of 4 modules :
*  Client.py
*  Server.py
*  Simulation.py
*  MNIST.py (CI)

Model: Simple Neural Network 

Optimizer: Particle Swarm Optimizer, Genetic Algorithm Refiner, SGD 

Dataset: MNIST

The model used is a simple Neural Network and the dataset, MNIST. The simulation is performed on 10 clients with 6000 images each. 
The server controls the roles of the clients connected to it. It chooses from a pool of K clients a fractional C clients of which a
randomly chosen subset of 5 participate in the training. The updates from these clients are aggregated using FedAVG. The model is trained
on a batch size of 64 in the clients to reduce time, compute and improve accuracy.
