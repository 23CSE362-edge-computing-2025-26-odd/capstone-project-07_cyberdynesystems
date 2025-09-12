import client, server
import numpy as np
import MNIST

client_datasets, test_dataset = MNIST.load_and_prepare_data(10)

# Initializing clients
# Providing unique ID to every client connected to the server
server = server.Server()
for i in range(10):
    server.CLIENTS.append(client.Client(i,
                                        np.random.randint(20, 90),
                                        np.random.randint(20, 90),
                                        np.random.randint(25, 100),
                                        bool(np.random.rand() > 0.5),
                                        bool(np.random.rand() > 0.5),
                                        client_datasets[i]))
    print(f"[+] CLIENT{i} connected")


# Core logic
comm_round = int(input("[!] Enter number of communication rounds: "))
server.choose_c()  # Choose c clients from the k clients

for number in range(comm_round):
    print(f"[*] Communication round {number + 1}")
    print(f"[*] Training on {[i.ID for i in server.CHOSEN]}")
    server.send_parameters() # Send parameters to a subset of clients 
    epochs = int(input("[!] Enter number of epochs: "))
    # Run the training loop on every chosen client for epoch number of times
    print("[*] Training on clients in progress")
    for i in server.CHOSEN:
        if i.TRAINING: 
            server.UPDATES[i], fitness = i.local_training(epochs) # Store the locally trained parameters for fedAVG
            print(f"[*] Client{i.ID} Accuracy: {fitness}")
    server.fedAvg() # Run weighted average on the received local parameters and update the global parameters for the next round
    print()

# Aggregated model accuracy calculation
net = MNIST.SimpleNN()
net.set_params(server.GLOBAL_PARAMETERS)
accuracy = MNIST.evaluate_model(net, test_dataset)
print(f"[*] Accuracy: {accuracy}")
