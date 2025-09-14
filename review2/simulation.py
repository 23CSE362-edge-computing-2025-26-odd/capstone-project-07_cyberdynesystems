import client, server

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
for number in range(comm_round):
    print("[*] Communication round {number}")
    server.choose_c() # Choose c clients from the k clients
    server.send_parameters() # Send parameters to a set of clients 
    epochs = int(input("[!] Enter number of epochs: "))
    # Run the training loop on every chosen client for epoch number of times
    print("[*] Training on clients in progress")
    for i in server.CHOSEN:
        if i.TRAINING: 
            server.UPDATES[i] = i.local_training(epochs) # Store the locally trained paramters for fedAVG
    server.fedAvg() # Run weighted average on the received local parameters and update the global parameters for the next round
