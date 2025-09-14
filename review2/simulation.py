import client, server

# Initializing clients 
client1 = client.Client()
client2 = client.Client()
client3 = client.Client()
client4 = client.Client()
client5 = client.Client()
client6 = client.Client()
client7 = client.Client()
client8 = client.Client()

# Providing unique ID to every client connected to the server
server = server.Server()
server.CLIENTS = [client1, client2, client3, client4, client5, client6, client7, client8]
for i in range(len(server.CLIENTS)):
    server.CLIENTS[i].ID = i
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
