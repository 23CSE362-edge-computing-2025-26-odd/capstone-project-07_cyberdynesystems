import numpy as np
import tensorflow as tf
import random
import copy
import sys

# --- 1. Neural Network Model ---
# This class remains the same.
class SimpleNN:
    """A simple 2-layer neural network."""
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        self.w1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))

    def get_params(self):
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def set_params(self, params_dict: dict):
        self.w1 = params_dict["w1"]
        self.b1 = params_dict["b1"]
        self.w2 = params_dict["w2"]
        self.b2 = params_dict["b2"]

    def get_weights(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def set_weights(self, weights):
        self.w1, self.b1, self.w2, self.b2 = weights

    def forward(self, x):
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.maximum(0, z1) # ReLU activation
        z2 = np.dot(self.w2, a1) + self.b2
        exp_scores = np.exp(z2 - np.max(z2))
        a2 = exp_scores / np.sum(exp_scores, axis=0)
        return a2

    def flatten_weights(self):
        return np.concatenate([w.flatten() for w in self.get_weights()])

    def unflatten_weights(self, flat_weights):
        shapes = [w.shape for w in self.get_weights()]
        unflattened = []
        start = 0
        for shape in shapes:
            size = np.prod(shape)
            unflattened.append(flat_weights[start:start+size].reshape(shape))
            start += size
        self.set_weights(unflattened)

# --- 2. CI Part 1 (Client-Side): Particle Swarm Optimizer ---
class ParticleSwarmOptimizer:
    """Optimizes NN weights using PSO."""
    def __init__(self, model_template, client_data, n_particles=15, w=0.5, c1=0.8, c2=0.9):
        self.model_template = model_template
        self.client_data = client_data
        self.n_particles = n_particles
        self.w, self.c1, self.c2 = w, c1, c2

        # Each particle is a set of NN weights
        self.particles_pos = [model_template.flatten_weights() + np.random.randn(len(model_template.flatten_weights())) * 0.1 for _ in range(n_particles)]
        self.particles_vel = [np.zeros_like(p) for p in self.particles_pos]
        
        self.pbest_pos = list(self.particles_pos)
        self.pbest_fitness = [self.calculate_fitness(p) for p in self.pbest_pos]
        
        self.gbest_index = np.argmax(self.pbest_fitness)
        self.gbest_pos = self.pbest_pos[self.gbest_index]
        self.gbest_fitness = self.pbest_fitness[self.gbest_index]

    def calculate_fitness(self, particle_pos):
        temp_model = copy.deepcopy(self.model_template)
        temp_model.unflatten_weights(particle_pos)
        correct = sum(1 for x, y in self.client_data if np.argmax(temp_model.forward(x)) == np.argmax(y))
        return correct / len(self.client_data) if self.client_data else 0

    def step(self):
        """Runs one iteration of PSO."""
        for i in range(self.n_particles):
            # Update velocity
            r1, r2 = random.random(), random.random()
            cognitive_vel = self.c1 * r1 * (self.pbest_pos[i] - self.particles_pos[i])
            social_vel = self.c2 * r2 * (self.gbest_pos - self.particles_pos[i])
            self.particles_vel[i] = self.w * self.particles_vel[i] + cognitive_vel + social_vel

            # Update position
            self.particles_pos[i] += self.particles_vel[i]
            
            # Update personal best
            current_fitness = self.calculate_fitness(self.particles_pos[i])
            if current_fitness > self.pbest_fitness[i]:
                self.pbest_fitness[i] = current_fitness
                self.pbest_pos[i] = self.particles_pos[i]
        
        # Update global best
        self.gbest_index = np.argmax(self.pbest_fitness)
        self.gbest_pos = self.pbest_pos[self.gbest_index]
        self.gbest_fitness = self.pbest_fitness[self.gbest_index]

# --- 3. CI Part 2 (Client-Side): Genetic Algorithm for Refinement ---
class GeneticAlgorithmRefiner:
    """Uses GA to refine a population of solutions."""
    def __init__(self, model_template, initial_chromosomes, mutation_rate=0.05, crossover_rate=0.7):
        self.model_template = model_template
        self.population_size = len(initial_chromosomes)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [self._create_model_from_chromosome(c) for c in initial_chromosomes]

    def _create_model_from_chromosome(self, chromosome):
        model = copy.deepcopy(self.model_template)
        model.unflatten_weights(chromosome)
        return model

    # ... (Rest of the GA methods: calculate_fitness, selection, crossover, mutate, evolve, get_best_model_and_fitness) ...
    # These methods are identical to the previous version's GeneticAlgorithmOptimizer
    def calculate_fitness(self, model, data):
        correct = sum(1 for x, y in data if np.argmax(model.forward(x)) == np.argmax(y))
        return correct / len(data) if data else 0

    def selection(self, fitness_scores):
        tournament_size = 5
        def select_one():
            indices = random.sample(range(self.population_size), tournament_size)
            fitness = {i: fitness_scores[i] for i in indices}
            return self.population[max(fitness, key=fitness.get)]
        return select_one(), select_one()

    def crossover(self, parent1, parent2):
        child1, child2 = copy.deepcopy(self.model_template), copy.deepcopy(self.model_template)
        p1_chromo, p2_chromo = parent1.flatten_weights(), parent2.flatten_weights()
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(p1_chromo) - 2)
            c1_chromo = np.concatenate([p1_chromo[:point], p2_chromo[point:]])
            c2_chromo = np.concatenate([p2_chromo[:point], p1_chromo[point:]])
            child1.unflatten_weights(c1_chromo)
            child2.unflatten_weights(c2_chromo)
        else:
            child1.set_weights(parent1.get_weights())
            child2.set_weights(parent2.get_weights())
        return child1, child2

    def mutate(self, model):
        chromosome = model.flatten_weights()
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] += np.random.randn() * 0.1
        model.unflatten_weights(chromosome)
        return model

    def evolve(self, client_data):
        fitness_scores = [self.calculate_fitness(model, client_data) for model in self.population]
        new_population = [copy.deepcopy(self.population[np.argmax(fitness_scores)])]
        while len(new_population) < self.population_size:
            p1, p2 = self.selection(fitness_scores)
            c1, c2 = self.crossover(p1, p2)
            new_population.append(self.mutate(c1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(c2))
        self.population = new_population

    def get_best_model_and_fitness(self, client_data):
        fitness_scores = [self.calculate_fitness(model, client_data) for model in self.population]
        best_index = np.argmax(fitness_scores)
        return self.population[best_index], fitness_scores[best_index]


# --- 4. Federated Learning Components ---
def load_and_prepare_data(num_clients):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784, 1) / 255.0; x_test = x_test.reshape(-1, 784, 1) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10).reshape(-1, 10, 1)
    y_test = tf.keras.utils.to_categorical(y_test, 10).reshape(-1, 10, 1)
    data_zipped = list(zip(x_train, y_train)); random.shuffle(data_zipped)
    client_data, chunk_size = [], len(data_zipped) // num_clients
    for i in range(num_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_clients - 1 else len(data_zipped)
        client_data.append(data_zipped[start:end])
    return client_data, list(zip(x_test, y_test))

def client_update_hybrid_ci(client_data, global_model, pso_steps, ga_generations):
    """Client training using a PSO -> GA hybrid approach."""
    # Stage 1: Exploration with PSO
    pso = ParticleSwarmOptimizer(global_model, client_data)
    print("\n    - Stage 1: Exploring with PSO...")
    for _ in range(pso_steps):
        pso.step()

    # Stage 2: Refinement with GA, seeded by best PSO particles
    print("    - Stage 2: Refining with GA...")
    # Seed the GA with the best solutions found by PSO
    best_pso_particles = [pso.pbest_pos[i] for i in np.argsort(pso.pbest_fitness)[-ga_generations:]]
    ga = GeneticAlgorithmRefiner(global_model, best_pso_particles)
    for _ in range(ga_generations):
        ga.evolve(client_data)

    best_model, fitness = ga.get_best_model_and_fitness(client_data)
    return best_model.get_weights(), fitness

def federated_averaging(models_weights):
    """Standard, simple federated averaging."""
    if not models_weights: return None
    avg_weights = [np.mean([client_weights[i] for client_weights in models_weights], axis=0) for i in range(len(models_weights[0]))]
    return avg_weights

def evaluate_model(model, test_data):
    correct = sum(1 for x, y in test_data if np.argmax(model.forward(x)) == np.argmax(y))
    return correct / len(test_data)

# --- 5. Main Simulation Loop ---
if __name__ == "__main__":
    NUM_CLIENTS, CLIENT_FRACTION, COMM_ROUNDS = 10, 0.5, 20
    # CI parameters for the hybrid client optimizer
    PSO_STEPS = 5
    GA_GENERATIONS = 5

    client_datasets, test_dataset = load_and_prepare_data(NUM_CLIENTS)
    global_nn = SimpleNN()

    print("--- Starting Federated Training: Hybrid CI Clients + Simple Server Aggregation ---")
    print(f"Initial Global Model Accuracy: {evaluate_model(global_nn, test_dataset):.4f}")

    for round_num in range(1, COMM_ROUNDS + 1):
        print(f"\n--- Communication Round {round_num}/{COMM_ROUNDS} ---")
        
        num_selected = int(max(1, NUM_CLIENTS * CLIENT_FRACTION))
        selected_indices = random.sample(range(NUM_CLIENTS), num_selected)
        print(f"Selected clients: {selected_indices}")

        client_updates = []
        for client_idx in selected_indices:
            print(f"  Training client {client_idx} with Hybrid CI (PSO -> GA)...")
            client_data = client_datasets[client_idx]
            weights, fitness = client_update_hybrid_ci(client_data, copy.deepcopy(global_nn), PSO_STEPS, GA_GENERATIONS)
            client_updates.append(weights)
            print(f"    Client {client_idx} finished. Best local fitness: {fitness:.4f}")
        
        if not client_updates: continue
        print("Server: Aggregating client models with simple averaging...")
        global_weights = federated_averaging(client_updates)
        global_nn.set_weights(global_weights)

        accuracy = evaluate_model(global_nn, test_dataset)
        print(f"Global Model Accuracy after round {round_num}: {accuracy:.4f}")

    print("\n--- Federated Training Finished ---")
