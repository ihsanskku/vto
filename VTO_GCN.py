import random
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt


# Define the classes
class Task:
    def __init__(self, task_id, cpu_cores, data_size, memory_requirements, min_acceptable_delay):
        self.task_id = task_id
        self.cpu_cores = cpu_cores
        self.data_size = data_size
        self.memory_requirements = memory_requirements
        self.min_acceptable_delay = min_acceptable_delay


class Vehicle:
    def __init__(self, cpu_cores, bandwidth, processing_delay, cost, distance):
        self.cpu_cores = cpu_cores
        self.bandwidth = bandwidth
        self.processing_delay = processing_delay
        self.cost = cost
        self.distance = distance


class RoadSideUnit:
    def __init__(self, cpu_cores, bandwidth, processing_delay, cost, distance):
        self.cpu_cores = cpu_cores
        self.bandwidth = bandwidth
        self.processing_delay = processing_delay
        self.cost = cost
        self.distance = distance


class CloudServer:
    def __init__(self, cpu_cores, bandwidth, processing_delay, cost, distance):
        self.cpu_cores = cpu_cores
        self.bandwidth = bandwidth
        self.processing_delay = processing_delay
        self.cost = cost
        self.distance = distance


def generate_tasks(num_tasks):
    tasks = []
    for i in range(num_tasks):
        cpu_cores = random.randint(20, 50)
        data_size = random.randint(1, 3)
        memory_requirements = random.randint(10, 20)
        min_acceptable_delay = random.randint(1, 7)
        tasks.append(Task(i, cpu_cores, data_size, memory_requirements, min_acceptable_delay))
    return tasks


def generate_devices(num_vehicles, num_rsus, num_cloud_servers):
    vehicles = [Vehicle(cpu_cores=random.randint(50, 100), bandwidth=random.randint(100, 200),
                        processing_delay=random.randint(1, 3), cost=random.randint(5, 10),
                        distance=random.randint(100, 300)) for _ in range(num_vehicles)]
    rsus = [RoadSideUnit(cpu_cores=random.randint(200, 400), bandwidth=random.randint(200, 300),
                         processing_delay=random.randint(3, 5), cost=random.randint(10, 15),
                         distance=random.randint(300, 400)) for _ in range(num_rsus)]
    cloud_servers = [CloudServer(cpu_cores=random.randint(400, 1000), bandwidth=random.randint(400, 1000),
                                 processing_delay=random.randint(5, 7), cost=random.randint(15, 20),
                                 distance=random.randint(400, 500)) for _ in range(num_cloud_servers)]
    return vehicles, rsus, cloud_servers


# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Define a DQN class
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output Q-values for each node

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


def process_network(vehicles, rsus, cloud_servers):
    # Create a graph
    G = nx.Graph()

    # Add nodes to the graph with attributes as features
    for vehicle in vehicles:
        G.add_node(vehicle, type="Vehicle", cpu_cores=vehicle.cpu_cores, bandwidth=vehicle.bandwidth,
                   processing_delay=vehicle.processing_delay, cost=vehicle.cost, distance=vehicle.distance)

    for rsu in rsus:
        G.add_node(rsu, type="RoadSideUnit", cpu_cores=rsu.cpu_cores, bandwidth=rsu.bandwidth,
                   processing_delay=rsu.processing_delay, cost=rsu.cost, distance=rsu.distance)

    for cloud_server in cloud_servers:
        G.add_node(cloud_server, type="CloudServer", cpu_cores=cloud_server.cpu_cores, bandwidth=cloud_server.bandwidth,
                   processing_delay=cloud_server.processing_delay, cost=cloud_server.cost,
                   distance=cloud_server.distance)

    # Create fully connected topology by adding edges
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                G.add_edge(node1, node2)

    # Convert the network to a PyTorch Geometric Data object
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    edges = list(G.edges())
    edge_index = torch.tensor([[node_to_index[src], node_to_index[dst]] for src, dst in edges],
                              dtype=torch.long).t().contiguous()
    x = torch.tensor([[G.nodes[node]['cpu_cores'], G.nodes[node]['bandwidth'], G.nodes[node]['processing_delay'],
                       G.nodes[node]['cost'], G.nodes[node]['distance']] for node in G.nodes()], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    return G, data


def train_gcn(model, data, num_episodes=100, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        # Forward pass
        outputs = model(data)

        # Generate some dummy target values (you should replace this with your actual target values)
        targets = torch.rand_like(outputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for monitoring
        # print(f"Episode [{episode + 1}/{num_episodes}], Loss: {loss.item()}")

    return model


def plot_network(G):
    # Plot the network with node colors and font size
    pos = nx.spring_layout(G)  # You can change the layout algorithm if needed

    # Define node colors based on type
    node_colors = {
        node: 'blue' if G.nodes[node]['type'] == 'Vehicle' else
        'green' if G.nodes[node]['type'] == 'RoadSideUnit' else
        'red'
        for node in G.nodes()
    }

    labels = {
        node: f"CPU Cores: {G.nodes[node]['cpu_cores']}\nBandwidth: {G.nodes[node]['bandwidth']}\nProcessing Delay: {G.nodes[node]['processing_delay']}\nCost: {G.nodes[node]['cost']}\nDistance: {G.nodes[node]['distance']}"
        for node in G.nodes()
    }

    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=100,
        font_size=8,
        font_color='black',  # Font color for labels
        node_color=[node_colors[node] for node in G.nodes()],
    )

    plt.title("Fully Connected Network Topology")
    plt.show()


class CustomEnv(gym.Env):
    def __init__(self, num_tasks, num_vehicles, num_rsus, num_cloud_servers, final_node_embeddings):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        num_actions = num_vehicles + num_rsus + num_cloud_servers
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(final_node_embeddings.shape[0],),
                                            dtype=np.float32)




        # Define your initial state (e.g., final_node_embeddings)
        self.state = final_node_embeddings

        # Define other environment-specific parameters
        self.max_steps = 3  # Maximum number of steps per episode
        self.current_step = 0  # Current step

    def reset(self):
        # Reset the environment to the initial state
        print(
            " reset the env--------------------------------------------------------------------------------------------")

        self.tasks = generate_tasks(num_tasks)
        vehicles, rsus, cloud_servers = generate_devices(num_vehicles, num_rsus, num_cloud_servers)

        # Update network applied action and resource changes
        G, data = process_network(vehicles, rsus, cloud_servers)

        # Instantiate the GCN model
        input_dim = data.x.size(1)  # Number of input features (5 per node)
        hidden_dim = 64
        output_dim = 64
        gcn_model = GCN(input_dim, hidden_dim, output_dim)

        # Train the GCN model for updated network and resources and get embedding
        trained_model = train_gcn(gcn_model, data, num_episodes=100)
        final_node_embeddings = trained_model(data)
        state = final_node_embeddings

        # Plot the network
        # plot_network(G)

        self.current_step = 0
        self.current_task_idx = 0
        return self.state, G

    def step(self, action, G, state):
        task = self.tasks[self.current_task_idx]
        selected_node = list(G.nodes())[action]

        # Print details of the current step
        print(f"Step: {self.current_step}")
        print(f"Step Action taken: {action}")
        print(
            f"Task Details: ID={task.task_id}, CPU Cores={task.cpu_cores}, Data Size={task.data_size}, Min Delay={task.min_acceptable_delay}")
        print(
            f"Selected Node Details: CPU Cores={selected_node.cpu_cores}, Bandwidth={selected_node.bandwidth}, Processing Delay={selected_node.processing_delay}, Cost={selected_node.cost}, Distance={selected_node.distance}")

        if selected_node.cpu_cores >= task.cpu_cores:
            selected_node.cpu_cores -= task.cpu_cores
            reward = calculate_reward(task, selected_node)

            # Update network applied action and resource changes
            G, data = process_network(vehicles, rsus, cloud_servers)
            # Train the GCN model for updated network and resources and get embedding
            trained_model = train_gcn(gcn_model, data, num_episodes=100)
            final_node_embeddings = trained_model(data)
            next_state = final_node_embeddings
        else:
            next_state = state
            reward = -1

        # Print the reward for the current step
        print(f"Reward: {reward}")

        # Plot the network
        # plot_network(G)

        self.current_task_idx += 1
        self.current_step += 1
        done = self.current_task_idx >= len(self.tasks)

        # Print a separator for clarity
        print("-" * 60)

        return next_state, reward, done, {}


def calculate_reward(task, selected_node):
    # Check if the selected node has enough CPU cores for the task
    if selected_node.cpu_cores >= task.cpu_cores:
        # Calculate a delay penalty (higher delay is penalized)
        delay_penalty = (selected_node.processing_delay - task.min_acceptable_delay)

        # Calculate a cost penalty (higher cost is penalized)
        cost_penalty = selected_node.cost
        success = 10

        # Calculate the reward as an inverse of the total penalty (higher penalty gets lower reward)
        reward = success - selected_node.cost / 10 + delay_penalty / 10

        return reward

    # If the selected node doesn't have enough CPU cores, return a negative reward
    return -1.0


# Create your network
num_tasks = 1
num_vehicles = 1
num_rsus = 1
num_cloud_servers = 20


# Call the generate_devices method on the environment instance
vehicles, rsus, cloud_servers = generate_devices(num_vehicles, num_rsus, num_cloud_servers)

# Process the network and obtain data and model and plot
G, data = process_network(vehicles, rsus, cloud_servers)
# plot_network(G)

# Instantiate the GCN model
input_dim = data.x.size(1)  # Number of input features (5 per node)
hidden_dim = 64
output_dim = 64
gcn_model = GCN(input_dim, hidden_dim, output_dim)

# Train the GCN model for number of episodes
trained_model = train_gcn(gcn_model, data, num_episodes=10)
final_node_embeddings = trained_model(data)

# Create an instance of the environment
env = CustomEnv(num_tasks, num_vehicles, num_rsus, num_cloud_servers, final_node_embeddings)

# Instantiate the DQN model
input_dim_dqn = final_node_embeddings.size(1)  # Update with the correct input dimension
hidden_dim_dqn = 64
output_dim_dqn = env.action_space.n
dqn_model = DQN(input_dim_dqn, hidden_dim_dqn, output_dim_dqn)
target_dqn_model = DQN(input_dim_dqn, hidden_dim_dqn, output_dim_dqn)
target_dqn_model.load_state_dict(dqn_model.state_dict())  # Initialize target network with DQN weights


# Define DQN hyperparameters
learning_rate_dqn = 0.01
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

# Define the optimizer for the DQN
dqn_optimizer = torch.optim.Adam(dqn_model.parameters(), lr=learning_rate_dqn)
dqn_loss = nn.MSELoss()

# DQN Training
num_episodes = 100
episode_rewards = []  # List to store episode rewards

# Define initial epsilon and decay rate
initial_epsilon = 0.5
decay_rate = 0.01  # Adjust the decay rate as needed

for episode in range(num_episodes):
    # Update epsilon using decay schedule
    epsilon = initial_epsilon - episode * decay_rate
    epsilon = max(epsilon, 0.1)  # Ensure epsilon doesn't go below a minimum value (e.g., 0.1)

    # Access the action space of the custom environment
    all_possible_actions = list(range(env.action_space.n))

    # Print the list of all possible actions
    print("All Possible Actions:", all_possible_actions)

    state, G = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore
            print("Action is taken randmly (exploration)", action)
        else:
            with torch.no_grad():
                q_value = dqn_model(torch.tensor(state).float())
                print(q_value.size())
                action = q_value.argmax().item()  # Exploit
                print("Action is taken by DQN (Exploitation)", action)

        next_state, reward, done, _ = env.step(action, G, state)
        episode_reward += reward

        # print ("Reward===========", reward)

        # Q-learning update
        with torch.no_grad():
            target_q_value = target_dqn_model(torch.tensor(next_state).float())
            target_max_q = target_q_value.max().item()  # Max Q-value
            target = reward + gamma * target_max_q

        q_value = dqn_model(torch.tensor(state).float())

        # Calculate the mean squared difference as loss
        loss = ((q_value - target) ** 2).mean()

        dqn_optimizer.zero_grad()
        loss.backward()
        dqn_optimizer.step()

        state = next_state

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

# Plot episode rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("DQN Episode Rewards")
plt.show()