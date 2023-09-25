#State contain task and nodes

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import random
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import wandb


# Define the classes
class Task:
    def __init__(self, task_id, cpu_cores, bandwidth_requirements, data_size, min_acceptable_delay):
        self.task_id = task_id
        self.cpu_cores = cpu_cores
        self.bandwidth_requirements = bandwidth_requirements
        self.data_size = data_size
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
        cpu_cores = random.randint(10, 15)
        bandwidth_requirements = random.randint(1, 5)
        data_size = random.randint(5, 10)
        min_acceptable_delay = random.randint(1, 5)
        tasks.append(Task(i, cpu_cores, bandwidth_requirements, data_size, min_acceptable_delay))
    return tasks


def generate_devices(num_vehicles, num_rsus, num_cloud_servers):
    vehicles = [Vehicle(cpu_cores=random.randint(100, 200), bandwidth=random.randint(100, 101),
                        processing_delay=random.randint(1, 2), cost=1,
                        distance= 0) for _ in range(num_vehicles)]
    rsus = [RoadSideUnit(cpu_cores=random.randint(300, 400), bandwidth=random.randint(200, 201),
                         processing_delay=random.randint(3, 4), cost=2,
                         distance= 0) for _ in range(num_rsus)]
    cloud_servers = [CloudServer(cpu_cores=random.randint(800, 900), bandwidth=random.randint(200, 201),
                                 processing_delay=random.randint(5, 6), cost=3,
                                 distance= 0) for _ in range(num_cloud_servers)]
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
# class DQN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(DQN, self).__init__()
#         self.gcn = GCN(input_dim, hidden_dim, 128)  # (input_dim[node feature 5], hidden_dim, output_dim) for GCN model
#         self.fc1 = nn.Linear(128, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)  # Output Q-values for each node
#
#     def forward(self, data):
#         embedding = self.gcn(data)
#         x = F.relu(self.fc1(embedding))
#         x = F.relu(self.fc2(x))
#         q_values = self.fc3(x)
#         #q_values = torch.sum(q_values, dim=1)
#         return q_values

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.avg_pooling = nn.AvgPool2d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * (output_dim + num_tasks), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)    # Output Q-values for each node

    def forward(self, data):
        node_feature, edge_index = data.x, data.edge_index
        embedding = F.relu(self.conv1(node_feature, edge_index))
        embedding = self.conv2(embedding, edge_index)
        # embedding = self.avg_pooling(embedding)
        embedding = torch.flatten(embedding)
        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        # q_values = torch.sum(q_values, dim=1)
        return q_values


# ===============================================================================================

def process_network(vehicles, rsus, cloud_servers, tasks):
    # Create a graph
    G = nx.Graph()

    # Add nodes for vehicles, RSUs, and cloud servers with attributes as features
    for vehicle in vehicles:
        G.add_node(vehicle, type="Vehicle", cpu_cores=vehicle.cpu_cores, bandwidth=vehicle.bandwidth,
                   processing_delay=vehicle.processing_delay, cost=vehicle.cost, distance=vehicle.distance)

    for rsu in rsus:
        G.add_node(rsu, type="RoadSideUnit", cpu_cores=rsu.cpu_cores, bandwidth=rsu.bandwidth,
                   processing_delay=rsu.processing_delay, cost=rsu.cost, distance=rsu.distance)

    for cloud_server in cloud_servers:
        G.add_node(cloud_server, type="CloudServer", cpu_cores=cloud_server.cpu_cores,
                   bandwidth=cloud_server.bandwidth, processing_delay=cloud_server.processing_delay,
                   cost=cloud_server.cost, distance=cloud_server.distance)

    # Add task nodes with attributes as features
    for task in tasks:
        G.add_node(task, type="Task", cpu_cores=task.cpu_cores, bandwidth_requirements=task.bandwidth_requirements, data_size=task.data_size,
                   min_acceptable_delay=task.min_acceptable_delay)

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

    # Create feature vectors for nodes, considering different attributes for task nodes
    x = torch.tensor([
        [G.nodes[node]['cpu_cores'], G.nodes[node]['bandwidth'], G.nodes[node]['processing_delay'],
         G.nodes[node]['cost'], G.nodes[node]['distance']]
        if G.nodes[node]['type'] != 'Task'  # Use attributes for non-task nodes
        else
        [G.nodes[node]['cpu_cores'], G.nodes[node]['data_size'], G.nodes[node]['bandwidth_requirements'],
         G.nodes[node]['min_acceptable_delay'], 0.0]  # Define attributes for task nodes
        for node in G.nodes()
    ], dtype=torch.float)

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
        'red' if G.nodes[node]['type'] == 'CloudServer' else
        'yellow'   # Assign a different color for 'Task' nodes
        for node in G.nodes()
    }

    labels = {}
    for node in G.nodes():
        #label = f"CPU Cores: {G.nodes[node]['cpu_cores']}\n"
        if G.nodes[node]['type'] != 'Task':
            label = f"Types: {G.nodes[node]['type']}\n"
            label += f"CPU Cores: {G.nodes[node]['cpu_cores']}\n"
            label += f"Bandwidth: {G.nodes[node].get('bandwidth', 'N/A')}\n"  # Use 'N/A' if attribute is missing
            label += f"Processing Delay: {G.nodes[node].get('processing_delay', 'N/A')}\n"  # Use 'N/A' if attribute is missing
            label += f"Cost: {G.nodes[node].get('cost', 'N/A')}\n"  # Use 'N/A' if attribute is missing
            label += f"Distance: {G.nodes[node]['distance']}"
        else:
            label = f"Types: {G.nodes[node]['type']}\n"
            label += f"CPU Cores: {G.nodes[node]['cpu_cores']}\n"
            label += f"Data size: {G.nodes[node]['data_size']}\n"
            label += f"Bandwidth req.: {G.nodes[node]['bandwidth_requirements']}\n"
            label += f"Min_acceptable_delay.: {G.nodes[node]['min_acceptable_delay']}\n"

        labels[node] = label

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

class TaskAllocationEnv(gym.Env):
    def __init__(self, num_tasks, num_vehicles, num_rsus, num_cloud_servers):
        super(TaskAllocationEnv, self).__init__()

        # Define action and observation spaces
        num_actions = num_vehicles + num_rsus + num_cloud_servers
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Define other environment-specific parameters
        self.max_steps = 3  # Maximum number of steps per episode
        self.current_step = 0  # Current step

        # Initialize tasks
        self.tasks = generate_tasks(num_tasks)
        self.current_task_idx = 0

        # Initialize devices
        self.vehicles, self.rsus, self.cloud_servers = generate_devices(num_vehicles, num_rsus, num_cloud_servers)

        # Initialize network
        self.G, self.data = process_network(self.vehicles, self.rsus, self.cloud_servers, self.tasks)

    def reset(self):
        # Reset the environment to the initial state

        self.current_step = 0
        self.current_task_idx = 0

        # Reset tasks
        self.tasks = generate_tasks(num_tasks)
        self.current_task_idx = 0

        ## Reset tasks
        self.vehicles, self.rsus, self.cloud_servers = generate_devices(num_vehicles, num_rsus, num_cloud_servers)

        # Reset  network
        self.G, self.data = process_network(self.vehicles, self.rsus, self.cloud_servers, self.tasks)

        # Initialize state
        self.state = self.data

        return self.state

    def step(self, action):
        print("Selected node", self.data.x[action])

        task = self.tasks[self.current_task_idx]

        # Calculate the reward and get the modified selected_node
        reward = self._calculate_reward(task)

        print(f"Step {env.current_step}: Step Reward: {reward}, Selected_Task: {task.__dict__}, Selected Node: CPU, {self.data.x[action][0]}, BW:{self.data.x[action][1]}, Pro.delay: {self.data.x[action][2]}, Cost: {self.data.x[action][3]}, Dist: {self.data.x[action][4]}")

        #print()
        # print( "Data after action\n", self.data.x)

        next_state = self.data
        #print("Next_state\n", next_state.x)

        print("--------------------------------------------------------------------------------------------------------")

        self.current_task_idx += 1
        self.current_step += 1
        done = self.current_task_idx >= len(self.tasks)

        return next_state, reward, done, {}

    def _calculate_reward(self, task):

        # Check if the selected node has enough CPU cores, bandwidth, and memory for the task
        if (self.data.x[action][0] >= task.cpu_cores and
            self.data.x[action][1] >= task.bandwidth_requirements):
            # selected_node.memory >= task.bandwidth_requirements):

            # Reduce the CPU cores and bandwidth of the selected node based on task requirements
            self.data.x[action][0] -= task.cpu_cores
            self.data.x[action][1] -= task.bandwidth_requirements

            # Task success reward
            success_task_reward = 10

            # Calculate the delay penalty based on the node's processing delay
            delay_penalty = max(0, self.data.x[action][2] - task.min_acceptable_delay)

            # Calculate the cost penalty based on node cost (if applicable)
            cost_penalty = self.data.x[action][3]

            # Calculate a combined penalty based on delay and cost factors
            combined_penalty = delay_penalty + cost_penalty

            # Calculate the reward as a negative of the combined penalty
            reward = success_task_reward - combined_penalty

        else:
            # If the selected node doesn't meet the resource requirements, return a negative reward
            reward = -1.0  # Return as a float

        return reward  # Return as a float, not a tuple


# Create your network
num_tasks = 50
num_vehicles = 50
num_rsus = 50
num_cloud_servers = 3

#wandb.login(key="31e5f2e0e26bf315d832e7fa7185b9ddd59adc32")

# wandb.init(project="VTO_DQN_GCN", config={
#      "num_tasks": num_tasks,
#      "num_vehicles": num_vehicles,
#      "num_rsus": num_rsus,
#      "num_cloud_servers": num_cloud_servers
# })

# Instantiate the custom environment
env = TaskAllocationEnv(num_tasks, num_vehicles, num_rsus, num_cloud_servers)

plot_network(env.G)

# Instantiate the DQN model
input_dim_dqn = env.data.x.size(1)  # Update with the correct input dimension
hidden_dim_dqn = 64
output_dim_dqn = env.action_space.n
dqn_model = DQN(input_dim_dqn, hidden_dim_dqn, output_dim_dqn)
target_dqn_model = DQN(input_dim_dqn, hidden_dim_dqn, output_dim_dqn)
target_dqn_model.load_state_dict(dqn_model.state_dict())  # Initialize target network with DQN weights

# Define DQN hyperparameters
learning_rate_dqn = 0.01
gamma = 0.99  # Discount factor
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Define the optimizer for the DQN
dqn_optimizer = torch.optim.Adam(dqn_model.parameters(), lr=learning_rate_dqn)
dqn_loss = nn.MSELoss()

# DQN Training
num_episodes = 1000
episode_rewards = []  # List to store episode rewards


for episode in range(num_episodes):

    state = env.reset()
    #print("State (Data.x)\n", env.state.x)

    episode_reward = 0.0
    successful_tasks = 0
    unsuccessful_tasks = 0
    rejected_tasks = 0
    done = False

    while not done:
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        #print("epsilon: ", epsilon)
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore (select a random action)
            print("Random Action:", action)
        else:
            with torch.no_grad():
                q_value = dqn_model(state)
                print("Q_values \n", q_value)
                #action = q_value.argmax().item() #Exploit (select the action with the highest Q-value)
                print("DQN action:", action)

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # print("data at next state\n",next_state)

        if reward >= 0:
            successful_tasks += 1
        else:
            unsuccessful_tasks += 1

        state = next_state

        #print("next_state \n", state.x)
        # Log summary statistics for this episode
        # wandb.log({
        #     "Epsilon": epsilon
        # })

    episode_rewards.append(episode_reward)

    #Log summary statistics for this episode
    # wandb.log({
    #     "Mean Episode Reward": np.mean(episode_rewards),
    #
    # })

    # Print the counts of successful, unsuccessful tasks, and episode reward for this episode
    #print()
    #print("\n============================================================================================================================================================================================================================================================")

    print( f"Episode {episode} - Successful Tasks: {successful_tasks}, Unsuccessful Tasks: {unsuccessful_tasks}, Episode Reward: {episode_reward}")

    #print( "\n============================================================================================================================================================================================================================================================")

# Plot episode rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("DQN Episode Rewards")
plt.show()