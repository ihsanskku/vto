#VTO with DoubleDQN v4. (with reward, successfull/unsuccessful tasks , and episode cost graph, extended code with validation)

import csv
import gym
from gym import spaces
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tabulate import tabulate  # Import the tabulate library

class Task:
    def __init__(self, task_id, cpu_cores, min_acceptable_delay, task_complexity):
        self.task_id = task_id
        self.cpu_cores = cpu_cores
        self.min_acceptable_delay = min_acceptable_delay
        self.task_complexity = task_complexity

class Vehicle:
    def __init__(self, vehicle_id, cpu_cores):
        self.vehicle_id = vehicle_id
        self.cpu_cores = cpu_cores

class EdgeServer:
    def __init__(self, server_id, cpu_cores):
        self.server_id = server_id
        self.cpu_cores = cpu_cores

class CloudServer:
    def __init__(self, server_id, cpu_cores):
        self.server_id = server_id
        self.cpu_cores = cpu_cores

def generate_tasks(num_tasks):
    tasks = []
    for i in range(num_tasks):
        cpu_cores = random.randint(1, 10)
        min_acceptable_delay = round(random.uniform(0.1, 1.0), 2)  # Generate a random delay between 0.1 and 1.0
        task_complexity = round((cpu_cores - 1) / (10 - 1), 2)  # Generate a random task complexity between 0 and 1
        tasks.append(Task(i, cpu_cores, min_acceptable_delay, task_complexity))
    return tasks

def create_resources(num_vehicles, num_edge_servers, num_cloud_servers):
    vehicles = [Vehicle(i, random.randint(2, 10)) for i in range(num_vehicles)]
    edge_servers = [EdgeServer(i, random.randint(20, 30)) for i in range(num_edge_servers)]
    cloud_servers = [CloudServer(i, random.randint(50, 100)) for i in range(num_cloud_servers)]
    return vehicles, edge_servers, cloud_servers

# Custom Gym Environment
class TaskOffloadingEnv(gym.Env):
    def __init__(self, num_tasks, num_vehicles, num_edge_servers, num_cloud_servers):
        super(TaskOffloadingEnv, self).__init__()

        # Initialize your resources and tasks
        self.tasks = generate_tasks(num_tasks)
        self.vehicles, self.edge_servers, self.cloud_servers = create_resources(num_vehicles, num_edge_servers, num_cloud_servers)

        # Define the action space (0: Vehicle, 1: Edge Server, 2: Cloud Server)
        self.action_space = spaces.Discrete(3)

        # Define the observation (state) space based on your scenario
        self.observation_space = spaces.Box(low=0, high=1, shape=(
        num_tasks * 3 + num_vehicles * 4 + num_edge_servers * 4 + num_cloud_servers * 4,))

        # Initialize current task index and resource availability
        self.current_task_idx = 0
        self.resource_availability = {
            'vehicles': [{'cores': vehicle.cpu_cores, 'delay': 0.1, 'cost': 0.15, 'distance': random.randint(10, 100)}
                         for vehicle in self.vehicles],
            'edge_servers': [
                {'cores': edge_server.cpu_cores, 'delay': 0.2, 'cost': 0.25, 'distance': random.randint(10, 100)} for
                edge_server in self.edge_servers],
            'cloud_servers': [
                {'cores': cloud_server.cpu_cores, 'delay': 0.3, 'cost': 0.35, 'distance': random.randint(10, 100)} for
                cloud_server in self.cloud_servers]
        }

    def reset(self):
        # Reset the environment to initial state and return the initial observation
        self.current_task_idx = 0
        self.tasks = generate_tasks(num_tasks)
        self.resource_availability = {
            'vehicles': [{'cores': vehicle.cpu_cores, 'delay': 0.1, 'cost': 0.15, 'distance': random.randint(10, 100)}
                         for vehicle in self.vehicles],
            'edge_servers': [
                {'cores': edge_server.cpu_cores, 'delay': 0.2, 'cost': 0.25, 'distance': random.randint(10, 100)} for
                edge_server in self.edge_servers],
            'cloud_servers': [
                {'cores': cloud_server.cpu_cores, 'delay': 0.3, 'cost': 0.35, 'distance': random.randint(10, 100)} for
                cloud_server in self.cloud_servers]
        }
        return self._get_state()

    def _get_state(self):
        # Construct and return the observation (state) vector
        state = []
        for task in self.tasks:
            state.append(task.cpu_cores)
            state.append(task.min_acceptable_delay)  # Include min_acceptable_delay in state
            state.append(task.task_complexity)  # Include task_complexity in state
        for resource_type, resources in self.resource_availability.items():
            for resource in resources:
                state.append(resource['cores'])
                state.append(resource['delay'])
                state.append(resource['cost'])
                state.append(resource['distance'])
        return state

    def step(self, action):
        task = self.tasks[self.current_task_idx]
        resource_type = ['vehicles', 'edge_servers', 'cloud_servers'][action]

        processing_info = self._process_task(task, resource_type)
        if processing_info is not None:
            reward = self._calculate_reward(task, processing_info, resource_type)

        else:
            reward = -1  # Set a large negative reward for unprocessable tasks

        self.current_task_idx += 1
        done = self.current_task_idx >= len(self.tasks)

        return self._get_state(), reward, processing_info, done, {}

    def _process_task(self, task, resource_type):
        resource_list = self.resource_availability[resource_type]
        for resource in resource_list:
            if task.cpu_cores <= resource['cores']:
                processing_delay = resource['delay']  # Simulated processing delay
                processing_cost = resource['cost']  # Simulated processing cost
                resource_success_reward = 1  # +1 for successful processing
                resource['cores'] -= task.cpu_cores
                resource_distance = resource['distance']
                return {'delay': processing_delay, 'cost': processing_cost,
                        'resource_success_reward': resource_success_reward, 'distance': resource_distance, }
        return None

    def _calculate_reward(self, task, processing_info, resource_type):
        task_complexity = task.task_complexity
        task_acceptable_delay = task.min_acceptable_delay
        resource_delay = processing_info['delay']
        resource_cost = processing_info['cost']
        resource_distance = processing_info['distance']
        resource_success_reward = processing_info['resource_success_reward']

        delay_factor = (task_acceptable_delay - resource_delay) / task_acceptable_delay
        cost_factor = resource_cost
        distance_factor = 1 - (resource_distance / 100)  # Normalize distance to a factor

        if 0 <= task_complexity <= 0.4 and resource_type == 'vehicles':
            complexity_factor = 1
        elif 0.4 < task_complexity <= 0.6 and resource_type == 'edge_servers':
            complexity_factor = 1
        elif task_complexity > 0.6 and resource_type == 'cloud_servers':
            complexity_factor = 1
        else:
            complexity_factor = 0

        reward = delay_factor + cost_factor + distance_factor + complexity_factor + resource_success_reward

        # normalized_reward = (reward + 4) / 8  # Normalize the reward between 0 and 1
        # reward = normalized_reward

        return reward

# DQN Network=================================================================
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_q_network = QNetwork(state_dim, action_dim)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize target network with the same weights as the main network
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return np.argmax(q_values.numpy())

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        q_values = self.q_network(state_batch)
        next_q_values = self.target_q_network(next_state_batch)  # Use target network for next state

        target_q_values = reward_batch + self.gamma * (1 - done_batch) * torch.max(next_q_values, dim=1)[0]

        q_values = q_values.gather(1, action_batch.unsqueeze(1))
        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# validate_agent
def validate_agent(env, agent, num_validation_episodes):
    validation_rewards = []

    for episode in range(num_validation_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0)  # Choose actions greedily (no exploration)
            next_state, reward, _, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        validation_rewards.append(episode_reward)

    average_validation_reward = sum(validation_rewards) / num_validation_episodes

    return average_validation_reward

# Training Loop
def train_double_dqn(env, agent, num_episodes, batch_size, epsilon_decay, validation_interval, num_validation_episodes):
    epsilon = 1.0
    epsilon_min = 0.01
    episode_rewards = []
    validation_rewards = []
    episode_cumulative_costs = []
    episode_successful_tasks = []
    episode_unsuccessful_tasks = []

    replay_buffer = ReplayBuffer(max_size=100000)

    for episode in range(num_episodes):
        state = env.reset()
        #print(state)
        episode_reward = 0
        episode_successful = 0
        episode_unsuccessful = 0
        episode_cumulative_cost = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, processing_info, done, _ = env.step(action)

            episode_reward += reward

            if reward > 0:
                episode_successful += 1
                episode_cumulative_cost += processing_info['cost']

            elif reward < 0:
                episode_unsuccessful += 1

            replay_buffer.add(state, action, reward, next_state, done)
            agent.update(replay_buffer, batch_size)
            state = next_state

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(episode_reward)
        episode_cumulative_costs.append(episode_cumulative_cost)
        episode_successful_tasks.append(episode_successful)
        episode_unsuccessful_tasks.append(episode_unsuccessful)


        if (episode) % validation_interval == 0:
            validation_reward = validate_agent(env, agent, num_validation_episodes)
            validation_rewards.append(validation_reward)
            print("=====================================================================================================")
            print(f"Episode {episode}: Validation Reward = {validation_reward:.2f}")
            print("=====================================================================================================")

        print(f"Episode {episode}: Total Reward = {episode_reward}, Successful Tasks = {episode_successful}, Unsuccessful Tasks = {episode_unsuccessful}, Episode Cost = {episode_cumulative_cost}")

    return episode_rewards,validation_rewards, episode_cumulative_costs, episode_successful_tasks, episode_unsuccessful_tasks


if __name__ == "__main__":
    num_tasks = 200
    num_vehicles = 200
    num_edge_servers = 1
    num_cloud_servers = 1

    env = TaskOffloadingEnv(num_tasks, num_vehicles, num_edge_servers, num_cloud_servers)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DoubleDQNAgent(state_dim, action_dim)

    epsilon_decay = 0.995
    batch_size = 32
    num_episodes = 3000

    validation_interval = 50  # Validate the agent every 200 episodes
    num_validation_episodes = 10  # Number of episodes for validation

    episode_rewards, validation_rewards, episode_cumulative_costs, episode_successful_tasks, episode_unsuccessful_tasks = train_double_dqn(env, agent, num_episodes, batch_size, epsilon_decay, validation_interval,  num_validation_episodes)

    # Saving episode rewards, successful tasks, and unsuccessful tasks to a CSV file
    with open('episode_results.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Episode', 'Reward', 'Successful Tasks', 'Unsuccessful Tasks'])
        for episode, (reward, successful, unsuccessful) in enumerate(zip(episode_rewards, episode_successful_tasks, episode_unsuccessful_tasks)):
            csv_writer.writerow([episode, reward, successful, unsuccessful])

    max_reward = max(episode_rewards)
    normalized_episode_rewards = [reward / max_reward for reward in episode_rewards]

    # Plotting reward
    plt.plot(normalized_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.title("Average Episode Rewards")
    plt.savefig('episode_rewards_VTO_DDQN_v4.png')
    plt.show()

    # Plotting successful and unsuccessful tasks
    plt.plot(episode_successful_tasks, label='Successful Tasks')
    plt.plot(episode_unsuccessful_tasks, label='Unsuccessful Tasks')
    plt.xlabel("Episode")
    plt.ylabel("Number of Tasks")
    plt.title("Successful and Unsuccessful Tasks")
    plt.legend()
    plt.savefig('Successful_and_Unsuccessful_Tasks_VTO_DDQN_v4.png')
    plt.show()

    # Plot cumulative cost graph at the end of each episode
    plt.plot(episode_cumulative_costs, label='Episode Cumulative Cost')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Cost")
    plt.title("Episode Cumulative Costs")
    plt.legend()
    plt.savefig('episode_cumulative_cost_VTO_DDQN_v4.png')
    plt.show()

 # Plot validation rewards
    plt.plot(validation_rewards, label='Validation Reward')
    plt.xlabel("Validation Interval")
    plt.ylabel("Average Validation Reward")
    plt.title("Average Validation Rewards")
    plt.legend()
    plt.savefig('Average Validation Rewards_VTO_DDQN_v4.png')
    plt.show()