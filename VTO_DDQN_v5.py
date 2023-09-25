#VTO with DoubleDQN v4. (with reward, successfull/unsuccessful tasks , and episode cost graph, extended code with validation)....>> task comlexity function + bandwith


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
import wandb
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''
wandb.init(project="task_offloading")
'''

class Task:
    def __init__(self, task_id, cpu_cores, data_size, memory_requirements, min_acceptable_delay, task_complexity):
        self.task_id = task_id
        self.cpu_cores = cpu_cores
        self.data_size = data_size
        self.memory_requirements = memory_requirements
        self.min_acceptable_delay = min_acceptable_delay
        self.task_complexity = task_complexity

class Vehicle:
    def __init__(self, vehicle_id, cpu_cores, bandwidth):
        self.vehicle_id = vehicle_id
        self.cpu_cores = cpu_cores
        self.bandwidth = bandwidth

class RoadSideUnit:
    def __init__(self, server_id, cpu_cores, bandwidth):
        self.server_id = server_id
        self.cpu_cores = cpu_cores
        self.bandwidth = bandwidth

class CloudServer:
    def __init__(self, server_id, cpu_cores, bandwidth):
        self.server_id = server_id
        self.cpu_cores = cpu_cores
        self.bandwidth = bandwidth

def complexity(cpu_cores, data_size, memory_requirements):
    complexity = cpu_cores + data_size + memory_requirements
    min_value = 1 + 10 + 10  # minimum values of pu_requirements, data_size, memory_requirements
    max_value = 10 + 100 + 100  # maximum values of pu_requirements, data_size, memory_requirements
    complexity = round((complexity - min_value) / (max_value - min_value), 2) # Normalize task complexity
    return complexity

def generate_tasks(num_tasks):
    tasks = []
    for i in range(num_tasks):
        cpu_cores = random.randint(1, 10)
        data_size = random.randint(10, 100)
        memory_requirements = random.randint(10, 100)
        min_acceptable_delay = round(random.uniform(0.1, 1.0), 2)
        task_complexity = complexity(cpu_cores, data_size, memory_requirements)
        tasks.append(Task(i, cpu_cores, data_size, memory_requirements, min_acceptable_delay, task_complexity))
    return tasks

def create_resources(num_vehicles, num_rsus, num_cloud_servers):
    vehicles = [Vehicle(i, random.randint(2, 10), random.randint(100, 500)) for i in range(num_vehicles)]
    road_side_units = [RoadSideUnit(i, random.randint(20, 30), random.randint(100, 500)) for i in range(num_rsus)]
    cloud_servers = [CloudServer(i, random.randint(50, 100), random.randint(1000, 2000)) for i in range(num_cloud_servers)]
    return vehicles, road_side_units, cloud_servers

# Custom Gym Environment
class TaskOffloadingEnv(gym.Env):
    def __init__(self, num_tasks, num_vehicles, num_rsus, num_cloud_servers):
        super(TaskOffloadingEnv, self).__init__()

        # Initialize your resources and tasks
        self.tasks = generate_tasks(num_tasks)
        self.vehicles, self.road_side_units, self.cloud_servers = create_resources(num_vehicles, num_rsus, num_cloud_servers)

        # Define the action space (0: Vehicle, 1: Road side units (RSU), 2: Cloud Server)
        self.action_space = spaces.Discrete(3)

        # Define the observation (state) space based on your scenario
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_tasks * 5 + num_vehicles * 5 + num_rsus * 5 + num_cloud_servers * 5,))

        # Initialize current task index and resource availability
        self.current_task_idx = 0
        self.resource_availability = {
            'vehicles': [{'cores': vehicle.cpu_cores, 'delay': 0.1, 'cost': 0.15, 'distance': random.randint(10, 100),
                          'bandwidth': random.randint(100, 500)}  # Assign bandwidth to vehicles
                         for vehicle in self.vehicles],
            'road_side_units': [
                {'cores': road_side_unit.cpu_cores, 'delay': 0.2, 'cost': 0.25, 'distance': random.randint(10, 100),
                 'bandwidth': random.randint(100, 500)}  # Assign bandwidth to RSUs
                for road_side_unit in self.road_side_units],
            'cloud_servers': [
                {'cores': cloud_server.cpu_cores, 'delay': 0.3, 'cost': 0.35, 'distance': random.randint(10, 100),
                 'bandwidth': random.randint(1000, 2000)}  # Assign bandwidth to Cloud Servers
                for cloud_server in self.cloud_servers]
        }
    def reset(self):
        # Reset the environment to initial state and return the initial observation
        self.current_task_idx = 0
        self.resource_availability = {
            'vehicles': [{'cores': vehicle.cpu_cores, 'delay': 0.1, 'cost': 0.15, 'distance': random.randint(10, 100),
                          'bandwidth': random.randint(100, 500)}  # Assign bandwidth to vehicles
                         for vehicle in self.vehicles],
            'road_side_units': [
                {'cores': road_side_unit.cpu_cores, 'delay': 0.2, 'cost': 0.25, 'distance': random.randint(10, 100),
                 'bandwidth': random.randint(100, 500)}  # Assign bandwidth to RSUs
                for road_side_unit in self.road_side_units],
            'cloud_servers': [
                {'cores': cloud_server.cpu_cores, 'delay': 0.3, 'cost': 0.35, 'distance': random.randint(10, 100),
                 'bandwidth': random.randint(1000, 2000)}  # Assign bandwidth to Cloud Servers
                for cloud_server in self.cloud_servers]
        }
        return self._get_state()

    def _get_state(self):
        # Construct and return the observation (state) vector
        state = []
        for task in self.tasks:
            state.append(task.cpu_cores)
            state.append(task.data_size)
            state.append(task.memory_requirements)
            state.append(task.min_acceptable_delay)
            state.append(task.task_complexity)
        for resource_type, resources in self.resource_availability.items():
            for resource in resources:
                state.append(resource['cores'])
                state.append(resource['delay'])
                state.append(resource['cost'])
                state.append(resource['distance'])
                state.append(resource['bandwidth'])  # Include bandwidth in state
        return state

    def step(self, action):
        task = self.tasks[self.current_task_idx]
        resource_type = ['vehicles', 'road_side_units', 'cloud_servers'][action]

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
        elif 0.4 < task_complexity <= 0.6 and resource_type == 'road_side_units':
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


def display_table(resources, resource_type):
    if resource_type == "Task":
        headers = ["Task", "CPU Cores", "Data size", "Memory requirements", "Min Acceptable Delay", "Task complexity"]
        rows = [[f"{resource_type} {idx + 1}", task.cpu_cores, task.data_size, task.memory_requirements,
                 task.min_acceptable_delay, task.task_complexity] for idx, task in enumerate(resources)]
    else:
        headers = ["Resource", "Cores", "Bandwidth", "Delay", "Cost", "Distance"]
        rows = []
        for idx, resource in enumerate(resources):
            if resource_type == "Vehicle":
                rows.append([f"{resource_type} {idx + 1}", resource['cores'], resource['bandwidth'],
                             resource['delay'], resource['cost'], resource['distance']])
            else:
                rows.append([f"{resource_type} {idx + 1}", resource['cores'], resource['bandwidth'],
                             resource['delay'], resource['cost'], resource['distance']])
    print(tabulate(rows, headers=headers, tablefmt="grid"))


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

            ##After calculating episode rewards
            #wandb.log({"Episode Reward": episode_reward})

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
            print("##################################################################################################")
            print(f"Episode {episode}: Validation Reward = {validation_reward:.2f}")
            print("##################################################################################################")

        print(f"Episode {episode}: Total Reward = {round(episode_reward, 2)}, Successful Tasks = {episode_successful}, Unsuccessful Tasks = {episode_unsuccessful}, Episode Cost = {round(episode_cumulative_cost,2)}")

        max_reward = max(episode_rewards)
        normalized_episode_rewards = [reward / max_reward for reward in episode_rewards]

        '''
        wandb.log({"Episode Reward":  normalized_episode_rewards})
        wandb.log({"Validation Reward": validation_reward})
        wandb.log({"Successful Tasks": episode_successful})
        wandb.log({"Unsuccessful Tasks": episode_unsuccessful})
        wandb.log({"Episode_cumulative_cost": episode_cumulative_cost})
        '''

    return  normalized_episode_rewards,validation_rewards, episode_cumulative_costs, episode_successful_tasks, episode_unsuccessful_tasks

if __name__ == "__main__":
    num_tasks = 200
    num_vehicles = 200
    num_rsus = 1   # rsus (road side units)
    num_cloud_servers = 1


    '''
    wandb.init(project="task_offloading")
    wandb.config.num_tasks = num_tasks
    wandb.config.num_vehicles = num_vehicles
    wandb.config.num_rsus = num_rsus
    wandb.config.num_cloud_servers = num_cloud_servers
    '''

    env = TaskOffloadingEnv(num_tasks, num_vehicles, num_rsus, num_cloud_servers)

    # Display task and resource details before training
    print("Task and Resource Details before Training:")
    print("Task Details:")
    display_table(env.tasks, "Task")
    print("\nVehicle Details:")
    display_table(env.resource_availability['vehicles'], "Vehicle")
    print("\nRoad Side Units Details:")
    display_table(env.resource_availability['road_side_units'], "Road Side Unit")
    print("\nCloud Server Details:")
    display_table(env.resource_availability['cloud_servers'], "Cloud Server")

    state_dim = env.observation_space.shape[0]
    print(state_dim)
    action_dim = env.action_space.n

    agent = DoubleDQNAgent(state_dim, action_dim)

    epsilon_decay = 0.995
    batch_size = 32
    num_episodes = 100

    validation_interval = 10  # Validate the agent every 200 episodes
    num_validation_episodes = 10  # Number of episodes for validation

    normalized_episode_rewards, validation_rewards, episode_cumulative_costs, episode_successful_tasks, episode_unsuccessful_tasks = train_double_dqn(env, agent, num_episodes, batch_size, epsilon_decay, validation_interval,  num_validation_episodes)


    # Saving episode rewards, successful tasks, and unsuccessful tasks to a CSV file
    with open('episode_results.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Episode', 'Reward', 'Successful Tasks', 'Unsuccessful Tasks'])
        for episode, (reward, successful, unsuccessful) in enumerate(zip( normalized_episode_rewards, episode_successful_tasks, episode_unsuccessful_tasks)):
            csv_writer.writerow([episode, reward, successful, unsuccessful])

    # Plotting reward
    plt.plot(normalized_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.title("Average Episode Rewards")
    plt.savefig('episode_rewards_VTO_DDQN_v5.png')
    plt.show()

    # Plotting successful and unsuccessful tasks
    plt.plot(episode_successful_tasks, label='Successful Tasks')
    plt.plot(episode_unsuccessful_tasks, label='Unsuccessful Tasks')
    plt.xlabel("Episode")
    plt.ylabel("Number of Tasks")
    plt.title("Successful and Unsuccessful Tasks")
    plt.legend()
    plt.savefig('Successful_and_Unsuccessful_Tasks_VTO_DDQN_v5.png')
    plt.show()

    # Plot cumulative cost graph at the end of each episode
    plt.plot(episode_cumulative_costs, label='Episode Cumulative Cost')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Cost")
    plt.title("Episode Cumulative Costs")
    plt.legend()
    plt.savefig('episode_cumulative_cost_VTO_DDQN_v5.png')
    plt.show()

    # Plot validation rewards
    plt.plot(validation_rewards, label='Validation Reward')
    plt.xlabel("Validation Interval")
    plt.ylabel("Average Validation Reward")
    plt.title("Average Validation Rewards")
    plt.legend()
    plt.savefig('Average Validation Rewards_VTO_DDQN_v5.png')
    plt.show()

    # After all training is complete
    wandb.log({"Reward Graph": plt})
    wandb.finish()