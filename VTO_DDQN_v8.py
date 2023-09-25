#VTO with DoubleDQN v8. (with reward, successfull/unsuccessful tasks , and episode cost graph, extended code with validation)....>> bandwith + resource utlization + change the action state

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

class Task:
    def __init__(self, task_id, cpu_cores, data_size, memory_requirements, min_acceptable_delay):
        self.task_id = task_id
        self.cpu_cores = cpu_cores
        self.data_size = data_size
        self.memory_requirements = memory_requirements
        self.min_acceptable_delay = min_acceptable_delay

class Vehicle:
    def __init__(self, vehicle_id, cpu_cores, bandwidth, server_id):
        self.vehicle_id = vehicle_id
        self.cpu_cores = cpu_cores
        self.bandwidth = bandwidth
        self.server_id = server_id

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

def generate_tasks(num_tasks):
    tasks = []
    for i in range(num_tasks):
        data_size = random.randint(10, 100)
        memory_requirements = random.randint(10, 100)
        min_acceptable_delay = random.randint(5, 25) # 5 to 25 second delay
        cpu_cores = random.randint(15, 15)
        tasks.append(Task(i, cpu_cores, data_size, memory_requirements, min_acceptable_delay))
    return tasks

def create_resources(num_vehicles, num_rsus, num_cloud_servers):
    vehicles = [Vehicle(i, random.randint(10, 10), random.randint(100, 500), i) for i in range(num_vehicles)]
    road_side_units = [RoadSideUnit(i, random.randint(30, 50), random.randint(100, 500)) for i in range(num_rsus)]
    cloud_servers = [CloudServer(i, random.randint(500, 500), random.randint(1000, 2000)) for i in range(num_cloud_servers)]
    return vehicles, road_side_units, cloud_servers


# Custom Gym Environment
class TaskOffloadingEnv(gym.Env):
    def __init__(self, num_tasks, num_vehicles, num_rsus, num_cloud_servers):
        super(TaskOffloadingEnv, self).__init__()

        # Initialize your resources and tasks
        self.tasks = generate_tasks(num_tasks)
        self.vehicles, self.road_side_units, self.cloud_servers = create_resources(num_vehicles, num_rsus, num_cloud_servers)

        # Define the action space (0: Vehicle, 1: Road side units (RSU), 2: Cloud Server)
        #self.action_space = spaces.Discrete(3)

        self.action_space = spaces.Discrete(len(self.vehicles) + len(self.road_side_units) + len(self.cloud_servers))


        # Define the observation (state) space based on your scenario
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_tasks * 4 + num_vehicles * 5 + num_rsus * 5 + num_cloud_servers * 5,))

        # Initialize resource utilization dictionary
        self.resource_utilization = {
            'vehicles': [0] * num_vehicles,
            'road_side_units': [0] * num_rsus,
            'cloud_servers': [0] * num_cloud_servers
        }

        # Initialize current task index and resource availability
        self.current_task_idx = 0
        self.resource_availability = {
            'vehicles': [{'cores': vehicle.cpu_cores, 'delay': (2 + 1), 'cost': 0.15, 'distance': random.randint(100, 100), #(2 + 1) computation dealy +  nework latency
                          'bandwidth': random.randint(100, 500)}  # Assign bandwidth to vehicles
                         for vehicle in self.vehicles],
            'road_side_units': [
                {'cores': road_side_unit.cpu_cores, 'delay': (0.3 + 2), 'cost': 0.25, 'distance': random.randint(100, 100), #(0.3 + 2) computation dealy +  nework latency
                 'bandwidth': random.randint(100, 500)}  # Assign bandwidth to RSUs
                for road_side_unit in self.road_side_units],
            'cloud_servers': [
                {'cores': cloud_server.cpu_cores, 'delay': (0.1 + 3), 'cost': 0.35, 'distance':100,  #(0.1 + 3) computation dealy + nework latency
                 'bandwidth': random.randint(1000, 2000)}  # Assign bandwidth to Cloud Servers
                for cloud_server in self.cloud_servers]
        }

        # Initialize resource utilization at the beginning of each episode
        self.resource_utilization = {
            'vehicles': [0] * len(self.vehicles),
            'road_side_units': [0] * len(self.road_side_units),
            'cloud_servers': [0] * len(self.cloud_servers)
        }

    def reset(self):
        # Reset the environment to initial state and return the initial observation
        self.current_task_idx = 0
        self.resource_availability = {
            'vehicles': [{'cores': vehicle.cpu_cores, 'delay': (2 + 1), 'cost': 0.15, 'distance': random.randint(100, 100), #(2 + 1) computation dealy +  nework latency
                          'bandwidth': random.randint(100, 500)}  # Assign bandwidth to vehicles
                         for vehicle in self.vehicles],
            'road_side_units': [
                {'cores': road_side_unit.cpu_cores, 'delay': (0.3 + 2), 'cost': 0.25, 'distance': random.randint(100, 100), #(0.3 + 2) computation dealy +  nework latency
                 'bandwidth': random.randint(100, 500)}  # Assign bandwidth to RSUs
                for road_side_unit in self.road_side_units],
            'cloud_servers': [
                {'cores': cloud_server.cpu_cores, 'delay': (0.1 + 3), 'cost': 0.35, 'distance':100,  #(0.1 + 3) computation dealy + nework latency
                 'bandwidth': random.randint(1000, 2000)}  # Assign bandwidth to Cloud Servers
                for cloud_server in self.cloud_servers]
        }
       # Reset resource utilization at the beginning of each episode
        self.resource_utilization = {
            'vehicles': [0] * len(self.vehicles),
            'road_side_units': [0] * len(self.road_side_units),
            'cloud_servers': [0] * len(self.cloud_servers)
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

        # Determine the resource type and index based on the chosen action
        if action < len(self.vehicles):
            resource_type = 'vehicles'
            resource_idx = action
        elif action < len(self.vehicles) + len(self.road_side_units):
            resource_type = 'road_side_units'
            resource_idx = action - len(self.vehicles)
        else:
            resource_type = 'cloud_servers'
            resource_idx = action - len(self.vehicles) - len(self.road_side_units)

        processing_info = self._process_task(task, resource_type, resource_idx)

        if processing_info is not None:
            resource_id = processing_info['resource_id']
            if resource_type == 'vehicles':
                self.resource_utilization['vehicles'][resource_id] += task.cpu_cores
            elif resource_type == 'road_side_units':
                self.resource_utilization['road_side_units'][resource_id] += task.cpu_cores
            elif resource_type == 'cloud_servers':
                self.resource_utilization['cloud_servers'][resource_id] += task.cpu_cores

            reward = self._calculate_reward(task, processing_info, resource_type)
            task_assignment = {
                'task_id': task.task_id,
                'assigned_to': resource_type,
                #'resource_type': resource_type,
                'resource_id': processing_info['resource_id'],
                'remaining_resource_core':processing_info['remaining_reource_core']
            }

        else:
            reward = -1  # Set a large negative reward for unprocessable tasks
            task_assignment = {
                'task_id': task.task_id,
                'assigned_to': 'None',
                #'resource_type': 'None',
                'resource_id': 'None',
                'remaining_resource_core':'None'

            }

        self.current_task_idx += 1
        done = self.current_task_idx >= len(self.tasks)

        return self._get_state(), reward, processing_info, done, task_assignment

    def _process_task(self, task, resource_type, resource_idx):
      # Use the resource type and index to access the selected resource
      resource_list = self.resource_availability[resource_type]
      chosen_resource = resource_list[resource_idx]

      if chosen_resource:
          processing_delay = chosen_resource['delay']  # Simulated processing delay
          processing_cost = chosen_resource['cost']  # Simulated processing cost
          resource_success_reward = 1  # +1 for successful processing
          if task.cpu_cores <= chosen_resource['cores']:
            chosen_resource['cores'] -= task.cpu_cores
            resource_success_reward = 1  # +1 for successful processing
          else:
            resource_success_reward = -2  # +1 for successful processing
          resource_distance = chosen_resource['distance']
          resource_id = resource_list.index(chosen_resource)  # Use the index of the resource as resource_id
          return {
              'delay': processing_delay,
              'cost': processing_cost,
              'resource_success_reward': resource_success_reward,
              'distance': resource_distance,
              'resource_id': resource_id,
              'remaining_reource_core': chosen_resource['cores']  }
      return None

    def _calculate_reward(self, task, processing_info, resource_type):
        task_acceptable_delay = task.min_acceptable_delay
        resource_delay = processing_info['delay']
        resource_cost = processing_info['cost']
        resource_distance = processing_info['distance']
        resource_success_reward = processing_info['resource_success_reward']
        remaining_reource_core = processing_info['remaining_reource_core']

        delay_factor = (task_acceptable_delay - resource_delay) / task_acceptable_delay
        cost_factor = -resource_cost
        distance_factor = 1 - (resource_distance / 100)  # Normalize distance to a factor
        remaining_reource_core_factor = -2 if processing_info['remaining_reource_core'] < 0 else (2 if processing_info['remaining_reource_core'] == 0 else 1)

        reward = delay_factor + cost_factor + distance_factor + resource_success_reward + remaining_reource_core_factor

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
                print("q_value", q_values)
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
        #env.reset_resource_utilization()
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
    episode_losses = []
    validation_rewards = []
    validation_reward = 0
    episode_cumulative_costs = []
    episode_successful_tasks = []
    episode_unsuccessful_tasks = []
    vehicle_utilization = []  # List to store vehicle utilization across episodes
    rsu_utilization = []  # List to store RSU utilization across episodes
    cloud_server_utilization = []  # List to store cloud server utilization across episodes

    replay_buffer = ReplayBuffer(max_size=100000)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_successful = 0
        episode_unsuccessful = 0
        episode_cumulative_cost = 0
        episode_task_assignments = []  # List to store task assignments in this episode
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            print("action", action)
            next_state, reward, processing_info, done, task_assignment = env.step(action)

            episode_reward += reward

            # print(action)
            # print(task_assignment) # If want to see task assigment details in evry steps
            # print(next_state)
            # print(reward)
            # print("----------------------------------------------------------------------------")

            if reward > 0:
                episode_successful += 1
                episode_cumulative_cost += processing_info['cost']

            elif reward < 0:
                episode_unsuccessful += 1

            replay_buffer.add(state, action, reward, next_state, done)
            agent.update(replay_buffer, batch_size)
            state = next_state

            episode_task_assignments.append(task_assignment)  # Store task assignment information

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        episode_rewards.append(episode_reward)
        episode_cumulative_costs.append(episode_cumulative_cost)
        episode_successful_tasks.append(episode_successful)
        episode_unsuccessful_tasks.append(episode_unsuccessful)

        if (episode) % validation_interval == 0:
            validation_reward = validate_agent(env, agent, num_validation_episodes)
            validation_rewards.append(validation_reward)
            print("##############################################################")
            print(f"Episode {episode}: Validation Reward = {validation_reward:.2f}")
            print("##############################################################")
            wandb.log({ "Validation_reward": np.mean(validation_reward)}),

        print("-----------------------------------------------------------------------------------------------------------------------")
        print(f"Episode {episode}: Total Reward = {round(episode_reward, 2)}, Successful Tasks = {episode_successful}, Unsuccessful Tasks = {episode_unsuccessful}, Episode Cost = {round(episode_cumulative_cost,2)}, Epsilon={round(epsilon, 4)}")

        # Collect resource utilization data
        vehicle_utilization.append(env.resource_utilization['vehicles'])
        rsu_utilization.append(env.resource_utilization['road_side_units'])
        cloud_server_utilization.append(env.resource_utilization['cloud_servers'])

        # # # Print resource utilization at the end of the episode
        # print_resource_utilization(vehicle_utilization[episode], "Vehicle")
        # print_resource_utilization(rsu_utilization[episode], "RSU")
        # print_resource_utilization(cloud_server_utilization[episode], "Cloud Server")

        #Log summary statistics for this episode
        wandb.log({
            "Mean Episode Reward": np.mean(episode_rewards),
            "vehicle_Resource_utilization": np.mean(vehicle_utilization),
            "RSU_Resource_utilization": np.mean(rsu_utilization),
            "Cloud_Resource_utilization": np.mean(cloud_server_utilization),
            "Mean Successful Tasks": np.mean(episode_successful_tasks),
            "Mean Unsuccessful Tasks": np.mean(episode_unsuccessful_tasks),
            "Mean Episode Cumulative Cost": np.mean(episode_cumulative_costs),
            "Epsilon": epsilon
        })

    return episode_rewards, validation_rewards, episode_cumulative_costs, episode_successful_tasks, episode_unsuccessful_tasks, vehicle_utilization, rsu_utilization, cloud_server_utilization

def print_resource_utilization(resource_utilization, resource_type):
    print(f"{resource_type} Utilization:")
    for idx, utilization in enumerate(resource_utilization):
        print(f"{resource_type} {idx + 1}: {utilization}")

# Resource utilization plot function to normalize utilization between 1% and 100%
def plot_resource_utilization(utilization_data, resource_name):
    avg_utilization = np.mean(utilization_data, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(avg_utilization)), avg_utilization)
    plt.xticks(np.arange(len(avg_utilization)), np.arange(1, len(avg_utilization) + 1))
    plt.xlabel("Resource ID")
    plt.ylabel(f"{resource_name} Utilization")
    plt.title(f"{resource_name} Utilization Across Episodes")
    plt.tight_layout()  # Improve spacing
   #plt.savefig(f'figures/resources_u1tilization Across Episodes_VTO_DDQN_v7_{resource_name}_{num_episodes}_{num_tasks}.png')
    plt.show()

if __name__ == "__main__":
    num_tasks = 100
    num_vehicles = 100
    num_rsus = 1  # rsus (road side units)
    num_cloud_servers = 1

    wandb.login(key="31e5f2e0e26bf315d832e7fa7185b9ddd59adc32")

    wandb.init(project="VTO v3", config={
         "num_tasks": num_tasks,
         "num_vehicles": num_vehicles,
         "num_rsus": num_rsus,
         "num_cloud_servers": num_cloud_servers
    })

    env = TaskOffloadingEnv(num_tasks, num_vehicles, num_rsus, num_cloud_servers)

    state_dim = env.observation_space.shape[0]
    #print(state_dim)
    action_dim = env.action_space.n
    agent = DoubleDQNAgent(state_dim, action_dim)

    epsilon_decay = 0.995
    batch_size = 32
    num_episodes = 100

    validation_interval = 20  # Validate the agent every 200 episodes
    num_validation_episodes = 50  # Number of episodes for validation

    episode_rewards, validation_rewards, episode_cumulative_costs, episode_successful_tasks, episode_unsuccessful_tasks, vehicle_utilization, rsu_utilization, cloud_server_utilization = train_double_dqn(env, agent, num_episodes, batch_size, epsilon_decay, validation_interval, num_validation_episodes)

    # # Saving episode rewards, successful tasks, and unsuccessful tasks to a CSV file
    # with open('figures/episode_results.csv', 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(['Episode', 'Reward', 'Successful Tasks', 'Unsuccessful Tasks'])
    #     for episode, (reward, successful, unsuccessful) in enumerate(zip(episode_rewards, episode_successful_tasks, episode_unsuccessful_tasks)):
    #         csv_writer.writerow([episode, reward, successful, unsuccessful])

    max_reward = max(episode_rewards)
    normalized_episode_rewards = [reward / max_reward for reward in episode_rewards]

    # Plotting episode reward
    plt.plot(normalized_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.title("Average Episode Rewards")
    #plt.savefig(f'figures/episode_rewards_VTO_DDQN_v7_{num_episodes}_{num_tasks}.png')
    plt.show()

    # Plotting successful and unsuccessful tasks
    plt.plot(episode_successful_tasks, label='Successful Tasks')
    plt.plot(episode_unsuccessful_tasks, label='Unsuccessful Tasks')
    plt.xlabel("Episode")
    plt.ylabel("Number of Tasks")
    plt.title("Successful and Unsuccessful Tasks")
    plt.legend()
    #plt.savefig(f'figures/Successful_and_Unsuccessful_Tasks_VTO_DDQN_v7_{num_episodes}_{num_tasks}.png')
    plt.show()

    # Plot cumulative cost graph at the end of each episode
    plt.plot(episode_cumulative_costs, label='Episode Cumulative Cost')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Cost")
    plt.title("Episode Cumulative Costs")
    plt.legend()
    #plt.savefig(f'figures/episode_cumulative_cost_VTO_DDQN_v7_{num_episodes}_{num_tasks}.png')
    plt.show()

    # Plot validation rewards
    plt.plot(validation_rewards, label='Validation Reward')
    plt.xlabel("Validation Interval")
    plt.ylabel("Average Validation Reward")
    plt.title("Average Validation Rewards")
    plt.legend()
    #plt.savefig(f'figures/Average Validation Rewards_VTO_DDQN_v7_{num_episodes}_{num_tasks}.png')
    plt.show()

    # Plot normalized resource utilization between 1% and 100% for each type of resource using bar plots
    plot_resource_utilization(vehicle_utilization, "Vehicle")
    plot_resource_utilization(rsu_utilization, "RSU")
    plot_resource_utilization(cloud_server_utilization, "Cloud Server")

    # Calculate mean utilization values for each resource
    mean_vehicle_utilization = np.mean(vehicle_utilization, axis=1)
    mean_rsu_utilization = np.mean(rsu_utilization, axis=1)
    mean_cloud_server_utilization = np.mean(cloud_server_utilization, axis=1)

    x = [0, 1, 2]
    y = [mean_vehicle_utilization[0], mean_rsu_utilization[0],
         mean_cloud_server_utilization[0]]  # Mean utilization values
    labels = ['Vehicles', 'RSUs', 'Cloud Servers']

    plt.bar(x, y, tick_label=labels)
    plt.xlabel('Resource')
    plt.ylabel('Mean Utilization')
    plt.title('Mean Resource Utilization, Vehicles, RSUs, and CLoud Servers')
    plt.tight_layout()
    #plt.savefig(f'figures/Mean Resource Utilization_VTO_DDQN_v7_{num_episodes}_{num_tasks}.png')
    plt.show()
