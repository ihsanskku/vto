# Version 2 (step and reward function are resdesigned )
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gym
from gym import spaces
import random
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
        task_complexity = round((cpu_cores - 1) / (4 - 1), 2)  # Generate a random task complexity between 0 and 1
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
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_tasks*3 + num_vehicles*4 + num_edge_servers*4 + num_cloud_servers*4,))

        # Initialize current task index and resource availability
        self.current_task_idx = 0
        self.resource_availability = {
            'vehicles': [{'cores': vehicle.cpu_cores, 'delay': 0.1, 'cost': 0.15, 'distance': random.randint(10, 100)} for vehicle in self.vehicles],
            'edge_servers': [{'cores': edge_server.cpu_cores, 'delay': 0.2, 'cost': 0.25, 'distance': random.randint(10, 100)} for edge_server in self.edge_servers],
            'cloud_servers': [{'cores': cloud_server.cpu_cores, 'delay': 0.3, 'cost': 0.35, 'distance': random.randint(10, 100)} for cloud_server in self.cloud_servers]
        }

    def reset(self):
        # Reset the environment to initial state and return the initial observation
        self.current_task_idx = 0
        self.resource_availability = {
            'vehicles': [{'cores': vehicle.cpu_cores, 'delay': 0.1, 'cost': 0.15, 'distance': random.randint(10, 100)} for vehicle in self.vehicles],
            'edge_servers': [{'cores': edge_server.cpu_cores, 'delay': 0.2, 'cost': 0.25, 'distance': random.randint(10, 100)} for edge_server in self.edge_servers],
            'cloud_servers': [{'cores': cloud_server.cpu_cores, 'delay': 0.3, 'cost': 0.35, 'distance': random.randint(10, 100)} for cloud_server in self.cloud_servers]
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
              reward = self._calculate_reward(task, processing_info,  resource_type)
          else:
              reward = -1  # Set a large negative reward for unprocessable tasks

          self.current_task_idx += 1
          done = self.current_task_idx >= len(self.tasks)

          return self._get_state(), reward, done, {}


    def _process_task(self, task, resource_type):
        resource_list = self.resource_availability[resource_type]
        for resource in resource_list:
            if task.cpu_cores <= resource['cores']:
                processing_delay = resource['delay']  # Simulated processing delay
                processing_cost = resource['cost']  # Simulated processing cost
                resource_success_reward = 1  # +1 for successful processing
                resource['cores'] -= task.cpu_cores
                resource_distance = resource['distance']
                return {'delay': processing_delay, 'cost': processing_cost, 'resource_success_reward': resource_success_reward, 'distance': resource_distance, }
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

        #normalized_reward = (reward + 4) / 8  # Normalize the reward between 0 and 1
        #reward = normalized_reward

        return reward


# A2C Network
class A2CNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(A2CNetwork, self).__init__()

        # Define your network architecture
        self.fc = nn.Linear(input_dim, 64)
        self.actor = nn.Linear(64, num_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

# Print and Display Table Function
def display_table(resources, resource_type):
    if resource_type == "Task":
        headers = ["Task", "CPU Cores", "Min Acceptable Delay", "Task complexity"]  # Updated headers
        rows = [[f"{resource_type} {idx+1}", task.cpu_cores, task.min_acceptable_delay, task.task_complexity] for idx, task in enumerate(resources)]
    else:
        headers = ["Resource", "Cores", "Delay", "Cost", "Distance"]
        rows = []
        for idx, resource in enumerate(resources):
            rows.append([f"{resource_type} {idx+1}", resource['cores'], resource['delay'], resource['cost'], resource['distance']])
    print(tabulate(rows, headers=headers, tablefmt="grid"))


# Training Loop
def train_a2c(env, num_episodes, gamma, lr):
    input_dim = env.observation_space.shape[0]
    #print(input_dim)
    num_actions = env.action_space.n
    a2c_net = A2CNetwork(input_dim, num_actions)
    optimizer = optim.Adam(a2c_net.parameters(), lr=lr)

    episode_rewards = []  # Track rewards for each episode

    for episode in range(num_episodes):
        state = env.reset()
        #print(state)
        print("=============================================================")
        done = False  # Initialize 'done' here
        episode_reward = 0
        log_probs = []
        values = []
        rewards = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs, value = a2c_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            #print(action_probs)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            #print("action>>>>>>>>>", action.item())
            next_state, reward, done, _ = env.step(action.item())
            #print("next_state",next_state,"/", reward, "/", done)
            episode_reward += reward

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Calculate advantages and update the A2C network
        R = 0
        discounted_rewards = []
        for reward in reversed(rewards):
            R = reward + gamma * R
            discounted_rewards.insert(0, R)

        advantages = torch.tensor(discounted_rewards) - torch.stack(values)
        actor_loss = torch.stack(log_probs) * advantages
        critic_loss = advantages.pow(2).mean()
        total_loss = -actor_loss.mean() + critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        episode_rewards.append(episode_reward)  # Store the episode reward
        print(f"Episode {episode}: Total Reward = {episode_reward}")


# Normalize episode rewards between 0 and 1
    max_reward = max(episode_rewards)
    normalized_episode_rewards = [reward / max_reward for reward in episode_rewards]
    episode_rewards = normalized_episode_rewards

# Plot average episode rewards
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.title("Average Episode Rewards")
    plt.savefig('episode_rewards_VTO_a2c_v2.png')
    plt.show()

# Main function
if __name__ == "__main__":

    num_tasks = 21
    num_vehicles = 1
    num_edge_servers = 5
    num_cloud_servers = 7

    # A2C Training Parameters
    gamma = 0.99  # Discount factor for future rewards
    lr = 0.01  # Learning rate for the A2C network
    num_episodes = 10  # Traning episodes

    env = TaskOffloadingEnv(num_tasks, num_vehicles, num_edge_servers, num_cloud_servers)

    # Display task and resource details before training
    print("Task Details:")
    display_table(env.tasks, "Task")
    print("\nVehicle Details:")
    display_table(env.resource_availability['vehicles'], "Vehicle")
    print("\nEdge Server Details:")
    display_table(env.resource_availability['edge_servers'], "Edge Server")
    print("\nCloud Server Details:")
    display_table(env.resource_availability['cloud_servers'], "Cloud Server")

    train_a2c(env, num_episodes, gamma, lr)
