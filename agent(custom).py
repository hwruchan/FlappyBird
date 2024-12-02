import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

from custom_flappy_env import CustomFlappyBirdEnv



# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, hyperparameter_set):
        with open("C:/Users/user/iCloudDrive/iCloud~md~obsidian/obsidian/SMU/2024-2/강화학습/project/FlappyBird3/hyperparameters.yml", 'r') as file:
            try:
                all_hyperparameter_sets = yaml.safe_load(file)
                hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            except UnicodeDecodeError:
                file.close()
                file = open("C:/Users/user/iCloudDrive/iCloud~md~obsidian/obsidian/SMU/2024-2/강화학습/project/FlappyBird3/hyperparameters.yml", 'r', encoding='utf-8')
                all_hyperparameter_sets = yaml.safe_load(file)
                hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            #print(hyperparameters)
            
        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False, max_episodes=None):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        #env = gym.make(self.env_id, render_mode="human" if render else None, use_lidar=False)
        env = CustomFlappyBirdEnv(render_mode="human" if render else None)
            
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)


        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999

        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()
            




        for episode in itertools.count():
            if is_training and episode % 10000 == 0:
                print(f"Episode {episode}")
                print(f"Epsilon: {epsilon:.4f}")
                if len(rewards_per_episode) > 0:
                    print(f"Average Reward: {np.mean(rewards_per_episode[-10000:]):.2f}")
                    print(f"Max Reward: {max(rewards_per_episode[-10000:]):.2f}")            
                print("-" * 30)

            if is_training and episode > 0 and (episode % 100000 == 0 or (max_episodes and episode == max_episodes)):
                model_number = (episode + 99999) // 100000  
                base_name = self.hyperparameter_set.split('1')[0]
                checkpoint_path = os.path.join(RUNS_DIR, f'{base_name}{model_number}.pt')
                torch.save(policy_dqn.state_dict(), checkpoint_path)
                print(f"Model saved at episode {episode}: {checkpoint_path}")

            if max_episodes and episode >= max_episodes:
                print(f"Reached maximum episodes: {max_episodes}")
                break

            state, _ = env.reset()
            
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while(not terminated and episode_reward < self.stop_on_reward):
                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else: # just evaluating state
                    with torch.no_grad():

                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze(dim=0).argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                # accumulate reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated))

                    # increment step count
                    step_count += 1

                # Update state(move to next state)
                state = new_state

            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)


        # Calculate target Q values (expected returns)
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
        # Compute loss for the whole mini-batch
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--max_episodes', type=int, default=None, help='Maximum number of episodes')
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        agent.run(is_training=True, max_episodes=args.max_episodes)
    else:
        agent.run(is_training=False, render=True)
