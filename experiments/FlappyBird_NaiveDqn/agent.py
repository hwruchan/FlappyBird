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
        with open('C:/Users/jeongchan/iCloudDrive/iCloud~md~obsidian/obsidian/SMU/2024-2/강화학습/project/FlappyBird_NaiveDqn/hyperparameters.yml', 'r') as file:
            try:
                all_hyperparameter_sets = yaml.safe_load(file)
                hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            except UnicodeDecodeError:
                file.close()
                file = open('C:/Users/jeongchan/iCloudDrive/iCloud~md~obsidian/obsidian/SMU/2024-2/강화학습/project/FlappyBird_NaiveDqn/hyperparameters.yml', 'r', encoding='utf-8')
                all_hyperparameter_sets = yaml.safe_load(file)
                hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            #print(hyperparameters)
            
        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        # self.network_sync_rate  = hyperparameters['network_sync_rate']    # 주석 처리
        # self.replay_memory_size = hyperparameters['replay_memory_size']   # 주석 처리
        # self.mini_batch_size    = hyperparameters['mini_batch_size']      # 주석 처리
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']        # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']     # double dqn on/off flag

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

        env = gym.make(self.env_id, render_mode="human" if render else None, use_lidar=False)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)


        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            # memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            # target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            # target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            # step_count=0

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

                if is_training:
                    # Replay Memory 대신 즉시 학습
                    new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                    reward = torch.tensor(reward, dtype=torch.float, device=device)
                    
                    # 현재 상태에서의 Q값 계산
                    current_q = policy_dqn(state.unsqueeze(0)).squeeze(0)
                    
                    # 다음 상태에서의 최대 Q값 계산 (Target Network 없이)
                    with torch.no_grad():
                        next_q = policy_dqn(new_state.unsqueeze(0)).squeeze(0).max()
                    
                    # TD Target 계산
                    target_q = current_q.clone()
                    target_q[action] = reward + (1-terminated) * self.discount_factor_g * next_q
                    
                    # 손실 계산 및 업데이트
                    loss = self.loss_fn(current_q.unsqueeze(0), target_q.unsqueeze(0))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Epsilon decay
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

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
