import gymnasium as gym
import torch
import argparse
import os
import yaml
from dqn import DQN
import flappy_bird_gymnasium
from custom_flappy_env import CustomFlappyBirdEnv

# Directory for loading models
RUNS_DIR = "runs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play(model_number):
    # Load play configurations
    with open('play_config.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # Initialize environment
    env = CustomFlappyBirdEnv(render_mode="human")    
    # Initialize DQN
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    policy_dqn = DQN(num_states, num_actions, config['fc1_nodes']).to(device)
    
    # Load trained model
    model_path = os.path.join(RUNS_DIR, f'flappybird{model_number}.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
        
    policy_dqn.load_state_dict(torch.load(model_path))
    policy_dqn.eval()  # Set to evaluation mode
    
    print(f"Loaded model from {model_path}")
    print("Starting game...")
    
    while True:  # Keep playing until user quits
        state, info = env.reset()
        total_reward = 0
        score = 0  # 실제 게임 스코어 추가
        done = False
        
        while not done:
            # Convert state to tensor
            state = torch.tensor(state, dtype=torch.float32, device=device)
            
            # Get action from policy
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax().item()
            
            # Take action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # 실제 게임 스코어 업데이트 (info에서 score 가져오기)
            score = info.get('score', 0)
        
        print(f"Game Over!")
        print(f"Score: {score}")  # 실제 게임 스코어
        print(f"Reward: {total_reward:.2f}")  # 에이전트가 받은 총 reward
        print("-" * 30)
        
        # Ask if user wants to play again
        play_again = input("Play again? (y/n): ").lower()
        if play_again != 'y':
            break
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play FlappyBird with a trained model')
    parser.add_argument('model_number', type=int, help='Model number to load (e.g., 1 for flappybird1.pt)')
    args = parser.parse_args()
    
    play(args.model_number)