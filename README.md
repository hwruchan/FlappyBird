# Flappy Bird DQN Project  
---

## **Project Aim**  
This project explores the use of Deep Q-Learning Networks (DQN) to train an AI agent to play Flappy Bird efficiently. Key objectives include:  
1. Comparing learning efficiency across different numbers of episodes.  
2. Evaluating the impact of hidden layer count (1 vs. 2).  
3. Testing the influence of hyperparameters such as the discount factor.  
4. Comparing the performance of Naïve DQN with enhanced DQN.  
5. Experimenting with a custom environment.  

---

## **Game Overview**  
- **Gameplay**: Tap (or click) to make the bird fly; do nothing, and it falls. Survive as long as possible while passing through pipes.  
- **State Representation (Gymnasium)**:  
    - Last and next pipe positions (x, y) and the following pipe positions.  
    - Player's vertical position, velocity, and rotation.  
- **Actions**:  
    - Jump or do nothing.  
- **Reward Structure**:  
    - Passing a pipe: +1  
    - Dying: -1.0  
    - Touching the screen top: -0.5  
    - Staying alive (per frame): +0.1  

---

## **Key Components**  
### 1. Experience Replay  
Randomly samples mini-batches from stored experiences to reduce correlation between consecutive experiences and ensure a diverse training set.  

### 2. Target Network  
A secondary network is periodically updated to stabilize learning by fixing target predictions for a while.  

### 3. Learning Algorithm  
- **Loss Function**: Mean Squared Error (MSE).  
- **Training Steps**:  
  - `zero_grad()`: Clears previous gradients.  
  - `backward()`: Computes gradients based on loss.  
  - `step()`: Updates weights using computed gradients.  

---

## **Experimentation and Results**  

### 1. Learning Amount  
| Episodes      | Max Reward | Time Taken |  
|---------------|------------|------------|  
| 10,000        | 6.0        | 5m         |  
| 100,000       | 50         | 46m        |  
| 500,000       | 80         | 3h         |  

### 2. Discount Factor  
| Discount Factor | Episodes | Best Reward | Time Taken |  
|-----------------|----------|-------------|------------|  
| 0.99            | 100,000  | 59.9        | 46m        |  
| 0.90            | 100,000  | 82.9        | 1h 42m     |  

### 3. Hidden Layers  
| Hidden Layers | Episodes | Best Reward | Time Taken |  
|---------------|----------|-------------|------------|  
| 1             | 500,000  | 80.6        | 3h         |  
| 2             | 250,000  | 100.1       | 1h 36m     |  

### 4. Naïve DQN vs DQN  
| Model       | Episodes | Best Reward | Time Taken |  
|-------------|----------|-------------|------------|  
| Original DQN| 100,000  | 59.9        | 46m        |  
| Naïve DQN   | 100,000  | 50.6        | 54m        |  

### 5. Custom Environment  
| Configuration      | Episodes | Best Reward | Time Taken   |  
|--------------------|----------|-------------|--------------|  
| Custom Flappybird  | 500,000  | 71.8        | 18h 30m      |  

---

## **Discussion**  
- Combining **2 hidden layers** and **discount factor of 0.90** improves performance but risks overfitting to short-term rewards.  
- Optimized networks can learn complex patterns effectively but need careful tuning to balance short- and long-term rewards.  
- Further improvements involve grid search, custom environments, and advanced hyperparameter tuning.  

---
