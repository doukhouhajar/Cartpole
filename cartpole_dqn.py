import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt


class DQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)  
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)    
        self.fc3 = nn.Linear(fc2_dims, n_actions)   
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)  
        return actions


class DQNAgent:
    def __init__(self, gamma=0.99, epsilon=1.0, lr=0.001, input_dims=4,
                 batch_size=64, n_actions=2, max_mem_size=100000,
                 eps_end=0.01, eps_dec=0.995):
        self.gamma = gamma              # discount factor
        self.epsilon = epsilon          # exploration rate:
        self.eps_min = eps_end  
        self.eps_dec = eps_dec        
        self.lr = lr                  
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DQNetwork(lr, input_dims, fc1_dims=128, fc2_dims=128, 
                               n_actions=n_actions)
        
        # memory for storing experiences (state, action, reward, next_state, done)
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        
        # current Q-values for the actions that were taken
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        
        # Q-values for next states
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0  # No future reward if episode ended
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]   # reward + discounted max future Q-value
        
        # update network
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        # decay epsilon (reduce exploration over time)
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min


def train_dqn(episodes=500):
    env = gym.make('CartPole-v1')
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=4,
                     batch_size=64, n_actions=2, eps_dec=0.995)
    
    scores = []
    eps_history = []
    
    for i in range(episodes):
        score = 0
        done = False
        truncated = False
        observation, info = env.reset()  
        
        while not (done or truncated):
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        
        if i % 10 == 0:
            print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}')
    
    return scores, eps_history


if __name__ == '__main__':
    scores, eps_history = train_dqn(episodes=500)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(eps_history)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.tight_layout()
    plt.savefig('dqn_training.png')
    print("done!")