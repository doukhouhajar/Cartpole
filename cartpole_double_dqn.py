import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
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


class DoubleDQNAgent:
    def __init__(self, gamma=0.99, epsilon=1.0, lr=0.001, input_dims=4,
                 batch_size=64, n_actions=2, max_mem_size=100000,
                 eps_end=0.01, eps_dec=0.995, replace_target=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.replace_target_cnt = replace_target  
        self.learn_step_counter = 0

        # TWO networks: main (Q_eval) and target (Q_target)
        self.Q_eval = DQNetwork(lr, input_dims, fc1_dims=128, fc2_dims=128, 
                               n_actions=n_actions)
        self.Q_target = DQNetwork(lr, input_dims, fc1_dims=128, fc2_dims=128, 
                                 n_actions=n_actions)
        

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

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
       
        self.replace_target_network()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        
        # Current Q-values
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        
        # DOUBLE DQN MAGIC HAPPENS HERE:
        # use Q_eval to SELECT the best action for next state
        q_next_eval = self.Q_eval.forward(new_state_batch)
        max_actions = T.argmax(q_next_eval, dim=1)
        
        # use Q_target to EVALUATE that action
        q_next_target = self.Q_target.forward(new_state_batch)
        q_next_target[terminal_batch] = 0.0
        
        # use the action selected by Q_eval but evaluated by Q_target
        q_target = reward_batch + self.gamma * q_next_target[batch_index, max_actions]
        
        # update network
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min


def train_double_dqn(episodes=500):
    env = gym.make('CartPole-v1')
    agent = DoubleDQNAgent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=4,
                          batch_size=64, n_actions=2, eps_dec=0.995, 
                          replace_target=100)
    
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
    scores, eps_history = train_double_dqn(episodes=500)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Double DQN: Scores over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(eps_history)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.tight_layout()
    plt.savefig('double_dqn_training.png')
    print("done!")
