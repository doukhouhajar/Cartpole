
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class StandardDQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(StandardDQNetwork, self).__init__()
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


class DuelingDQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DuelingDQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

class BaseAgent:
    def __init__(self, gamma=0.99, epsilon=1.0, lr=0.001, input_dims=4,
                 batch_size=64, n_actions=2, max_mem_size=100000,
                 eps_end=0.01, eps_dec=0.995):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
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


class DQNAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Q_eval = StandardDQNetwork(self.lr, kwargs['input_dims'], 
                                       fc1_dims=128, fc2_dims=128, 
                                       n_actions=kwargs['n_actions'])

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
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min


class DoubleDQNAgent(BaseAgent):
    def __init__(self, replace_target=100, **kwargs):
        super().__init__(**kwargs)
        self.replace_target_cnt = replace_target
        self.learn_step_counter = 0
        
        self.Q_eval = StandardDQNetwork(self.lr, kwargs['input_dims'], 
                                       fc1_dims=128, fc2_dims=128, 
                                       n_actions=kwargs['n_actions'])
        self.Q_target = StandardDQNetwork(self.lr, kwargs['input_dims'], 
                                         fc1_dims=128, fc2_dims=128, 
                                         n_actions=kwargs['n_actions'])

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
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next_eval = self.Q_eval.forward(new_state_batch)
        max_actions = T.argmax(q_next_eval, dim=1)
        q_next_target = self.Q_target.forward(new_state_batch)
        q_next_target[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next_target[batch_index, max_actions]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min


class DuelingDQNAgent(BaseAgent):
    def __init__(self, replace_target=100, **kwargs):
        super().__init__(**kwargs)
        self.replace_target_cnt = replace_target
        self.learn_step_counter = 0
        
        self.Q_eval = DuelingDQNetwork(self.lr, kwargs['input_dims'], 
                                      fc1_dims=128, fc2_dims=128, 
                                      n_actions=kwargs['n_actions'])
        self.Q_target = DuelingDQNetwork(self.lr, kwargs['input_dims'], 
                                        fc1_dims=128, fc2_dims=128, 
                                        n_actions=kwargs['n_actions'])

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
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next_eval = self.Q_eval.forward(new_state_batch)
        max_actions = T.argmax(q_next_eval, dim=1)
        q_next_target = self.Q_target.forward(new_state_batch)
        q_next_target[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next_target[batch_index, max_actions]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

def train_agent(agent, env, episodes=500, agent_name="Agent"):
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
        
        if i % 50 == 0:
            print(f'{agent_name} - Episode {i}, Score: {score:.2f}, Avg: {avg_score:.2f}, ε: {agent.epsilon:.3f}')
    
    return scores, eps_history


def compare_algorithms(episodes=500):
    print("COMPARING DQN VARIANTS ON CARTPOLE")
    
    config = {
        'gamma': 0.99,
        'epsilon': 1.0,
        'lr': 0.001,
        'input_dims': 4,
        'batch_size': 64,
        'n_actions': 2,
        'eps_dec': 0.995,
        'eps_end': 0.01
    }
    
    results = {}
    
    print("\nStandard DQN")
    env = gym.make('CartPole-v1')
    agent_dqn = DQNAgent(**config)
    scores_dqn, eps_dqn = train_agent(agent_dqn, env, episodes, "DQN")
    results['DQN'] = {'scores': scores_dqn, 'eps': eps_dqn}
    
    print("\nDouble DQN")
    env = gym.make('CartPole-v1')
    agent_double = DoubleDQNAgent(replace_target=100, **config)
    scores_double, eps_double = train_agent(agent_double, env, episodes, "Double DQN")
    results['Double DQN'] = {'scores': scores_double, 'eps': eps_double}
    
    print("\nDueling DQN")
    env = gym.make('CartPole-v1')
    agent_dueling = DuelingDQNAgent(replace_target=100, **config)
    scores_dueling, eps_dueling = train_agent(agent_dueling, env, episodes, "Dueling DQN")
    results['Dueling DQN'] = {'scores': scores_dueling, 'eps': eps_dueling}
    
    plot_comparison(results, episodes)
    

    print("FINAL RESULTS")
    for name, data in results.items():
        avg_score = np.mean(data['scores'][-100:])
        max_score = np.max(data['scores'][-100:])
        print(f"{name:15s}: Avg = {avg_score:6.2f}, Max = {max_score:6.2f}")
    
    return results


def plot_comparison(results, episodes):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = {'DQN': 'blue', 'Double DQN': 'green', 'Dueling DQN': 'red'}
    
    # Raw scores
    ax = axes[0, 0]
    for name, data in results.items():
        ax.plot(data['scores'], label=name, color=colors[name], alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Raw Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Moving average 
    ax = axes[0, 1]
    for name, data in results.items():
        scores = data['scores']
        moving_avg = [np.mean(scores[max(0, i-19):i+1]) for i in range(len(scores))]
        ax.plot(moving_avg, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Score')
    ax.set_title('Moving Average')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Moving average
    ax = axes[1, 0]
    for name, data in results.items():
        scores = data['scores']
        moving_avg = [np.mean(scores[max(0, i-99):i+1]) for i in range(len(scores))]
        ax.plot(moving_avg, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Score')
    ax.set_title('Moving Average')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final comparison
    ax = axes[1, 1]
    names = list(results.keys())
    final_scores = [np.mean(results[name]['scores'][-100:]) for name in names]
    bars = ax.bar(names, final_scores, color=[colors[name] for name in names])
    ax.set_ylabel('Average Score')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dqn_comparison.png', dpi=150)


if __name__ == '__main__':
    results = compare_algorithms(episodes=500)
