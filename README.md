# DQN Variants for CartPole

All implementations use **Gymnasium** (the maintained fork of OpenAI Gym).

## Files Overview

### Main Implementations
1. **dqn_cartpole.py** - Standard DQN implementation
2. **double_dqn_cartpole.py** - Double DQN implementation  
3. **dueling_dqn_cartpole.py** - Dueling DQN implementation
4. **compare_all_dqn.py** - Compare all three algorithms

---

## Quick Start

### Option 1: Train Individual Algorithms
```bash
# Standard DQN
python dqn_cartpole.py

# Double DQN
python double_dqn_cartpole.py

# Dueling DQN
python dueling_dqn_cartpole.py
```

### Option 2: Compare All Three
```bash
python compare_all_dqn.py
```
This trains all three and creates comparison plots.

---

## Algorithm Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    STANDARD DQN                                 │
│                                                                 │
│  Input → Hidden(128) → Hidden(128) → Output(2) → Q-values      │
│  [s₀, s₁, s₂, s₃]                           [Q(left), Q(right)]│
│                                                                 │
│  Update: Q_target = r + γ × max Q(s', a')                      │
│                                                                 │
│  ✓ Simple, easy to understand                                  │
│  ✗ Can overestimate Q-values                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DOUBLE DQN                                   │
│                                                                 │
│  Q_eval:  Input → Hidden → Hidden → Output → Q-values          │
│  Q_target: Input → Hidden → Hidden → Output → Q-values         │
│                                                                 │
│  Update: Q_target = r + γ × Q_target(s', argmax Q_eval(s'))   │
│           ↑ evaluate with target    ↑ select with eval         │
│                                                                 │
│  ✓ Reduces overestimation                                      │
│  ✓ More stable learning                                        │
│  ✓ Industry standard                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DUELING DQN                                  │
│                                                                 │
│         Input → Hidden(128) → Hidden(128)                       │
│                                  ├─→ V(s) [1 value]            │
│                                  └─→ A(s,a) [2 values]         │
│                                                                 │
│  Combine: Q(s,a) = V(s) + [A(s,a) - mean(A)]                   │
│                                                                 │
│  ✓ Learns state value separately from action advantages        │
│  ✓ Faster convergence                                          │
│  ✓ Better generalization                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Simple Explanations

### Standard DQN
> "I'll learn Q-values for each action using one neural network"
- Like having one teacher grade all your work

### Double DQN  
> "I'll use one network to pick actions, another to evaluate them"
- Like having one teacher suggest answers, another grade them
- Prevents being overconfident

### Dueling DQN
> "I'll learn how good each STATE is, plus how much better each ACTION is"
- Splits learning into: "Is this a good situation?" + "Which action is best?"
- Learns faster because state quality is independent of actions

---

## Hyperparameters

```python
# Good defaults for CartPole
gamma = 0.99           # Discount factor
epsilon = 1.0          # Start with 100% exploration
epsilon_min = 0.01     # End with 1% exploration
epsilon_decay = 0.995  # Decay rate per episode
learning_rate = 0.001  # Network learning rate
batch_size = 64        # Experiences per update
memory_size = 100000   # Replay buffer size
hidden_layers = [128, 128]  # Network architecture
```



## 📝 Requirements

```bash
pip install torch numpy gymnasium matplotlib
```

**Note:** We use `gymnasium` (not `gym`). 

For additional environments:
```bash
# For Box2D environments (LunarLander)
pip install gymnasium[box2d]

# For Atari games
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```

