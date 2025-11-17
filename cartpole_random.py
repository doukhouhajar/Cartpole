import gymnasium as gym
import random

env = gym.make("CartPole-v1")

def Random_games():
    for episode in range(10):
        state, info = env.reset()          
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(t, next_state, reward, done, info, action)
            if done:
                break

    env.close()

Random_games()
