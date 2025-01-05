import gymnasium as gym
from agent import Agent
import numpy as np
import torch

env = gym.make('CartPole-v1')


agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape, tau=0.001, batch_size=64,
                  fc1_dims=2048, fc2_dims=1536, n_actions=2)


no_episode = 4
eps = 0.5
for i in range(no_episode):

    episode_over = False
    observation, info = env.reset()
    while not episode_over:
        if np.random.random()  < eps: 
            action = env.action_space.sample()  # agent policy that uses the observation and info
            log_prob = 0.5
            
        else:
            action, log_prob = agent.choose_action(observation)
           

        observation_, reward, terminated, truncated, info = env.step(action)
    
        episode_over = terminated or truncated
        agent.remember(observation, action, reward, observation_, truncated, log_prob)
        observation = observation_

    agent.compute_total_loss()

env.close()
