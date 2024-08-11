import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from network import Agent

if __name__ == '__main__':
    env = gymnasium.make("LunarLander-v2", render_mode="human")
    agent = Agent(lr=5e-6, input_dims=[8], n_actions=4, fc1_dims=2048, fc2_dims=1536, gamma=0.99)

    n_games = 3000
    scores = []
    for i in range(n_games):
        done=False
        observation, _ = env.reset(seed=42)
        score = 0
        while not done:
            action = agent.choose_action(observation)
            reaction = env.step(action)
            observation_, reward, done, _, info = reaction
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print('episode_', i, " score %.2f" % score, 'average score %.2f' % avg_score)
