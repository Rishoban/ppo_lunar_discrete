import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

if __name__ == '__main__':
    env = gymnasium.make("LunarLanderContinuous-v3", render_mode="human")
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape, tau=0.001, batch_size=64,
                  fc1_dims=2048, fc2_dims=1536, n_actions=env.action_space.shape[0])

    n_games = 1000
    scores = []
    for i in range(n_games):
        done=False
        observation, _ = env.reset(seed=42)
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            reaction = env.step(action)
            observation_, reward, done, _, info = reaction
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print('episode_', i, " score %.2f" % score, 'average score %.2f' % avg_score)
