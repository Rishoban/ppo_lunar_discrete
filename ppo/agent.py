import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network import CriticNetwork, ActorNetwork
from torchrl.data.replay_buffers import ReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torch.distributions import Categorical
from tensordict import TensorDict


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.GAMMA = 0.99          # Discount factor
        self.LAMBDA = 0.95         # GAE parameter

        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(
                        storage=LazyMemmapStorage(max_size),
                        sampler=SamplerWithoutReplacement(), batch_size=batch_size)

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                 n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                  n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                     n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                     n_actions=n_actions, name='target_critic')


    def choose_action(self, observation):
        self.actor.eval()
        observation = np.array([observation])
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        dist = Categorical(mu)
        action = dist.sample()
        log_prob = dist.log_prob(action).detach().cpu().numpy()[0]
        self.actor.train()

        # self.critic.eval()
        # state = T.tensor([observation], dtype=T.float).to(self.critic.device)
        # critic_value = self.critic.forward(state).to(self.critic.device)
        # value = T.flatten(critic_value).detach().cpu().numpy()
        # self.critic.train()

        return action.item(), log_prob.item()

    def remember(self, state, action, reward, next_state, done, old_log_prob):
        data = TensorDict(
                        {
                            "observation": T.tensor(state, dtype=T.float32).unsqueeze(0),  # Example observation tensor
                            "action":  T.tensor([action], dtype=T.float32),  # Example action tensor
                            "reward": T.tensor([reward], dtype=T.float32),  # Example reward tensor
                            "next_observation": T.tensor(next_state, dtype=T.float32).unsqueeze(0),  # Example next observation tensor
                            "done": T.tensor([done], dtype=bool),  # Example done flag tensor
                            "old_log_prob":  T.tensor([old_log_prob], dtype=T.float32),
                        },
                        batch_size=[1],  # Specify batch size
                    )
        self.memory.add(data)

    def compute_total_loss(self):
        batch = self.memory.sample()
        epsilon=0.2
        entropy_coeff=0.01
        value_coeff=0.5
        reward_batch = batch["reward"]
        returns = compute_discounted_return(reward_batch)

        old_log_probs = batch["old_log_prob"]
        dones = batch["done"]
        states = batch["observation"]
        actions = batch["action"]

        values = self.critic(states)
        advantages = advantages = T.FloatTensor(compute_gae(reward_batch, values, dones, self.GAMMA, self.LAMBDA))

        # 1. Policy Loss (Actor)
        dist = T.distributions.Categorical(self.actor(states))
        log_probs = dist.log_prob(actions)
        ratios = T.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = T.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -T.min(surr1, surr2).mean()

        # 2. Value Loss (Critic)
        value_loss = nn.MSELoss()(values.squeeze(-1), returns)

        # Optimize Actor
        self.actor.optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)  # Retain graph for critic update
        self.actor.optimizer.step()

        # Optimize Critic
        self.critic.optimizer.zero_grad()
        value_loss.backward()
        self.critic.optimizer.step()



# Helper function for advantage calculation using PyTorch tensors
def compute_gae(rewards, values, dones, gamma, lam):
    # Ensure tensors are correctly shaped
    rewards = rewards.view(-1)  # Flatten if necessary
    values = values.view(-1)
    dones = dones.view(-1).float()
    
    len_rewards = len(rewards)  # Number of time steps
    advantages = T.zeros(len_rewards, device=rewards.device)
    gae = 0  # Initialize GAE
    
    # Iterate in reverse to calculate advantages
    for t in reversed(range(len_rewards-1)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae  # Assign to tensor
    
    return advantages



def compute_discounted_return(rewards, gamma=0.99):
    """
    Calculates the discounted return for a sequence of rewards.

    Args:
        rewards: A tensor of rewards.
        gamma: Discount factor.

    Returns:
        A tensor of discounted returns.
    """
    discounted_return = T.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted_return[t] = running_add
    return discounted_return


