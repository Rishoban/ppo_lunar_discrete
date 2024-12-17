import jax
import jax.numpy as jnp
from flax import linen as nn
import gymnax
import optax
from flax import nnx
from model import create_model, choose_action, learn

rng = jax.random.PRNGKey(0)
rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

# Create the Pendulum-v1 environment
env, env_params = gymnax.make("CartPole-v1")
obs, state = env.reset(key_reset, env_params)

model = create_model()

gamma=0.99
optimizer = optax.adam(learning_rate=5e-6)



def rollout(rng_input, policy_params, env_params, opt_state, max_steps_in_episode):
    """Rollout a jitted gymnax episode with lax.while_loop."""
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset, env_params)

    def cond_fun(carry):
        """Condition for while loop: not done and steps < max_steps."""
        obs, state, step, done, _, reward, policy_params, opt_state = carry
        return jnp.logical_not(done) & (step < max_steps_in_episode)

    def body_fun(carry):
        """Step transition in jax env."""
        obs, state, step, done, rng, reward, policy_params, opt_state = carry
        rng, rng_step, rng_net = jax.random.split(rng, 3)

        # In here we need to choose the action
        action, log_probs = choose_action(model,obs, policy_params, rng_net)

        next_obs, next_state, reward, next_done, _ = env.step(
            rng_step, state, action, env_params
        )
        # We need to train the model
        policy_params, opt_state = learn(obs, reward, next_obs , done, policy_params, rng, model, gamma, log_probs, opt_state, optimizer)
        new_carry = [next_obs, next_state, step + 1, next_done, rng, reward, policy_params, opt_state]
        return new_carry

    initial_carry = [obs, state, 0, False, rng_episode, 0, policy_params, opt_state] # Add step counter and done flag
    final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
    obs, state, steps, done, _, reward, policy_params, opt_state = final_carry
    # return obs, steps, done
    return obs, steps, done, reward, policy_params, opt_state


def batched_rollout(rng_input, policy_params, env_params,opt_state, max_steps_in_episode, num_episodes):
    """Rollout multiple episodes in parallel using vmap."""
    rng_inputs = jax.random.split(rng_input, num_episodes)  # Split RNG for each episode
    batched_rollout_fn = jax.vmap(rollout, in_axes=(0, None, None,None, None)) # vmap over the rng_input

    obs, steps, done, reward, policy_params, opt_state = batched_rollout_fn(
        rng_inputs, policy_params, env_params, opt_state, max_steps_in_episode
    )
    return obs, steps, done, reward


# In here max_steps_in_episode, num_episodes are static arguments
jit_batched_rollout = jax.jit(batched_rollout, static_argnums=(4,5))
policy_params = model.init(rng, obs)
opt_state = optimizer.init(policy_params)
num_episodes = 10
max_steps_in_episode = 200

obs, steps, done, reward = jit_batched_rollout(rng, policy_params, env_params,opt_state, max_steps_in_episode, num_episodes)

print(obs.shape, steps.shape, done.shape, reward.shape)
print(reward)