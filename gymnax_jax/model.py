from network import ActorCriticNetwork
import haiku as hk
import optax
import jax
import jax.numpy as jnp
from jax.random import categorical

from jax.experimental import host_callback

def forward_fn(x):
    model = ActorCriticNetwork(input_dim=4, fc1_dims=2048, fc2_dims=1536, n_actions=3)  # 10 output classes (e.g., for classification)
    return model(x)


# Transform the forward function using Haiku
def create_model():
    return hk.transform(forward_fn)


def choose_action(model, observation, params, rng):
    logits, _ = model.apply(params, rng, observation)
    probabilities = jax.nn.softmax(logits)

    action = categorical(rng, logits=logits)
    # action = jnp.squeeze(action)  # Squeeze to remove unnecessary dimensions
    log_probs = jnp.log(probabilities[action])

    return action, log_probs


def loss_fn(params, model, rng, state, reward, state_, done, gamma, log_prob):
    """Compute actor-critic loss."""
    _, critic_value = model.apply(params, rng, state)
    _, critic_value_ = model.apply(params, rng, state_)

    delta = reward + gamma * critic_value_ * (1 - done.astype(int)) - critic_value
    actor_loss = -log_prob * delta
    critic_loss = delta ** 2

    loss = actor_loss + critic_loss
    return jnp.mean(loss)


def debug_tree(tree, name="Tree"):
    host_callback.id_print(tree, what=name)


def update(params, opt_state, optimizer, loss_params, model, rng):
    """Compute gradients and update the parameters."""
    state, reward, state_, done, gamma, log_prob = loss_params

    # Define a loss function that only takes params
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, model, rng, state, reward, state_, done, gamma, log_prob)

    # Apply gradients
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def learn(state, reward, state_, done, params, rng, model, gamma, log_prob, opt_state, optimizer):
    """Wrapper to perform a learning step."""
    loss_params = (state, reward, state_, done, gamma, log_prob)
    params, opt_state = update(params, opt_state, optimizer, loss_params, model, rng)
    return params, opt_state
