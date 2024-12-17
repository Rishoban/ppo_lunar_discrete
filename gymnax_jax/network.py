import haiku as hk
import jax

class ActorCriticNetwork(hk.Module):
    def __init__(self, input_dim: int, fc1_dims: int, fc2_dims: int, n_actions: int):
        super().__init__()
        self.input_dim = input_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

    def __call__(self, x):
        x = hk.Linear(self.fc1_dims)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.fc2_dims)(x)
        x = jax.nn.relu(x)

        pi = hk.Linear(self.n_actions)(x)
        v = hk.Linear(1)(x)

        return pi, v