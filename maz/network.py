"""SENet with SE-PRE residual blocks for multiplayer AlphaZero.

Architecture: input conv → 8× SE-PRE blocks → policy head + value head.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from maz.game import ROWS, COLS, NUM_PLAYERS, COLS as NUM_ACTIONS


NUM_FILTERS = 64
NUM_BLOCKS = 8
SE_RATIO = 4
HEAD_FILTERS = 32


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    channels: int
    ratio: int = SE_RATIO

    @nn.compact
    def __call__(self, x):
        # Squeeze: global average pool over spatial dims
        se = x.mean(axis=(-3, -2))  # (..., C)
        # Excitation
        se = nn.Dense(self.channels // self.ratio)(se)
        se = nn.relu(se)
        se = nn.Dense(self.channels)(se)
        se = nn.sigmoid(se)
        # Scale: broadcast over spatial dims
        return x * se[..., None, None, :]


class SEPreBlock(nn.Module):
    """SE-PRE residual block: BN→ReLU→Conv→BN→ReLU→Conv→SE→add."""
    channels: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        h = nn.BatchNorm(use_running_average=not train)(x)
        h = nn.relu(h)
        h = nn.Conv(self.channels, (3, 3), padding="SAME")(h)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.relu(h)
        h = nn.Conv(self.channels, (3, 3), padding="SAME")(h)
        h = SEBlock(self.channels)(h)
        return h + residual


class AlphaZeroNet(nn.Module):
    """Full AlphaZero network with SE-PRE blocks."""
    num_filters: int = NUM_FILTERS
    num_blocks: int = NUM_BLOCKS
    num_actions: int = NUM_ACTIONS
    num_players: int = NUM_PLAYERS

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (..., 6, 7, 6)
        # Input conv
        h = nn.Conv(self.num_filters, (3, 3), padding="SAME")(x)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.relu(h)

        # Residual tower
        for _ in range(self.num_blocks):
            h = SEPreBlock(self.num_filters)(h, train=train)

        # Policy head
        p = nn.Conv(HEAD_FILTERS, (1, 1))(h)
        p = nn.BatchNorm(use_running_average=not train)(p)
        p = nn.relu(p)
        p = p.reshape(*p.shape[:-3], -1)  # flatten spatial dims
        p = nn.Dense(self.num_actions)(p)  # logits (7,)

        # Value head
        v = nn.Conv(HEAD_FILTERS, (1, 1))(h)
        v = nn.BatchNorm(use_running_average=not train)(v)
        v = nn.relu(v)
        v = v.reshape(*v.shape[:-3], -1)
        v = nn.Dense(64)(v)
        v = nn.relu(v)
        v = nn.Dense(self.num_players)(v)
        v = nn.tanh(v)  # (3,) in [-1, 1]

        return p, v


def create_network():
    """Create network and initialize parameters."""
    net = AlphaZeroNet()
    return net


def init_params(rng):
    """Initialize network parameters with a dummy input."""
    net = create_network()
    dummy = jnp.zeros((1, ROWS, COLS, 6))
    variables = net.init(rng, dummy, train=False)
    return variables


def params_to_fp16(variables):
    """Cast params to float16 for faster inference."""
    return jax.tree.map(
        lambda x: x.astype(jnp.float16) if x.dtype == jnp.float32 else x,
        variables,
    )


def apply_inference(net, variables, x):
    """Run inference in float16."""
    fp16_vars = params_to_fp16(variables)
    x_fp16 = x.astype(jnp.float16)
    policy_logits, value = net.apply(fp16_vars, x_fp16, train=False)
    return policy_logits.astype(jnp.float32), value.astype(jnp.float32)


def test_network():
    """Smoke test: verify output shapes."""
    rng = jax.random.PRNGKey(0)
    net = create_network()
    variables = init_params(rng)

    dummy = jnp.zeros((2, ROWS, COLS, 6))
    policy, value = net.apply(variables, dummy, train=False)
    assert policy.shape == (2, NUM_ACTIONS), f"policy shape: {policy.shape}"
    assert value.shape == (2, NUM_PLAYERS), f"value shape: {value.shape}"

    # Test fp16 inference
    policy_fp16, value_fp16 = apply_inference(net, variables, dummy)
    assert policy_fp16.dtype == jnp.float32
    assert value_fp16.dtype == jnp.float32

    print("test_network passed!")


if __name__ == "__main__":
    test_network()
