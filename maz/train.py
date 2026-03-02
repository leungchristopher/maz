"""Training loop with AdamW, 1-cycle LR, replay buffer, and position averaging."""

import functools
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp
import optax
import chex

from maz.game import ROWS, COLS, NUM_PLAYERS, COLS as NUM_ACTIONS
from maz.selfplay import GameRecord


class ReplayBuffer:
    """Replay buffer with slow window: starts at 4 generations, grows to 20."""

    def __init__(self, initial_capacity: int = 4, max_capacity: int = 20,
                 grow_every: int = 5):
        self.capacity = initial_capacity
        self.max_capacity = max_capacity
        self.grow_every = grow_every
        self.generations: deque[list[GameRecord]] = deque()
        self.generation_count = 0

    def add_generation(self, games: list[GameRecord]):
        self.generations.append(games)
        self.generation_count += 1
        # Grow window periodically
        if self.generation_count % self.grow_every == 0:
            self.capacity = min(self.capacity + 1, self.max_capacity)
        # Drop oldest if over capacity
        while len(self.generations) > self.capacity:
            self.generations.popleft()

    def get_all_positions(self):
        """Flatten all games into (states, policies, scores) arrays."""
        all_states = []
        all_policies = []
        all_scores = []
        for gen in self.generations:
            for game in gen:
                if game.length > 0:
                    all_states.append(game.states)
                    all_policies.append(game.policies)
                    all_scores.append(game.scores)
        if not all_states:
            return None, None, None
        return (
            jnp.concatenate(all_states),
            jnp.concatenate(all_policies),
            jnp.concatenate(all_scores),
        )


def average_positions(states: jnp.ndarray, policies: jnp.ndarray,
                      scores: jnp.ndarray):
    """Deduplicate positions by board hash, averaging π and z.

    Uses numpy vectorized hashing + pandas-free groupby via np.unique.
    """
    # Move to numpy for fast hashing
    s_np = np.asarray(states).reshape(len(states), -1)
    p_np = np.asarray(policies)
    z_np = np.asarray(scores)

    # Hash each row to a single int64 for fast grouping
    # Use a view-based approach: treat each flattened state as bytes
    s_bytes = np.ascontiguousarray(s_np).view(
        np.dtype((np.void, s_np.dtype.itemsize * s_np.shape[1])))
    _, unique_idx, inverse = np.unique(
        s_bytes, return_index=True, return_inverse=True)
    inverse = inverse.ravel()

    n_unique = len(unique_idx)

    # Scatter-add policies and scores, then divide by counts
    counts = np.bincount(inverse, minlength=n_unique).astype(np.float32)
    avg_policies = np.zeros((n_unique, p_np.shape[1]), dtype=np.float32)
    avg_scores = np.zeros((n_unique, z_np.shape[1]), dtype=np.float32)
    np.add.at(avg_policies, inverse, p_np)
    np.add.at(avg_scores, inverse, z_np)
    avg_policies /= counts[:, None]
    avg_scores /= counts[:, None]

    return (
        jnp.asarray(states[unique_idx]),
        jnp.asarray(avg_policies),
        jnp.asarray(avg_scores),
    )


def make_1cycle_schedule(peak_lr: float, total_steps: int):
    """1-cycle LR: warm up first 30%, cosine decay remaining 70%."""
    warmup_steps = int(total_steps * 0.3)
    decay_steps = total_steps - warmup_steps
    min_lr = peak_lr / 25.0

    return optax.join_schedules([
        optax.linear_schedule(min_lr, peak_lr, warmup_steps),
        optax.cosine_decay_schedule(peak_lr, decay_steps, alpha=min_lr / peak_lr),
    ], boundaries=[warmup_steps])


def create_optimizer(peak_lr: float = 1e-3, weight_decay: float = 1e-4,
                     total_steps: int = 1000):
    """Create AdamW optimizer with 1-cycle schedule."""
    schedule = make_1cycle_schedule(peak_lr, total_steps)
    return optax.adamw(schedule, weight_decay=weight_decay)


def compute_loss(params, batch_state, net, states, policies, scores):
    """Compute combined MSE + CE loss.

    L = MSE(z, v) + CE(π, p)
    MSE = mean((z - v)^2) over 3-player vector
    CE = -sum(π * log(p))
    """
    policy_logits, values = net.apply(
        {"params": params, "batch_stats": batch_state},
        states, train=True,
        mutable=["batch_stats"],
    )
    # Unpack mutable returns
    if isinstance(policy_logits, tuple) and len(policy_logits) == 2:
        (policy_logits, values), new_batch_state = policy_logits, values
    else:
        new_batch_state = {"batch_stats": batch_state}

    # Value loss: MSE over 3-player vector
    value_loss = jnp.mean((scores - values) ** 2)

    # Policy loss: cross-entropy
    log_probs = jax.nn.log_softmax(policy_logits)
    policy_loss = -jnp.mean(jnp.sum(policies * log_probs, axis=-1))

    total_loss = value_loss + policy_loss
    return total_loss, new_batch_state


def make_train_step(net, optimizer):
    """Create a JIT-compiled training step."""

    @jax.jit
    def train_step(params, batch_stats, opt_state, states, policies, scores):
        def loss_fn(params):
            (policy_logits, values), updates = net.apply(
                {"params": params, "batch_stats": batch_stats},
                states, train=True,
                mutable=["batch_stats"],
            )
            value_loss = jnp.mean((scores - values) ** 2)
            log_probs = jax.nn.log_softmax(policy_logits)
            policy_loss = -jnp.mean(jnp.sum(policies * log_probs, axis=-1))
            total_loss = value_loss + policy_loss
            return total_loss, (value_loss, policy_loss, updates)

        grads, (v_loss, p_loss, updates) = jax.grad(loss_fn, has_aux=True)(params)
        param_updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, param_updates)
        new_batch_stats = updates["batch_stats"]
        return new_params, new_batch_stats, new_opt_state, v_loss, p_loss

    return train_step


def train_on_buffer(net, variables, optimizer, opt_state,
                    replay_buffer: ReplayBuffer,
                    batch_size: int = 64,
                    epochs: int = 2,
                    do_averaging: bool = True,
                    show_progress: bool = True):
    """Train on all positions in the replay buffer.

    Returns updated (variables, opt_state, metrics).
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    states, policies, scores = replay_buffer.get_all_positions()
    if states is None:
        return variables, opt_state, {}

    # Position averaging
    if do_averaging and len(states) > 0:
        states, policies, scores = average_positions(states, policies, scores)

    num_positions = len(states)
    print(f"  Training on {num_positions} unique positions")

    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    train_step = make_train_step(net, optimizer)

    total_v_loss = 0.0
    total_p_loss = 0.0
    num_steps = 0

    num_batches = ((num_positions + batch_size - 1) // batch_size) * epochs
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=num_batches, desc="Training", unit="batch")

    for epoch in range(epochs):
        # Shuffle
        rng = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(rng, num_positions)
        states_shuf = states[perm]
        policies_shuf = policies[perm]
        scores_shuf = scores[perm]

        for start in range(0, num_positions, batch_size):
            end = min(start + batch_size, num_positions)
            batch_s = states_shuf[start:end]
            batch_p = policies_shuf[start:end]
            batch_z = scores_shuf[start:end]

            params, batch_stats, opt_state, v_loss, p_loss = train_step(
                params, batch_stats, opt_state, batch_s, batch_p, batch_z
            )
            total_v_loss += float(v_loss)
            total_p_loss += float(p_loss)
            num_steps += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(v_loss=f"{float(v_loss):.4f}",
                                 p_loss=f"{float(p_loss):.4f}")

    if pbar is not None:
        pbar.close()

    metrics = {
        "value_loss": total_v_loss / max(num_steps, 1),
        "policy_loss": total_p_loss / max(num_steps, 1),
        "total_loss": (total_v_loss + total_p_loss) / max(num_steps, 1),
        "num_positions": num_positions,
        "num_steps": num_steps,
    }

    new_variables = {"params": params, "batch_stats": batch_stats}
    return new_variables, opt_state, metrics
