"""Fully vectorized self-play: vmap over games, JIT-fused sim loop on GPU.

All tree operations are vmapped over N games. The simulation loop uses
jax.lax.fori_loop + scan so an entire move compiles to a single XLA kernel.
K=8 leaves are selected per iteration (virtual-loss steered) for larger NN
batch sizes and fewer sequential steps.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import chex

from maz.game import (
    ROWS, COLS, NUM_PLAYERS, COLS as NUM_ACTIONS,
    GameState, init_state, step, get_valid_actions, encode_state, get_scores,
)
from maz.mcts import (
    search, NUM_SIMULATIONS,
    new_tree, init_root, expand_node, backpropagate, select_leaf, get_policy,
    batched_new_tree, expand_or_noop,
)

SIMS_PER_BATCH = 8  # K parallel sims per fori_loop iteration


class GameRecord(NamedTuple):
    """A single completed game's data."""
    states: chex.Array     # (T, 6, 7, 6) encoded states
    policies: chex.Array   # (T, 7) MCTS policies
    scores: chex.Array     # (T, 3) outcome scores (same for all timesteps)
    length: int            # actual game length


class PositionCache:
    """LRU cache for NN evaluations keyed on board bytes."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: dict[bytes, tuple] = {}
        self._order: list[bytes] = []

    def get(self, board: jnp.ndarray):
        key = board.tobytes()
        if key in self.cache:
            return self.cache[key]
        return None

    def put(self, board: jnp.ndarray, prior: jnp.ndarray, value: jnp.ndarray):
        key = board.tobytes()
        if key not in self.cache:
            self._order.append(key)
            if len(self._order) > self.max_size:
                oldest = self._order.pop(0)
                del self.cache[oldest]
        self.cache[key] = (prior, value)

    def clear(self):
        self.cache.clear()
        self._order.clear()


def play_one_game(net, variables, rng: chex.PRNGKey,
                  num_simulations: int = NUM_SIMULATIONS,
                  max_moves: int = 42,
                  temperature: float = 1.0,
                  temp_threshold: int = 15) -> GameRecord:
    """Play a single self-play game using MCTS.

    Args:
        net: AlphaZero network
        variables: network parameters
        rng: random key
        num_simulations: MCTS simulations per move
        max_moves: maximum game length
        temperature: exploration temperature
        temp_threshold: after this many moves, use temperature=0
    """
    state = init_state()
    states_list = []
    policies_list = []

    for move_num in range(max_moves):
        if state.done:
            break

        rng, search_rng, action_rng = jax.random.split(rng, 3)
        temp = temperature if move_num < temp_threshold else 0.01
        policy = search(state, net, variables, search_rng,
                        num_simulations=num_simulations, temperature=temp)

        obs = encode_state(state)
        states_list.append(obs)
        policies_list.append(policy)

        # Sample action from policy
        action = jax.random.choice(action_rng, NUM_ACTIONS, p=policy)
        state = step(state, action)

    # Get final scores
    scores = get_scores(state)
    game_len = len(states_list)

    if game_len == 0:
        # Edge case: game was already done
        return GameRecord(
            states=jnp.zeros((1, ROWS, COLS, 6)),
            policies=jnp.ones((1, NUM_ACTIONS)) / NUM_ACTIONS,
            scores=jnp.zeros((1, NUM_PLAYERS)),
            length=0,
        )

    states_arr = jnp.stack(states_list)
    policies_arr = jnp.stack(policies_list)
    # Broadcast scores to all timesteps
    scores_arr = jnp.broadcast_to(scores, (game_len, NUM_PLAYERS))

    return GameRecord(
        states=states_arr,
        policies=policies_arr,
        scores=scores_arr,
        length=game_len,
    )


def batched_init_states(n: int) -> GameState:
    """Create N initial game states with leading batch dimension."""
    return GameState(
        board=jnp.zeros((n, ROWS, COLS), dtype=jnp.int8),
        current_player=jnp.zeros(n, dtype=jnp.int32),
        done=jnp.zeros(n, dtype=jnp.bool_),
        winner=jnp.full(n, -1, dtype=jnp.int32),
    )


def _make_move_fns(net, num_games, num_simulations):
    """Factory returning JIT-compiled functions for one move.

    Returns (root_fn, sim_batch_fn, finish_fn):
      - root_fn: root eval + tree init → (obs, priors, values, trees)
      - sim_batch_fn: K parallel select + batched NN + expand/backprop
      - finish_fn: policy extraction + action sampling

    Splitting avoids nesting fori_loop/scan inside a single JIT, which
    causes multi-minute XLA compilation. The Python sim loop (7 iters)
    adds negligible overhead vs. the old 50-iter loop.
    """
    N = num_games
    K = SIMS_PER_BATCH
    num_batches = (num_simulations + K - 1) // K  # ceil division

    # Vmapped ops (closure-captured, static for JIT)
    v_encode = jax.vmap(encode_state)
    v_init_root = jax.vmap(init_root)
    v_select = jax.vmap(select_leaf)
    v_expand_or_noop = jax.vmap(expand_or_noop)
    v_backprop = jax.vmap(backpropagate)
    v_get_policy = jax.vmap(get_policy, in_axes=(0, None))
    v_step = jax.vmap(step)
    v_get_scores = jax.vmap(get_scores)
    v_expand = jax.vmap(expand_node)

    # Shared constants
    root_paths = jnp.broadcast_to(
        jnp.array([0] + [-1] * 19, jnp.int32), (N, 20))
    node_zeros = jnp.zeros(N, dtype=jnp.int32)
    ones_N = jnp.ones(N, dtype=jnp.int32)

    @jax.jit
    def root_fn(states, variables, noise_rngs):
        """Root eval + tree init."""
        obs = v_encode(states)
        logits, values = net.apply(variables, obs, train=False)
        priors = jax.nn.softmax(logits)
        trees = batched_new_tree(N)
        trees = v_init_root(trees, states, priors, noise_rngs)
        trees = v_expand(trees, node_zeros, states, priors, values)
        trees = v_backprop(trees, node_zeros, values, root_paths, ones_N)
        return obs, trees

    @jax.jit
    def sim_batch_fn(trees, states, variables):
        """One batch of K parallel sims: select K leaves, batch NN, expand+backprop."""
        # 1) Select K leaves sequentially (virtual loss steers diversity)
        def select_one(trees, _):
            trees, idxs, lstates, paths, plens = v_select(trees, states)
            return trees, (idxs, lstates, paths, plens)
        trees_vl, (all_idxs, all_lstates, all_paths, all_plens) = \
            jax.lax.scan(select_one, trees, None, length=K)

        # 2) Batch NN eval: (K,N,...) → (K*N,...) → NN → reshape
        flat_lstates = jax.tree.map(
            lambda x: x.reshape(K * N, *x.shape[2:]), all_lstates)
        flat_obs = v_encode(flat_lstates)
        flat_logits, flat_vals = net.apply(
            variables, flat_obs, train=False)
        flat_scores = v_get_scores(flat_lstates)

        all_priors = jax.nn.softmax(flat_logits).reshape(K, N, NUM_ACTIONS)
        all_vals = flat_vals.reshape(K, N, NUM_PLAYERS)
        all_scores = flat_scores.reshape(K, N, NUM_PLAYERS)
        all_done = all_lstates.done  # (K, N)
        all_vals = jnp.where(all_done[:, :, None], all_scores, all_vals)

        # 3) Expand + backprop K leaves sequentially
        def expand_backprop_one(trees, k_data):
            idxs, lstates_k, priors_k, vals_k, done_k, paths_k, plens_k = k_data
            trees = v_expand_or_noop(
                trees, idxs, lstates_k, priors_k, vals_k, done_k)
            trees = v_backprop(trees, idxs, vals_k, paths_k, plens_k)
            return trees, None

        k_data = (all_idxs, all_lstates, all_priors, all_vals,
                  all_done, all_paths, all_plens)
        trees, _ = jax.lax.scan(
            expand_backprop_one, trees_vl, k_data)
        return trees

    @jax.jit
    def finish_fn(trees, states, action_rngs, temp):
        """Policy extraction + action sampling."""
        policies = v_get_policy(trees, temp)
        safe_policies = jnp.where(
            states.done[:, None],
            jnp.ones((N, NUM_ACTIONS)) / NUM_ACTIONS, policies)
        actions = jax.vmap(
            lambda r, p: jax.random.choice(r, NUM_ACTIONS, p=p)
        )(action_rngs, safe_policies)
        new_states = v_step(states, actions)
        return safe_policies, actions, new_states

    return root_fn, sim_batch_fn, finish_fn, num_batches


def run_selfplay(net, variables, rng: chex.PRNGKey,
                 num_games: int = 16,
                 num_simulations: int = NUM_SIMULATIONS,
                 temperature: float = 1.0,
                 show_progress: bool = True,
                 max_moves: int = 42,
                 temp_threshold: int = 15) -> list[GameRecord]:
    """Run fully vectorized self-play with JIT-fused simulation loop.

    Each move compiles to a single XLA kernel via jax.lax.fori_loop.
    K=8 leaves are selected per iteration for larger NN batch sizes.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    N = num_games
    root_fn, sim_batch_fn, finish_fn, num_batches = _make_move_fns(
        net, N, num_simulations)
    states = batched_init_states(N)

    # Pre-allocate recording arrays: (N, max_moves, ...)
    all_obs = jnp.zeros((N, max_moves, ROWS, COLS, 6))
    all_policies = jnp.zeros((N, max_moves, NUM_ACTIONS))
    game_lengths = jnp.zeros(N, dtype=jnp.int32)

    # Pre-split all RNG keys: (max_moves, N, 2, 2) — 2 keys per game per move
    move_rngs = jax.random.split(rng, max_moves * N * 2).reshape(
        max_moves, N, 2, 2
    )

    move_iter = range(max_moves)
    if show_progress and tqdm is not None:
        move_iter = tqdm(move_iter, desc="Self-play", unit="move")

    for move_num in move_iter:
        if bool(states.done.all()):
            break
        active_mask = ~states.done
        temp = jnp.float32(temperature if move_num < temp_threshold else 0.01)

        obs, trees = root_fn(states, variables, move_rngs[move_num, :, 0])
        for _ in range(num_batches):
            trees = sim_batch_fn(trees, states, variables)
        policies, actions, new_states = finish_fn(
            trees, states, move_rngs[move_num, :, 1], temp)

        # Record (cheap array updates)
        all_obs = all_obs.at[:, move_num].set(
            jnp.where(active_mask[:, None, None, None], obs, 0.0))
        all_policies = all_policies.at[:, move_num].set(
            jnp.where(active_mask[:, None], policies, 0.0))
        game_lengths = game_lengths + active_mask.astype(jnp.int32)

        # Mask done games: keep old state for finished games
        states = jax.tree.map(
            lambda new, old: jnp.where(
                active_mask.reshape(-1, *([1] * (new.ndim - 1))), new, old),
            new_states, states)

        if show_progress and tqdm is not None:
            n_active = int(active_mask.sum())
            move_iter.set_postfix(active=n_active)

    # Build GameRecord list (CPU, once)
    final_scores = jax.vmap(get_scores)(states)
    games = []
    for i, L in enumerate(int(x) for x in game_lengths):
        if L == 0:
            games.append(GameRecord(
                states=jnp.zeros((1, ROWS, COLS, 6)),
                policies=jnp.ones((1, NUM_ACTIONS)) / NUM_ACTIONS,
                scores=jnp.zeros((1, NUM_PLAYERS)),
                length=0,
            ))
        else:
            games.append(GameRecord(
                states=all_obs[i, :L],
                policies=all_policies[i, :L],
                scores=jnp.broadcast_to(final_scores[i], (L, NUM_PLAYERS)),
                length=L,
            ))

    if show_progress and tqdm is None:
        for i, g in enumerate(games):
            print(f"  Game {i+1}/{num_games}: {g.length} moves, "
                  f"winner={'draw' if g.scores[0].sum() == 0 else 'player'}")

    return games


if __name__ == "__main__":
    from maz.network import create_network, init_params

    rng = jax.random.PRNGKey(0)
    net = create_network()
    rng, init_rng = jax.random.split(rng)
    variables = init_params(init_rng)

    rng, sp_rng = jax.random.split(rng)
    games = run_selfplay(net, variables, sp_rng, num_games=2, num_simulations=5)
    for i, g in enumerate(games):
        print(f"Game {i}: length={g.length}, states={g.states.shape}")
    print("selfplay test passed!")
