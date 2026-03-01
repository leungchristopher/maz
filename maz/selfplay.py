"""Self-play game generation with batched inference.

Runs multiple games in lock-step, batching NN evaluations across all active
games for full GPU utilization.
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
)


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


def run_selfplay(net, variables, rng: chex.PRNGKey,
                 num_games: int = 16,
                 num_simulations: int = NUM_SIMULATIONS,
                 temperature: float = 1.0,
                 show_progress: bool = True,
                 max_moves: int = 42,
                 temp_threshold: int = 15) -> list[GameRecord]:
    """Run batched lock-step self-play across all games simultaneously.

    All active games share each NN evaluation call, so the GPU processes
    a batch of size num_active_games instead of batch=1.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    # 1. Initialize all game states and per-game storage
    states = [init_state() for _ in range(num_games)]
    per_game_obs = [[] for _ in range(num_games)]       # encoded observations
    per_game_policies = [[] for _ in range(num_games)]  # MCTS policies

    # Split one RNG per game
    rng, *game_rngs = jax.random.split(rng, num_games + 1)

    move_iter = range(max_moves)
    if show_progress and tqdm is not None:
        move_iter = tqdm(move_iter, desc="Self-play", unit="move")

    for move_num in move_iter:
        # Determine which games are still active
        active = [i for i in range(num_games) if not states[i].done]
        if not active:
            break

        n_active = len(active)

        # Determine temperature for this move
        temp = temperature if move_num < temp_threshold else 0.01

        # 2. BATCHED root evaluation — single net.apply for all active games
        root_obs = jnp.stack([encode_state(states[i]) for i in active])
        root_logits, root_values = net.apply(variables, root_obs, train=False)

        # 3. Initialize per-game trees with root expansion
        trees = {}
        for k, gi in enumerate(active):
            game_rngs[gi], noise_rng = jax.random.split(game_rngs[gi])
            root_prior = jax.nn.softmax(root_logits[k])
            root_val = root_values[k]

            tree = new_tree()
            tree = init_root(tree, states[gi], root_prior, noise_rng)
            tree = expand_node(tree, 0, states[gi], root_prior, root_val)
            tree = backpropagate(
                tree, 0, root_val,
                jnp.array([0] + [-1] * 49, dtype=jnp.int32),
                jnp.int32(1),
            )
            trees[gi] = tree

        # 4. Lock-step simulations — batch NN across games at each sim step
        for _sim in range(num_simulations):
            # Select leaf per game (fast JIT ops, no NN)
            leaves = []
            for gi in active:
                tree, leaf_idx, leaf_state, path, path_len = select_leaf(
                    trees[gi], states[gi]
                )
                trees[gi] = tree
                leaves.append((gi, leaf_idx, leaf_state, path, path_len))

            # BATCHED leaf evaluation — single net.apply
            leaf_obs = jnp.stack([encode_state(lf[2]) for lf in leaves])
            leaf_logits, leaf_values = net.apply(
                variables, leaf_obs, train=False
            )

            # Expand + backprop per game (fast JIT ops, no NN)
            for k, (gi, leaf_idx, leaf_state, path, path_len) in enumerate(
                leaves
            ):
                leaf_prior = jax.nn.softmax(leaf_logits[k])
                leaf_val = leaf_values[k]

                # If game is over at this leaf, use actual scores
                actual_scores = get_scores(leaf_state)
                leaf_val = jnp.where(leaf_state.done, actual_scores, leaf_val)

                # Expand only if not done
                trees[gi] = jax.lax.cond(
                    ~leaf_state.done,
                    lambda t: expand_node(
                        t, leaf_idx, leaf_state, leaf_prior, leaf_val
                    ),
                    lambda t: t,
                    trees[gi],
                )

                # Backpropagate
                trees[gi] = backpropagate(
                    trees[gi], leaf_idx, leaf_val, path, path_len
                )

        # 5. Extract policies, sample actions, step games
        for gi in active:
            policy = get_policy(trees[gi], temp)

            # Record observation and policy
            per_game_obs[gi].append(encode_state(states[gi]))
            per_game_policies[gi].append(policy)

            # Sample action
            game_rngs[gi], action_rng = jax.random.split(game_rngs[gi])
            action = jax.random.choice(action_rng, NUM_ACTIONS, p=policy)
            states[gi] = step(states[gi], action)

        if show_progress and tqdm is not None:
            move_iter.set_postfix(active=n_active)

    # 6. Build GameRecord list
    games = []
    for gi in range(num_games):
        scores = get_scores(states[gi])
        game_len = len(per_game_obs[gi])

        if game_len == 0:
            games.append(GameRecord(
                states=jnp.zeros((1, ROWS, COLS, 6)),
                policies=jnp.ones((1, NUM_ACTIONS)) / NUM_ACTIONS,
                scores=jnp.zeros((1, NUM_PLAYERS)),
                length=0,
            ))
        else:
            states_arr = jnp.stack(per_game_obs[gi])
            policies_arr = jnp.stack(per_game_policies[gi])
            scores_arr = jnp.broadcast_to(scores, (game_len, NUM_PLAYERS))
            games.append(GameRecord(
                states=states_arr,
                policies=policies_arr,
                scores=scores_arr,
                length=game_len,
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
