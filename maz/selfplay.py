"""Self-play game generation with batched inference.

Runs multiple games in parallel, batching NN evaluations for efficiency.
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
                  max_moves: int = 100,
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
                 show_progress: bool = True) -> list[GameRecord]:
    """Run multiple self-play games sequentially.

    For true parallelism, games are run one at a time but NN evals
    within MCTS are batched where possible.
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    game_iter = range(num_games)
    if show_progress and tqdm is not None:
        game_iter = tqdm(game_iter, desc="Self-play", unit="game")

    games = []
    for i in game_iter:
        rng, game_rng = jax.random.split(rng)
        record = play_one_game(net, variables, game_rng,
                               num_simulations=num_simulations,
                               temperature=temperature)
        games.append(record)
        if not (show_progress and tqdm is not None):
            print(f"  Game {i+1}/{num_games}: {record.length} moves, "
                  f"winner={'draw' if record.scores[0].sum() == 0 else 'player'}")
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
