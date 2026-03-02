"""Benchmark: AlphaZero vs classical multiplayer agents on 4x5 connect-3.

Evaluates a trained AlphaZero agent against classical game-tree search baselines
and plots win rates across homogeneous/heterogeneous player combinations.

GPU-accelerated: AZ moves use the batched selfplay MCTS infrastructure
(K×N NN batch size per sim batch) instead of sequential batch-size-1 calls.

Usage:
    python -m maz.benchmark checkpoint.pkl [--sims 200] [--games 50] \
      [--maxn-depth 4] [--paranoid-depth 6] [--shapley-depth 3] \
      [--output benchmark_results]
"""

import argparse
import itertools
import json
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from maz.game import (
    ROWS, COLS, NUM_PLAYERS, CONNECT_K,
    GameState, init_state, step, get_valid_actions, get_scores, check_win,
    encode_state,
)
from maz.mcts import batched_new_tree
from maz.network import create_network
from maz.play import load_from_pkl
from maz.selfplay import _make_move_fns, batched_init_states

# Center-first move ordering for better pruning
CENTER_ORDER = [2, 1, 3, 0, 4]


# ---------------------------------------------------------------------------
# Shared heuristic evaluation
# ---------------------------------------------------------------------------

def _enumerate_lines():
    """Enumerate all possible connect-K lines on the board.

    Returns list of (row_indices, col_indices) tuples, each of length CONNECT_K.
    """
    lines = []
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        for r in range(ROWS):
            for c in range(COLS):
                rows = [r + i * dr for i in range(CONNECT_K)]
                cols = [c + i * dc for i in range(CONNECT_K)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in zip(rows, cols)):
                    lines.append((rows, cols))
    return lines

# Precompute line positions as numpy arrays
_LINES = _enumerate_lines()
_LINE_ROWS = [np.array(r, dtype=np.intp) for r, _ in _LINES]
_LINE_COLS = [np.array(c, dtype=np.intp) for _, c in _LINES]


def heuristic_eval(board_np):
    """Evaluate board position for all players.

    Args:
        board_np: numpy array (ROWS, COLS) with 0=empty, 1/2/3=player pieces.

    Returns:
        [v0, v1, v2] scores in [-1, 1].
    """
    scores = [0.0, 0.0, 0.0]
    for rows, cols in zip(_LINE_ROWS, _LINE_COLS):
        cells = board_np[rows, cols]
        for p in range(3):
            player_val = p + 1
            own = np.sum(cells == player_val)
            others = np.sum((cells != 0) & (cells != player_val))
            if others > 0:
                continue  # line blocked by opponent
            if own == CONNECT_K:
                scores[p] += 10.0  # completed line
            elif own == 2:
                scores[p] += 0.5
            elif own == 1:
                scores[p] += 0.1
    # Normalize to [-1, 1]
    max_abs = max(abs(s) for s in scores) if any(s != 0 for s in scores) else 1.0
    if max_abs > 1.0:
        scores = [s / max_abs for s in scores]
    return scores


# ---------------------------------------------------------------------------
# Game simulation helpers (pure numpy, no JAX overhead)
# ---------------------------------------------------------------------------

def _np_get_valid(board_np):
    """Return list of valid columns."""
    return [c for c in range(COLS) if board_np[0, c] == 0]


def _np_drop(board_np, col, player_val):
    """Drop piece, return new board. Mutates nothing."""
    b = board_np.copy()
    for r in range(ROWS - 1, -1, -1):
        if b[r, col] == 0:
            b[r, col] = player_val
            return b
    return b  # shouldn't happen if col is valid


def _np_check_win(board_np, player_val):
    """Check if player_val has connect-K."""
    for rows, cols in zip(_LINE_ROWS, _LINE_COLS):
        if np.all(board_np[rows, cols] == player_val):
            return True
    return False


def _np_is_full(board_np):
    return np.all(board_np != 0)


# ---------------------------------------------------------------------------
# Agent ABC
# ---------------------------------------------------------------------------

class Agent(ABC):
    name: str

    @abstractmethod
    def select_action(self, state, rng_key=None):
        """Select an action given a GameState. Returns int column."""
        ...

    def __repr__(self):
        return self.name


# ---------------------------------------------------------------------------
# 1. RandomAgent
# ---------------------------------------------------------------------------

class RandomAgent(Agent):
    name = "Random"

    def select_action(self, state, rng_key=None):
        valid = np.array(get_valid_actions(state))
        cols = [c for c in range(COLS) if valid[c]]
        if rng_key is not None:
            idx = int(jax.random.randint(rng_key, (), 0, len(cols)))
        else:
            idx = np.random.randint(len(cols))
        return cols[idx]


# ---------------------------------------------------------------------------
# 2. GreedyAgent
# ---------------------------------------------------------------------------

class GreedyAgent(Agent):
    name = "Greedy"

    def select_action(self, state, rng_key=None):
        board_np = np.asarray(state.board)
        current = int(state.current_player)
        current_val = current + 1
        valid = _np_get_valid(board_np)

        # Can I win?
        for c in valid:
            b = _np_drop(board_np, c, current_val)
            if _np_check_win(b, current_val):
                return c

        # Can any opponent win next? Block the first threat found.
        for opp in range(NUM_PLAYERS):
            if opp == current:
                continue
            opp_val = opp + 1
            for c in valid:
                b = _np_drop(board_np, c, opp_val)
                if _np_check_win(b, opp_val):
                    return c

        # Center preference
        for c in CENTER_ORDER:
            if c in valid:
                return c
        return valid[0]


# ---------------------------------------------------------------------------
# 3. MaxNAgent — multiplayer minimax
# ---------------------------------------------------------------------------

class MaxNAgent(Agent):
    def __init__(self, depth=4):
        self.depth = depth
        self.name = f"MaxN(d={depth})"
        self._tt = {}

    def select_action(self, state, rng_key=None):
        self._tt.clear()
        board_np = np.asarray(state.board)
        current = int(state.current_player)
        best_action = CENTER_ORDER[0]
        best_val = -float("inf")

        valid = _np_get_valid(board_np)
        for c in sorted(valid, key=lambda x: CENTER_ORDER.index(x) if x in CENTER_ORDER else 99):
            child = _np_drop(board_np, c, current + 1)
            if _np_check_win(child, current + 1):
                return c
            vals = self._maxn(child, (current + 1) % NUM_PLAYERS, self.depth - 1)
            if vals[current] > best_val:
                best_val = vals[current]
                best_action = c
        return best_action

    def _maxn(self, board_np, player, depth):
        key = (board_np.tobytes(), player, depth)
        if key in self._tt:
            return self._tt[key]

        if _np_is_full(board_np):
            result = [0.0, 0.0, 0.0]
            self._tt[key] = result
            return result

        # Check if previous player won (they just moved)
        prev = (player - 1) % NUM_PLAYERS
        if _np_check_win(board_np, prev + 1):
            result = [-1.0, -1.0, -1.0]
            result[prev] = 1.0
            self._tt[key] = result
            return result

        if depth == 0:
            result = heuristic_eval(board_np)
            self._tt[key] = result
            return result

        valid = _np_get_valid(board_np)
        best_vals = None
        for c in sorted(valid, key=lambda x: CENTER_ORDER.index(x) if x in CENTER_ORDER else 99):
            child = _np_drop(board_np, c, player + 1)
            if _np_check_win(child, player + 1):
                vals = [-1.0, -1.0, -1.0]
                vals[player] = 1.0
            else:
                vals = self._maxn(child, (player + 1) % NUM_PLAYERS, depth - 1)
            if best_vals is None or vals[player] > best_vals[player]:
                best_vals = vals
        self._tt[key] = best_vals
        return best_vals


# ---------------------------------------------------------------------------
# 4. ParanoidAgent — all opponents minimize root player's score
# ---------------------------------------------------------------------------

class ParanoidAgent(Agent):
    def __init__(self, depth=6):
        self.depth = depth
        self.name = f"Paranoid(d={depth})"
        self._tt = {}

    def select_action(self, state, rng_key=None):
        self._tt.clear()
        board_np = np.asarray(state.board)
        self._root_player = int(state.current_player)
        current = self._root_player
        best_action = CENTER_ORDER[0]
        best_val = -float("inf")

        valid = _np_get_valid(board_np)
        for c in sorted(valid, key=lambda x: CENTER_ORDER.index(x) if x in CENTER_ORDER else 99):
            child = _np_drop(board_np, c, current + 1)
            if _np_check_win(child, current + 1):
                return c
            val = self._paranoid(child, (current + 1) % NUM_PLAYERS,
                                 self.depth - 1, -float("inf"), float("inf"))
            if val > best_val:
                best_val = val
                best_action = c
        return best_action

    def _paranoid(self, board_np, player, depth, alpha, beta):
        key = (board_np.tobytes(), player, depth)
        if key in self._tt:
            return self._tt[key]

        if _np_is_full(board_np):
            self._tt[key] = 0.0
            return 0.0

        prev = (player - 1) % NUM_PLAYERS
        if _np_check_win(board_np, prev + 1):
            val = 1.0 if prev == self._root_player else -1.0
            self._tt[key] = val
            return val

        if depth == 0:
            h = heuristic_eval(board_np)
            val = h[self._root_player]
            self._tt[key] = val
            return val

        valid = _np_get_valid(board_np)
        is_max = (player == self._root_player)

        if is_max:
            val = -float("inf")
            for c in sorted(valid, key=lambda x: CENTER_ORDER.index(x) if x in CENTER_ORDER else 99):
                child = _np_drop(board_np, c, player + 1)
                if _np_check_win(child, player + 1):
                    self._tt[key] = 1.0
                    return 1.0
                v = self._paranoid(child, (player + 1) % NUM_PLAYERS,
                                   depth - 1, alpha, beta)
                val = max(val, v)
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
        else:
            val = float("inf")
            for c in sorted(valid, key=lambda x: CENTER_ORDER.index(x) if x in CENTER_ORDER else 99):
                child = _np_drop(board_np, c, player + 1)
                if _np_check_win(child, player + 1):
                    v = -1.0
                else:
                    v = self._paranoid(child, (player + 1) % NUM_PLAYERS,
                                       depth - 1, alpha, beta)
                val = min(val, v)
                beta = min(beta, val)
                if alpha >= beta:
                    break

        self._tt[key] = val
        return val


# ---------------------------------------------------------------------------
# 5. ShapleyAgent — coalition-based evaluation via Shapley values
# ---------------------------------------------------------------------------

class ShapleyAgent(Agent):
    def __init__(self, depth=3):
        self.depth = depth
        self.name = f"Shapley(d={depth})"
        self._tt = {}

    def select_action(self, state, rng_key=None):
        self._tt.clear()
        board_np = np.asarray(state.board)
        current = int(state.current_player)
        valid = _np_get_valid(board_np)

        # Check for immediate win
        for c in valid:
            child = _np_drop(board_np, c, current + 1)
            if _np_check_win(child, current + 1):
                return c

        best_action = CENTER_ORDER[0]
        best_shapley = -float("inf")

        for c in sorted(valid, key=lambda x: CENTER_ORDER.index(x) if x in CENTER_ORDER else 99):
            child = _np_drop(board_np, c, current + 1)
            shapley_vals = self._compute_shapley(child, (current + 1) % NUM_PLAYERS)
            if shapley_vals[current] > best_shapley:
                best_shapley = shapley_vals[current]
                best_action = c
        return best_action

    def _compute_shapley(self, board_np, next_player):
        """Compute Shapley values for all 3 players at this position."""
        players = [0, 1, 2]
        shapley = [0.0, 0.0, 0.0]

        # Enumerate all non-empty coalitions (2^3 - 1 = 7)
        for size in range(1, NUM_PLAYERS + 1):
            for coalition in itertools.combinations(players, size):
                coal_set = set(coalition)
                val = self._coalition_eval(board_np, next_player, coal_set, self.depth)
                # Distribute marginal contributions
                for p in players:
                    if p in coal_set:
                        # Marginal: v(S) - v(S \ {p})
                        if len(coal_set) == 1:
                            marginal = val
                        else:
                            sub = coal_set - {p}
                            sub_val = self._coalition_eval(board_np, next_player,
                                                           sub, self.depth)
                            marginal = val - sub_val
                        # Weight: |S-1|!(n-|S|)!/n!
                        n = NUM_PLAYERS
                        s = len(coal_set)
                        weight = math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
                        shapley[p] += weight * marginal
        return shapley

    def _coalition_eval(self, board_np, player, coalition, depth):
        """Alpha-beta where coalition members maximize, outsiders minimize."""
        key = (board_np.tobytes(), player, frozenset(coalition), depth)
        if key in self._tt:
            return self._tt[key]

        if _np_is_full(board_np):
            self._tt[key] = 0.0
            return 0.0

        prev = (player - 1) % NUM_PLAYERS
        if _np_check_win(board_np, prev + 1):
            val = 1.0 if prev in coalition else -1.0
            self._tt[key] = val
            return val

        if depth == 0:
            h = heuristic_eval(board_np)
            # Sum of coalition members' heuristic scores
            val = sum(h[p] for p in coalition) / len(coalition)
            self._tt[key] = val
            return val

        valid = _np_get_valid(board_np)
        is_max = (player in coalition)

        if is_max:
            val = -float("inf")
            alpha, beta = -float("inf"), float("inf")
            for c in sorted(valid, key=lambda x: CENTER_ORDER.index(x) if x in CENTER_ORDER else 99):
                child = _np_drop(board_np, c, player + 1)
                if _np_check_win(child, player + 1):
                    v = 1.0
                else:
                    v = self._coalition_eval(child, (player + 1) % NUM_PLAYERS,
                                             coalition, depth - 1)
                val = max(val, v)
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
        else:
            val = float("inf")
            alpha, beta = -float("inf"), float("inf")
            for c in sorted(valid, key=lambda x: CENTER_ORDER.index(x) if x in CENTER_ORDER else 99):
                child = _np_drop(board_np, c, player + 1)
                if _np_check_win(child, player + 1):
                    v = -1.0
                else:
                    v = self._coalition_eval(child, (player + 1) % NUM_PLAYERS,
                                             coalition, depth - 1)
                val = min(val, v)
                beta = min(beta, val)
                if alpha >= beta:
                    break

        self._tt[key] = val
        return val


# ---------------------------------------------------------------------------
# 6. AlphaZeroAgent (marker for batched play; not used for select_action)
# ---------------------------------------------------------------------------

class AlphaZeroAgent(Agent):
    """Marker agent for AlphaZero. Actual move selection uses batched MCTS."""
    def __init__(self, sims=200):
        self.sims = sims
        self.name = f"AZ(s={sims})"

    def select_action(self, state, rng_key=None):
        raise RuntimeError("AlphaZeroAgent.select_action should not be called; "
                           "use batched play_games instead")


# ---------------------------------------------------------------------------
# Batched game runner — GPU-accelerated AZ via selfplay MCTS
# ---------------------------------------------------------------------------

def _play_games_batched(seat_agents, n_games, rng_key, net, variables, sims):
    """Run n_games in parallel with batched GPU MCTS for AZ seats.

    At move M, current_player = M % 3 (deterministic rotation), so all
    active games share the same seat. AZ turns batch all N game states
    through the selfplay MCTS infrastructure (K×N NN batch per sim batch).
    Classical turns loop through individual games on CPU.

    Args:
        seat_agents: list of 3 Agent instances (one per seat).
        n_games: number of games to run in parallel.
        rng_key: JAX PRNG key.
        net: AlphaZero network.
        variables: network parameters.
        sims: MCTS simulations for AZ.

    Returns:
        (winners, move_counts) — numpy arrays of shape (n_games,).
        winners[i] = -1 for draw, 0/1/2 for winning seat.
    """
    N = n_games
    max_moves = ROWS * COLS  # absolute max

    # Identify which seats are AZ
    az_seats = {i for i, a in enumerate(seat_agents) if isinstance(a, AlphaZeroAgent)}

    # Get batched MCTS functions (JIT-compiled, cached by (net, N, sims))
    root_fn, sim_batch_fn, finish_fn, num_batches = _make_move_fns(net, N, sims)
    v_step = jax.vmap(step)

    # Initialize batched game states
    states = batched_init_states(N)
    done_np = np.zeros(N, dtype=bool)
    winners_np = np.full(N, -1, dtype=np.int32)
    move_counts_np = np.zeros(N, dtype=np.int32)

    for move_num in range(max_moves):
        if done_np.all():
            break

        active = ~done_np
        current_seat = move_num % NUM_PLAYERS

        if current_seat in az_seats:
            # --- Batched GPU MCTS for all N games ---
            rng_key, noise_rng, action_rng = jax.random.split(rng_key, 3)
            noise_rngs = jax.random.split(noise_rng, N)
            action_rngs = jax.random.split(action_rng, N)

            _, trees = root_fn(states, variables, noise_rngs)
            for _ in range(num_batches):
                trees = sim_batch_fn(trees, states, variables)
            _policies, actions, new_states = finish_fn(
                trees, states, action_rngs, jnp.float32(0.01))
        else:
            # --- Classical agent: loop through active games on CPU ---
            agent = seat_agents[current_seat]
            action_list = []
            for i in range(N):
                if active[i]:
                    single_state = jax.tree.map(lambda x, _i=i: x[_i], states)
                    rng_key, sub = jax.random.split(rng_key)
                    action_list.append(agent.select_action(single_state, rng_key=sub))
                else:
                    action_list.append(0)  # dummy for done games
            actions = jnp.array(action_list, dtype=jnp.int32)
            new_states = v_step(states, actions)

        # Mask: keep old state for already-done games
        active_jnp = jnp.array(active)
        states = jax.tree.map(
            lambda new, old: jnp.where(
                active_jnp.reshape(-1, *([1] * (new.ndim - 1))), new, old),
            new_states, states)

        # Record newly finished games
        done_arr = np.array(states.done)
        winner_arr = np.array(states.winner)
        for i in range(N):
            if active[i] and done_arr[i] and not done_np[i]:
                done_np[i] = True
                winners_np[i] = winner_arr[i]
                move_counts_np[i] = move_num + 1

    # Games that hit max_moves without ending are draws
    for i in range(N):
        if not done_np[i]:
            done_np[i] = True
            move_counts_np[i] = max_moves

    return winners_np, move_counts_np


def play_one_game(agents, rng_key=None):
    """Play one game with 3 classical agents. Returns (winner, scores, num_moves)."""
    state = init_state()
    move = 0
    while not state.done:
        cp = int(state.current_player)
        agent = agents[cp]
        if rng_key is not None:
            rng_key, sub = jax.random.split(rng_key)
        else:
            sub = None
        action = agent.select_action(state, rng_key=sub)
        state = step(state, jnp.int32(action))
        move += 1
    return int(state.winner), np.array(get_scores(state)), move


# ---------------------------------------------------------------------------
# Matchup runner
# ---------------------------------------------------------------------------

def run_matchup(agent_factories, n_games, rng_key,
                net=None, variables=None, sims=200):
    """Run a matchup with seat rotation.

    agent_factories: list of 3 callables returning Agent instances.
    Each rotation plays n_games. Total = 3 * n_games.
    If any agent is AlphaZeroAgent, uses batched GPU play.
    """
    agents_base = [f() for f in agent_factories]
    agent_names = [a.name for a in agents_base]
    has_az = any(isinstance(a, AlphaZeroAgent) for a in agents_base)

    wins = {name: 0 for name in agent_names}
    draws = 0
    seat_wins = [0, 0, 0]
    total_games = 0
    total_moves = 0

    for rotation in range(3):
        rotated_factories = (agent_factories[-rotation:] + agent_factories[:-rotation]
                             if rotation > 0 else list(agent_factories))
        rotated_names = (agent_names[-rotation:] + agent_names[:-rotation]
                         if rotation > 0 else list(agent_names))

        if has_az and net is not None:
            # --- Batched GPU play ---
            seat_agents = [f() for f in rotated_factories]
            rng_key, batch_rng = jax.random.split(rng_key)
            winners, move_counts = _play_games_batched(
                seat_agents, n_games, batch_rng, net, variables, sims)
            for g in range(n_games):
                total_games += 1
                total_moves += move_counts[g]
                w = int(winners[g])
                if w == -1:
                    draws += 1
                else:
                    wins[rotated_names[w]] = wins.get(rotated_names[w], 0) + 1
                    seat_wins[w] += 1
        else:
            # --- Sequential play for classical-only matchups ---
            for g in range(n_games):
                rng_key, game_key = jax.random.split(rng_key)
                agents = [f() for f in rotated_factories]
                winner, _scores, moves = play_one_game(agents, rng_key=game_key)
                total_games += 1
                total_moves += moves
                if winner == -1:
                    draws += 1
                else:
                    wins[rotated_names[winner]] = wins.get(rotated_names[winner], 0) + 1
                    seat_wins[winner] += 1

    return {
        "agent_names": agent_names,
        "wins": wins,
        "draws": draws,
        "total_games": total_games,
        "avg_moves": total_moves / total_games if total_games > 0 else 0,
        "seat_wins": seat_wins,
    }


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def run_benchmark(checkpoint_path, sims=200, n_games=50,
                  maxn_depth=4, paranoid_depth=6, shapley_depth=3,
                  output_dir="benchmark_results"):
    """Run full benchmark suite."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    variables, gen = load_from_pkl(checkpoint_path)
    net = create_network()
    print(f"  Generation: {gen}")

    # Warm up batched MCTS JIT (compile once, reuse for all matchups)
    print("Warming up batched MCTS JIT (this takes ~30s on first run)...")
    t0 = time.time()
    root_fn, sim_batch_fn, finish_fn, num_batches = _make_move_fns(net, n_games, sims)
    warmup_states = batched_init_states(n_games)
    warmup_rngs = jax.random.split(jax.random.PRNGKey(999), n_games)
    _, trees = root_fn(warmup_states, variables, warmup_rngs)
    trees = sim_batch_fn(trees, warmup_states, variables)
    action_rngs = jax.random.split(jax.random.PRNGKey(998), n_games)
    _ = finish_fn(trees, warmup_states, action_rngs, jnp.float32(0.01))
    jax.block_until_ready(trees.visit_count)
    print(f"  JIT warmup done in {time.time() - t0:.1f}s")

    # Agent factories
    def make_random(): return RandomAgent()
    def make_greedy(): return GreedyAgent()
    def make_maxn(): return MaxNAgent(depth=maxn_depth)
    def make_paranoid(): return ParanoidAgent(depth=paranoid_depth)
    def make_shapley(): return ShapleyAgent(depth=shapley_depth)
    def make_az(): return AlphaZeroAgent(sims=sims)

    classical_agents = [
        ("Random", make_random),
        ("Greedy", make_greedy),
        (f"MaxN(d={maxn_depth})", make_maxn),
        (f"Paranoid(d={paranoid_depth})", make_paranoid),
        (f"Shapley(d={shapley_depth})", make_shapley),
    ]

    all_results = {}
    rng = jax.random.PRNGKey(42)

    # --- AZ vs 2×Agent matchups ---
    print("\n=== AZ vs 2×Agent Matchups ===")
    az_vs_results = {}
    for name, factory in classical_agents:
        label = f"AZ vs 2×{name}"
        print(f"\n  {label} ({3 * n_games} games)...", flush=True)
        t0 = time.time()
        rng, sub = jax.random.split(rng)
        result = run_matchup([make_az, factory, factory], n_games, sub,
                             net=net, variables=variables, sims=sims)
        elapsed = time.time() - t0
        az_name = result["agent_names"][0]
        opp_name = result["agent_names"][1]
        total = result["total_games"]
        print(f"    {az_name}: {result['wins'].get(az_name, 0)}/{total} "
              f"({100 * result['wins'].get(az_name, 0) / total:.0f}%), "
              f"{opp_name}: {result['wins'].get(opp_name, 0)}/{total}, "
              f"draws: {result['draws']}/{total}  [{elapsed:.1f}s]")
        az_vs_results[name] = result
    all_results["az_vs"] = az_vs_results

    # --- Homogeneous matchups ---
    print("\n=== Homogeneous Matchups (3×same agent) ===")
    homo_results = {}
    all_agent_types = [("Random", make_random), ("Greedy", make_greedy),
                       (f"MaxN(d={maxn_depth})", make_maxn),
                       (f"Paranoid(d={paranoid_depth})", make_paranoid),
                       (f"Shapley(d={shapley_depth})", make_shapley)]
    for name, factory in all_agent_types:
        label = f"3×{name}"
        print(f"\n  {label} ({3 * n_games} games)...", flush=True)
        t0 = time.time()
        rng, sub = jax.random.split(rng)
        result = run_matchup([factory, factory, factory], n_games, sub)
        elapsed = time.time() - t0
        total = result["total_games"]
        print(f"    Seat wins: {result['seat_wins']}, "
              f"draws: {result['draws']}/{total}  [{elapsed:.1f}s]")
        homo_results[name] = result
    all_results["homogeneous"] = homo_results

    # --- Curated mixed triples ---
    print("\n=== Curated Mixed Triples ===")
    mixed_results = {}
    mixed_matchups = [
        ("AZ vs MaxN vs Paranoid", [make_az, make_maxn, make_paranoid]),
        ("AZ vs Greedy vs Random", [make_az, make_greedy, make_random]),
        ("AZ vs Shapley vs Paranoid", [make_az, make_shapley, make_paranoid]),
    ]
    for label, factories in mixed_matchups:
        print(f"\n  {label} ({3 * n_games} games)...", flush=True)
        t0 = time.time()
        rng, sub = jax.random.split(rng)
        result = run_matchup(factories, n_games, sub,
                             net=net, variables=variables, sims=sims)
        elapsed = time.time() - t0
        total = result["total_games"]
        wins_str = ", ".join(f"{n}: {w}" for n, w in result["wins"].items())
        print(f"    {wins_str}, draws: {result['draws']}/{total}  [{elapsed:.1f}s]")
        mixed_results[label] = result
    all_results["mixed"] = mixed_results

    # --- Generate plots ---
    print("\n=== Generating Plots ===")
    _plot_az_vs_bar(az_vs_results, output_path)
    _plot_pairwise_heatmap(all_results, output_path)
    _plot_seat_advantage(homo_results, output_path)

    # --- Save JSON summary ---
    summary = _build_summary(all_results)
    json_path = output_path / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}/")
    print(f"  benchmark_az_vs.png")
    print(f"  benchmark_heatmap.png")
    print(f"  benchmark_seat_advantage.png")
    print(f"  benchmark_summary.json")


def _build_summary(all_results):
    """Convert results to JSON-serializable dict."""
    summary = {}
    for section, data in all_results.items():
        summary[section] = {}
        for name, result in data.items():
            summary[section][name] = {
                "wins": result["wins"],
                "draws": result["draws"],
                "total_games": result["total_games"],
                "avg_moves": round(result["avg_moves"], 1),
                "seat_wins": result["seat_wins"],
            }
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_az_vs_bar(az_vs_results, output_path):
    """Plot 1: Grouped bar chart — AZ win rate vs opponent win rate vs draw rate."""
    opponents = list(az_vs_results.keys())
    az_rates = []
    opp_rates = []
    draw_rates = []

    for name, result in az_vs_results.items():
        total = result["total_games"]
        az_name = result["agent_names"][0]  # AZ is always first
        az_wins = result["wins"].get(az_name, 0)
        opp_wins = sum(w for n, w in result["wins"].items() if n != az_name)
        az_rates.append(az_wins / total)
        opp_rates.append(opp_wins / total)
        draw_rates.append(result["draws"] / total)

    x = np.arange(len(opponents))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, az_rates, width, label="AlphaZero", color="#2196F3")
    ax.bar(x, opp_rates, width, label="Opponent", color="#FF9800")
    ax.bar(x + width, draw_rates, width, label="Draw", color="#9E9E9E")

    ax.set_ylabel("Rate")
    ax.set_title("AlphaZero vs 2×Classical Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "benchmark_az_vs.png", dpi=150)
    plt.close(fig)
    print("  Saved benchmark_az_vs.png")


def _plot_pairwise_heatmap(all_results, output_path):
    """Plot 2: Pairwise win-rate matrix from all matchup data."""
    # Collect all unique agent names
    all_names = set()
    for section_data in all_results.values():
        for result in section_data.values():
            all_names.update(result["agent_names"])
    all_names = sorted(all_names)
    n = len(all_names)
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    # Win matrix: wins[i][j] = games agent i won when playing against agent j
    win_counts = np.zeros((n, n))
    game_counts = np.zeros((n, n))

    for section_data in all_results.values():
        for result in section_data.values():
            names = result["agent_names"]
            total = result["total_games"]
            unique_names = list(set(names))
            for a_name in unique_names:
                a_idx = name_to_idx[a_name]
                a_wins = result["wins"].get(a_name, 0)
                for b_name in unique_names:
                    if b_name == a_name:
                        continue
                    b_idx = name_to_idx[b_name]
                    win_counts[a_idx, b_idx] += a_wins
                    game_counts[a_idx, b_idx] += total

    # Compute win rates
    with np.errstate(divide="ignore", invalid="ignore"):
        win_rates = np.where(game_counts > 0, win_counts / game_counts, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(win_rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent (row)")
    ax.set_title("Pairwise Win Rate (row vs column)")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(win_rates[i, j]):
                text = f"{win_rates[i, j]:.0%}"
                color = "white" if win_rates[i, j] < 0.3 or win_rates[i, j] > 0.7 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path / "benchmark_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved benchmark_heatmap.png")


def _plot_seat_advantage(homo_results, output_path):
    """Plot 3: Per-seat win rate from homogeneous matchups."""
    agents = list(homo_results.keys())
    seat_data = np.zeros((len(agents), 3))

    for i, (name, result) in enumerate(homo_results.items()):
        total = result["total_games"]
        for s in range(3):
            seat_data[i, s] = result["seat_wins"][s] / total if total > 0 else 0

    x = np.arange(len(agents))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for s in range(3):
        label = f"Seat {s} (Player {s + 1})"
        colors = ["#4CAF50", "#2196F3", "#FF5722"]
        ax.bar(x + s * width - width, seat_data[:, s], width,
               label=label, color=colors[s])

    ax.set_ylabel("Win Rate")
    ax.set_title("First-Mover Advantage (Homogeneous 3×Agent)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, max(0.6, seat_data.max() * 1.2))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "benchmark_seat_advantage.png", dpi=150)
    plt.close(fig)
    print("  Saved benchmark_seat_advantage.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaZero vs classical agents on 4x5 connect-3")
    parser.add_argument("checkpoint", help="Path to .pkl checkpoint file")
    parser.add_argument("--sims", type=int, default=200,
                        help="MCTS simulations for AlphaZero (default: 200)")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per rotation (total = 3×games per matchup, default: 50)")
    parser.add_argument("--maxn-depth", type=int, default=4,
                        help="MaxN search depth (default: 4)")
    parser.add_argument("--paranoid-depth", type=int, default=6,
                        help="Paranoid search depth (default: 6)")
    parser.add_argument("--shapley-depth", type=int, default=3,
                        help="Shapley search depth (default: 3)")
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output directory for plots/JSON (default: benchmark_results)")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        sims=args.sims,
        n_games=args.games,
        maxn_depth=args.maxn_depth,
        paranoid_depth=args.paranoid_depth,
        shapley_depth=args.shapley_depth,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
