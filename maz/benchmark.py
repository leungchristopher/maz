import argparse
import itertools
import json
import math
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
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

CENTER_ORDER = [2, 1, 3, 0, 4]


def _enumerate_lines():
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

_LINES = _enumerate_lines()
_LINE_ROWS = [np.array(r, dtype=np.intp) for r, _ in _LINES]
_LINE_COLS = [np.array(c, dtype=np.intp) for _, c in _LINES]


def heuristic_eval(board_np):
    scores = [0.0, 0.0, 0.0]
    for rows, cols in zip(_LINE_ROWS, _LINE_COLS):
        cells = board_np[rows, cols]
        for p in range(3):
            player_val = p + 1
            own = np.sum(cells == player_val)
            others = np.sum((cells != 0) & (cells != player_val))
            if others > 0:
                continue
            if own == CONNECT_K:
                scores[p] += 10.0
            elif own == 2:
                scores[p] += 0.5
            elif own == 1:
                scores[p] += 0.1
    max_abs = max(abs(s) for s in scores) if any(s != 0 for s in scores) else 1.0
    if max_abs > 1.0:
        scores = [s / max_abs for s in scores]
    return scores


def _np_get_valid(board_np):
    return [c for c in CENTER_ORDER if board_np[0, c] == 0]


def _np_drop(board_np, col, player_val):
    b = board_np.copy()
    for r in range(ROWS - 1, -1, -1):
        if b[r, col] == 0:
            b[r, col] = player_val
            return b
    return b


def _np_check_win(board_np, player_val):
    for rows, cols in zip(_LINE_ROWS, _LINE_COLS):
        if np.all(board_np[rows, cols] == player_val):
            return True
    return False


def _np_is_full(board_np):
    return np.all(board_np != 0)


class _NumpyState:
    __slots__ = ('board', 'current_player', 'done', 'winner')

    def __init__(self, board, current_player):
        self.board = board
        self.current_player = current_player
        self.done = False
        self.winner = -1


def _agent_constructor_args(agent):
    if isinstance(agent, MaxNAgent):
        return (MaxNAgent, {'depth': agent.depth})
    if isinstance(agent, ParanoidAgent):
        return (ParanoidAgent, {'depth': agent.depth})
    if isinstance(agent, ShapleyAgent):
        return (ShapleyAgent, {'depth': agent.depth})
    if isinstance(agent, GreedyAgent):
        return (GreedyAgent, {})
    if isinstance(agent, RandomAgent):
        return (RandomAgent, {})
    raise ValueError(f"Unknown agent type: {type(agent)}")


def _select_action_worker(args):
    agent_cls, agent_kwargs, board_np, current_player = args
    agent = agent_cls(**agent_kwargs)
    state = _NumpyState(board_np, current_player)
    return agent.select_action(state)


class Agent(ABC):
    name: str

    @abstractmethod
    def select_action(self, state, rng_key=None):
        ...

    def __repr__(self):
        return self.name


class RandomAgent(Agent):
    name = "Random"

    def select_action(self, state, rng_key=None):
        board_np = np.asarray(state.board)
        cols = [c for c in range(COLS) if board_np[0, c] == 0]
        idx = np.random.randint(len(cols))
        return cols[idx]


class GreedyAgent(Agent):
    name = "Greedy"

    def select_action(self, state, rng_key=None):
        board_np = np.asarray(state.board)
        current = int(state.current_player)
        current_val = current + 1
        valid = _np_get_valid(board_np)

        for c in valid:
            b = _np_drop(board_np, c, current_val)
            if _np_check_win(b, current_val):
                return c

        for opp in range(NUM_PLAYERS):
            if opp == current:
                continue
            opp_val = opp + 1
            for c in valid:
                b = _np_drop(board_np, c, opp_val)
                if _np_check_win(b, opp_val):
                    return c

        for c in CENTER_ORDER:
            if c in valid:
                return c
        return valid[0]


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
        for c in valid:
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
        for c in valid:
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
        for c in valid:
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
            for c in valid:
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
            for c in valid:
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

        for c in valid:
            child = _np_drop(board_np, c, current + 1)
            if _np_check_win(child, current + 1):
                return c

        best_action = CENTER_ORDER[0]
        best_shapley = -float("inf")

        for c in valid:
            child = _np_drop(board_np, c, current + 1)
            shapley_vals = self._compute_shapley(child, (current + 1) % NUM_PLAYERS)
            if shapley_vals[current] > best_shapley:
                best_shapley = shapley_vals[current]
                best_action = c
        return best_action

    def _compute_shapley(self, board_np, next_player):
        players = [0, 1, 2]
        shapley = [0.0, 0.0, 0.0]

        for size in range(1, NUM_PLAYERS + 1):
            for coalition in itertools.combinations(players, size):
                coal_set = set(coalition)
                val = self._coalition_eval(board_np, next_player, coal_set, self.depth)
                for p in players:
                    if p in coal_set:
                        if len(coal_set) == 1:
                            marginal = val
                        else:
                            sub = coal_set - {p}
                            sub_val = self._coalition_eval(board_np, next_player,
                                                           sub, self.depth)
                            marginal = val - sub_val
                        n = NUM_PLAYERS
                        s = len(coal_set)
                        weight = math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
                        shapley[p] += weight * marginal
        return shapley

    def _coalition_eval(self, board_np, player, coalition, depth):
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
            val = sum(h[p] for p in coalition) / len(coalition)
            self._tt[key] = val
            return val

        valid = _np_get_valid(board_np)
        is_max = (player in coalition)

        if is_max:
            val = -float("inf")
            alpha, beta = -float("inf"), float("inf")
            for c in valid:
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
            for c in valid:
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


class AlphaZeroAgent(Agent):
    def __init__(self, sims=200):
        self.sims = sims
        self.name = f"AZ(s={sims})"

    def select_action(self, state, rng_key=None):
        raise RuntimeError("AlphaZeroAgent.select_action should not be called; "
                           "use batched play_games instead")


def _play_games_batched(seat_agents, n_games, rng_key, net, variables, sims,
                        pool=None):
    N = n_games
    max_moves = ROWS * COLS

    az_seats = {i for i, a in enumerate(seat_agents) if isinstance(a, AlphaZeroAgent)}

    root_fn, sim_batch_fn, finish_fn, num_batches = _make_move_fns(net, N, sims)
    v_step = jax.vmap(step)

    states = batched_init_states(N)
    done_np = np.zeros(N, dtype=bool)
    winners_np = np.full(N, -1, dtype=np.int32)
    move_counts_np = np.zeros(N, dtype=np.int32)
    action_histories = [[] for _ in range(N)]

    for move_num in range(max_moves):
        if done_np.all():
            break

        active = ~done_np
        current_seat = move_num % NUM_PLAYERS

        if current_seat in az_seats:
            rng_key, noise_rng, action_rng = jax.random.split(rng_key, 3)
            noise_rngs = jax.random.split(noise_rng, N)
            action_rngs = jax.random.split(action_rng, N)

            _, trees = root_fn(states, variables, noise_rngs)
            for _ in range(num_batches):
                trees = sim_batch_fn(trees, states, variables)
            _policies, actions, new_states = finish_fn(
                trees, states, action_rngs, jnp.float32(0.01))
        else:
            agent = seat_agents[current_seat]
            active_indices = [i for i in range(N) if active[i]]

            if pool is not None and len(active_indices) > 1:
                cls, kwargs = _agent_constructor_args(agent)
                boards_np = np.asarray(states.board)
                players_np = np.asarray(states.current_player)
                tasks = [
                    (cls, kwargs, boards_np[i], int(players_np[i]))
                    for i in active_indices
                ]
                results = list(pool.map(_select_action_worker, tasks))
                action_arr = np.zeros(N, dtype=np.int32)
                for j, i in enumerate(active_indices):
                    action_arr[i] = results[j]
                actions = jnp.array(action_arr, dtype=jnp.int32)
            else:
                action_list = []
                for i in range(N):
                    if active[i]:
                        single_state = jax.tree.map(lambda x, _i=i: x[_i], states)
                        rng_key, sub = jax.random.split(rng_key)
                        action_list.append(agent.select_action(single_state, rng_key=sub))
                    else:
                        action_list.append(0)
                actions = jnp.array(action_list, dtype=jnp.int32)

            new_states = v_step(states, actions)

        actions_np = np.array(actions)
        for i in range(N):
            if active[i]:
                action_histories[i].append(int(actions_np[i]))

        active_jnp = jnp.array(active)
        states = jax.tree.map(
            lambda new, old: jnp.where(
                active_jnp.reshape(-1, *([1] * (new.ndim - 1))), new, old),
            new_states, states)

        done_arr = np.array(states.done)
        winner_arr = np.array(states.winner)
        for i in range(N):
            if active[i] and done_arr[i] and not done_np[i]:
                done_np[i] = True
                winners_np[i] = winner_arr[i]
                move_counts_np[i] = move_num + 1

    for i in range(N):
        if not done_np[i]:
            done_np[i] = True
            move_counts_np[i] = max_moves

    return winners_np, move_counts_np, action_histories


def play_one_game(agents, rng_key=None):
    state = init_state()
    move = 0
    actions = []
    while not state.done:
        cp = int(state.current_player)
        agent = agents[cp]
        if rng_key is not None:
            rng_key, sub = jax.random.split(rng_key)
        else:
            sub = None
        action = agent.select_action(state, rng_key=sub)
        actions.append(int(action))
        state = step(state, jnp.int32(action))
        move += 1
    return int(state.winner), np.array(get_scores(state)), move, actions


def run_matchup(agent_factories, n_games, rng_key,
                net=None, variables=None, sims=200, pool=None):
    agents_base = [f() for f in agent_factories]
    raw_names = [a.name for a in agents_base]
    seen = {}
    agent_names = []
    for i, name in enumerate(raw_names):
        count = raw_names.count(name)
        if count > 1:
            idx = seen.get(name, 0)
            seen[name] = idx + 1
            agent_names.append(f"{name}[{i}]")
        else:
            agent_names.append(name)
    has_az = any(isinstance(a, AlphaZeroAgent) for a in agents_base)

    wins = {name: 0 for name in agent_names}
    draws = 0
    seat_wins = [0, 0, 0]
    total_games = 0
    total_moves = 0
    games = []

    for rotation in range(3):
        rotated_factories = (agent_factories[-rotation:] + agent_factories[:-rotation]
                             if rotation > 0 else list(agent_factories))
        rotated_names = (agent_names[-rotation:] + agent_names[:-rotation]
                         if rotation > 0 else list(agent_names))

        if has_az and net is not None:
            seat_agents = [f() for f in rotated_factories]
            rng_key, batch_rng = jax.random.split(rng_key)
            winners, move_counts, action_histories = _play_games_batched(
                seat_agents, n_games, batch_rng, net, variables, sims,
                pool=pool)
            for g in range(n_games):
                total_games += 1
                total_moves += move_counts[g]
                w = int(winners[g])
                if w == -1:
                    draws += 1
                else:
                    wins[rotated_names[w]] = wins.get(rotated_names[w], 0) + 1
                    seat_wins[w] += 1
                games.append({
                    "agents": list(rotated_names),
                    "actions": action_histories[g],
                    "winner": w,
                    "num_moves": int(move_counts[g]),
                })
        else:
            for g in range(n_games):
                rng_key, game_key = jax.random.split(rng_key)
                agents = [f() for f in rotated_factories]
                winner, _scores, moves, actions = play_one_game(agents, rng_key=game_key)
                total_games += 1
                total_moves += moves
                if winner == -1:
                    draws += 1
                else:
                    wins[rotated_names[winner]] = wins.get(rotated_names[winner], 0) + 1
                    seat_wins[winner] += 1
                games.append({
                    "agents": list(rotated_names),
                    "actions": actions,
                    "winner": winner,
                    "num_moves": moves,
                })

    return {
        "agent_names": agent_names,
        "wins": wins,
        "draws": draws,
        "total_games": total_games,
        "avg_moves": total_moves / total_games if total_games > 0 else 0,
        "seat_wins": seat_wins,
        "games": games,
    }


def run_benchmark(checkpoint_path, sims=200, n_games=50,
                  maxn_depth=3, paranoid_depth=6, shapley_depth=3,
                  output_dir="benchmark_results", n_workers=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 16)

    print(f"Loading checkpoint: {checkpoint_path}")
    variables, gen = load_from_pkl(checkpoint_path)
    net = create_network()
    print(f"  Generation: {gen}")

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

    print(f"\nUsing {n_workers} worker processes for classical agent turns")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:

        print("\n=== 2×AZ vs 1×Classical Matchups ===")
        az2_vs_results = {}
        for name, factory in classical_agents:
            label = f"2×AZ vs {name}"
            print(f"\n  {label} ({3 * n_games} games)...", flush=True)
            t0 = time.time()
            rng, sub = jax.random.split(rng)
            result = run_matchup([make_az, make_az, factory], n_games, sub,
                                 net=net, variables=variables, sims=sims,
                                 pool=pool)
            elapsed = time.time() - t0
            names = result["agent_names"]
            total = result["total_games"]
            wins_str = ", ".join(f"{n}: {result['wins'].get(n, 0)}" for n in names)
            print(f"    {wins_str}, draws: {result['draws']}/{total}  [{elapsed:.1f}s]")
            az2_vs_results[name] = result
        all_results["az2_vs"] = az2_vs_results

        print("\n=== 1×AZ vs 2×Classical Matchups ===")
        az1_vs_results = {}
        for name, factory in classical_agents:
            label = f"AZ vs 2×{name}"
            print(f"\n  {label} ({3 * n_games} games)...", flush=True)
            t0 = time.time()
            rng, sub = jax.random.split(rng)
            result = run_matchup([make_az, factory, factory], n_games, sub,
                                 net=net, variables=variables, sims=sims,
                                 pool=pool)
            elapsed = time.time() - t0
            names = result["agent_names"]
            total = result["total_games"]
            wins_str = ", ".join(f"{n}: {result['wins'].get(n, 0)}" for n in names)
            print(f"    {wins_str}, draws: {result['draws']}/{total}  [{elapsed:.1f}s]")
            az1_vs_results[name] = result
        all_results["az1_vs"] = az1_vs_results

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
                                 net=net, variables=variables, sims=sims,
                                 pool=pool)
            elapsed = time.time() - t0
            total = result["total_games"]
            wins_str = ", ".join(f"{n}: {w}" for n, w in result["wins"].items())
            print(f"    {wins_str}, draws: {result['draws']}/{total}  [{elapsed:.1f}s]")
            mixed_results[label] = result
        all_results["mixed"] = mixed_results

    print("\n=== Generating Plots ===")
    _plot_az_majority_bar(az2_vs_results, output_path)
    _plot_az_minority_bar(az1_vs_results, output_path)
    _plot_pairwise_heatmap(all_results, output_path)

    summary = _build_summary(all_results)
    json_path = output_path / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    games_log = _build_games_log(all_results)
    games_path = output_path / "benchmark_games.json"
    with open(games_path, "w") as f:
        json.dump(games_log, f, separators=(",", ":"))
    print(f"\nResults saved to {output_path}/")
    print(f"  benchmark_az_majority.png")
    print(f"  benchmark_az_minority.png")
    print(f"  benchmark_heatmap.png")
    print(f"  benchmark_summary.json")
    print(f"  benchmark_games.json ({len(games_log)} games)")


def _build_summary(all_results):
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


def _build_games_log(all_results):
    games = []
    for section, data in all_results.items():
        for matchup_name, result in data.items():
            for game in result.get("games", []):
                games.append({
                    "matchup": matchup_name,
                    "section": section,
                    **game,
                })
    return games


def _plot_az_majority_bar(az2_vs_results, output_path):
    opponents = list(az2_vs_results.keys())
    az0_rates = []
    az1_rates = []
    opp_rates = []
    draw_rates = []

    for name, result in az2_vs_results.items():
        total = result["total_games"]
        names = result["agent_names"]
        wins = result["wins"]
        az0_rates.append(wins.get(names[0], 0) / total)
        az1_rates.append(wins.get(names[1], 0) / total)
        opp_rates.append(wins.get(names[2], 0) / total)
        draw_rates.append(result["draws"] / total)

    x = np.arange(len(opponents))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, az0_rates, width, label="AZ seat 0", color="#1565C0")
    ax.bar(x - 0.5 * width, az1_rates, width, label="AZ seat 1", color="#42A5F5")
    ax.bar(x + 0.5 * width, opp_rates, width, label="Opponent", color="#FF9800")
    ax.bar(x + 1.5 * width, draw_rates, width, label="Draw", color="#9E9E9E")

    ax.set_ylabel("Win Rate")
    ax.set_title("2×AlphaZero vs 1×Classical Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "benchmark_az_majority.png", dpi=150)
    plt.close(fig)
    print("  Saved benchmark_az_majority.png")


def _plot_az_minority_bar(az1_vs_results, output_path):
    opponents = list(az1_vs_results.keys())
    az_rates = []
    classical_rates = []
    draw_rates = []

    for name, result in az1_vs_results.items():
        total = result["total_games"]
        names = result["agent_names"]
        wins = result["wins"]
        az_wins = wins.get(names[0], 0)
        classical_wins = sum(wins.get(n, 0) for n in names[1:])
        az_rates.append(az_wins / total)
        classical_rates.append(classical_wins / total)
        draw_rates.append(result["draws"] / total)

    x = np.arange(len(opponents))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, az_rates, width, label="AlphaZero", color="#1565C0")
    ax.bar(x, classical_rates, width, label="2×Classical (combined)", color="#FF9800")
    ax.bar(x + width, draw_rates, width, label="Draw", color="#9E9E9E")

    ax.set_ylabel("Win Rate")
    ax.set_title("1×AlphaZero vs 2×Classical Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "benchmark_az_minority.png", dpi=150)
    plt.close(fig)
    print("  Saved benchmark_az_minority.png")


def _plot_pairwise_heatmap(all_results, output_path):
    all_names = set()
    for section_data in all_results.values():
        for result in section_data.values():
            all_names.update(result["agent_names"])
    all_names = sorted(all_names)
    n = len(all_names)
    name_to_idx = {name: i for i, name in enumerate(all_names)}

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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaZero vs classical agents on 4x5 connect-3")
    parser.add_argument("checkpoint", help="Path to .pkl checkpoint file")
    parser.add_argument("--sims", type=int, default=200,
                        help="MCTS simulations for AlphaZero (default: 200)")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per rotation (total = 3×games per matchup, default: 50)")
    parser.add_argument("--maxn-depth", type=int, default=3,
                        help="MaxN search depth (default: 3)")
    parser.add_argument("--paranoid-depth", type=int, default=6,
                        help="Paranoid search depth (default: 6)")
    parser.add_argument("--shapley-depth", type=int, default=3,
                        help="Shapley search depth (default: 3)")
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU workers for classical agents (default: cpu_count, max 16)")
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
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
