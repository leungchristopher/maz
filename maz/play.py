import sys
import pickle
import argparse

import jax
import jax.numpy as jnp

from maz.game import (
    ROWS, COLS, NUM_PLAYERS, COLS as NUM_ACTIONS,
    GameState, init_state, step, get_valid_actions, get_scores, encode_state,
)
from maz.mcts import search, NUM_SIMULATIONS
from maz.network import create_network


PIECE_CHARS = {0: ".", 1: "X", 2: "O", 3: "#"}
PLAYER_NAMES = ["Player 1 (X)", "Player 2 (O)", "Player 3 (#)"]


def load_from_pkl(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    variables = jax.tree.map(jnp.asarray, state["variables"])
    gen = state.get("generation", "?")
    return variables, gen


def print_board(state):
    board = state.board
    print()
    print("  " + " ".join(str(i) for i in range(COLS)))
    print("  " + "-" * (COLS * 2 - 1))
    for r in range(ROWS):
        row_str = " ".join(PIECE_CHARS[int(board[r, c])] for c in range(COLS))
        print(f"  {row_str}")
    print("  " + "-" * (COLS * 2 - 1))
    print()


def get_human_action(state):
    valid = get_valid_actions(state)
    valid_cols = [c for c in range(COLS) if valid[c]]
    while True:
        try:
            col = int(input(f"  Your move (columns {valid_cols}): "))
            if col in valid_cols:
                return col
            print(f"  Column {col} is not valid. Try again.")
        except (ValueError, EOFError):
            print("  Enter a column number.")


def ai_move(state, net, variables, rng, num_sims, temperature=0.01):
    rng, search_rng = jax.random.split(rng)
    policy = search(state, net, variables, search_rng,
                    num_simulations=num_sims, temperature=temperature)
    action = int(jnp.argmax(policy))
    return action, policy, rng


def play_game(variables, human_player=0, num_sims=NUM_SIMULATIONS):
    net = create_network()
    rng = jax.random.PRNGKey(42)
    state = init_state()

    print("\n=== MAZ: 3-Player Connect-3 ===")
    print(f"  You are {PLAYER_NAMES[human_player]}")
    print(f"  AI simulations: {num_sims}")
    print(f"  Pieces: X=P1  O=P2  #=P3")
    print_board(state)

    move_num = 0
    while not state.done:
        cp = int(state.current_player)
        print(f"--- Move {move_num + 1}: {PLAYER_NAMES[cp]} ---")

        if cp == human_player:
            action = get_human_action(state)
        else:
            print("  AI thinking...", end="", flush=True)
            action, policy, rng = ai_move(state, net, variables, rng, num_sims)
            policy_str = " ".join(f"{p:.0%}" for p in policy)
            print(f" col {action}  (policy: [{policy_str}])")

        state = step(state, jnp.int32(action))
        print_board(state)
        move_num += 1

    winner = int(state.winner)
    if winner == -1:
        print("=== DRAW ===")
    else:
        print(f"=== {PLAYER_NAMES[winner]} WINS! ===")
        if winner == human_player:
            print("  Congratulations!")
        else:
            print("  Better luck next time!")

    scores = get_scores(state)
    print(f"  Scores: {[f'{s:+.0f}' for s in scores]}")


def main():
    parser = argparse.ArgumentParser(description="Play against trained MAZ agent")
    parser.add_argument("checkpoint", help="Path to .pkl checkpoint file")
    parser.add_argument("--player", type=int, default=1, choices=[1, 2, 3],
                        help="Which player you are (1-3, default: 1)")
    parser.add_argument("--sims", type=int, default=200,
                        help="MCTS simulations per AI move (default: 200)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    variables, gen = load_from_pkl(args.checkpoint)
    print(f"  Generation: {gen}")

    human_player = args.player - 1

    while True:
        play_game(variables, human_player=human_player, num_sims=args.sims)
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
