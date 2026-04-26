import functools

import chex
import jax
import jax.numpy as jnp

ROWS = 4
COLS = 5
NUM_PLAYERS = 3
CONNECT_K = 3


@chex.dataclass
class GameState:
    board: chex.Array
    current_player: chex.Array
    done: chex.Array
    winner: chex.Array


@jax.jit
def init_state() -> GameState:
    return GameState(
        board=jnp.zeros((ROWS, COLS), dtype=jnp.int8),
        current_player=jnp.int32(0),
        done=jnp.bool_(False),
        winner=jnp.int32(-1),
    )


@jax.jit
def get_valid_actions(state: GameState) -> chex.Array:
    return state.board[0] == 0


def _check_direction(board: chex.Array, player_val: chex.Array,
                     dr: int, dc: int) -> chex.Array:
    rows = jnp.arange(ROWS)
    cols = jnp.arange(COLS)
    r_starts, c_starts = jnp.meshgrid(rows, cols, indexing="ij")
    r_starts = r_starts.reshape(-1)
    c_starts = c_starts.reshape(-1)

    def check_one(idx):
        r0 = r_starts[idx]
        c0 = c_starts[idx]
        offsets = jnp.arange(CONNECT_K)
        rs = r0 + offsets * dr
        cs = c0 + offsets * dc
        in_bounds = (
            (rs >= 0).all() & (rs < ROWS).all() &
            (cs >= 0).all() & (cs < COLS).all()
        )
        vals = board[rs % ROWS, cs % COLS]
        all_match = (vals == player_val).all()
        return in_bounds & all_match

    indices = jnp.arange(ROWS * COLS)
    results = jax.vmap(check_one)(indices)
    return results.any()


@jax.jit
def check_win(board: chex.Array, player: chex.Array) -> chex.Array:
    player_val = (player + 1).astype(jnp.int8)
    horiz = _check_direction(board, player_val, 0, 1)
    vert = _check_direction(board, player_val, 1, 0)
    diag1 = _check_direction(board, player_val, 1, 1)
    diag2 = _check_direction(board, player_val, 1, -1)
    return horiz | vert | diag1 | diag2


@jax.jit
def step(state: GameState, action: chex.Array) -> GameState:
    board = state.board
    player = state.current_player
    player_val = (player + 1).astype(jnp.int8)

    col = board[:, action]
    empty_mask = (col == 0)
    row_indices = jnp.arange(ROWS)
    candidates = jnp.where(empty_mask, row_indices, -1)
    row = jnp.argmax(candidates)

    valid = empty_mask.any()
    new_board = board.at[row, action].set(
        jnp.where(valid, player_val, board[row, action])
    )

    won = check_win(new_board, player)
    board_full = (new_board != 0).all()
    new_done = state.done | won | board_full
    new_winner = jnp.where(state.done, state.winner,
                           jnp.where(won, player, jnp.int32(-1)))
    next_player = jnp.where(new_done, state.current_player,
                            (player + 1) % NUM_PLAYERS)

    return GameState(
        board=new_board,
        current_player=next_player,
        done=new_done,
        winner=new_winner,
    )


@jax.jit
def get_scores(state: GameState) -> chex.Array:
    winner = state.winner
    is_draw = (winner == -1)
    scores = jnp.where(
        is_draw,
        jnp.zeros(NUM_PLAYERS, dtype=jnp.float32),
        jnp.where(
            jnp.arange(NUM_PLAYERS) == winner,
            jnp.ones(NUM_PLAYERS, dtype=jnp.float32),
            -jnp.ones(NUM_PLAYERS, dtype=jnp.float32),
        ),
    )
    return scores


@jax.jit
def encode_state(state: GameState) -> chex.Array:
    board = state.board
    piece_planes = jnp.stack([
        (board == 1).astype(jnp.float32),
        (board == 2).astype(jnp.float32),
        (board == 3).astype(jnp.float32),
    ], axis=-1)

    turn_planes = jnp.stack([
        jnp.full((ROWS, COLS), (state.current_player == 0).astype(jnp.float32)),
        jnp.full((ROWS, COLS), (state.current_player == 1).astype(jnp.float32)),
        jnp.full((ROWS, COLS), (state.current_player == 2).astype(jnp.float32)),
    ], axis=-1)

    return jnp.concatenate([piece_planes, turn_planes], axis=-1)


def test_game():
    state = init_state()
    assert state.board.shape == (ROWS, COLS)
    assert not state.done

    s = state
    for col in [0, 1, 0, 1, 0, 1, 0]:
        if s.done:
            break
        s = step(s, jnp.int32(col))

    assert s.board.shape == (ROWS, COLS)

    obs = encode_state(state)
    assert obs.shape == (ROWS, COLS, 6)

    valid = get_valid_actions(state)
    assert valid.shape == (COLS,)
    assert valid.all()

    s = state
    for _ in range(ROWS * NUM_PLAYERS):
        if s.done:
            break
        s = step(s, jnp.int32(0))

    print("test_game passed!")


if __name__ == "__main__":
    test_game()
