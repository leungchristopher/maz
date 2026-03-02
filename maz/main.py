"""Orchestration: self-play → train → repeat."""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np

# Persistent compilation cache — avoids re-compiling JIT kernels across runs
_default_cache = os.path.join(os.path.dirname(__file__), "..", ".jax_cache")
jax.config.update("jax_compilation_cache_dir",
                  os.environ.get("JAX_CACHE_DIR", os.path.abspath(_default_cache)))

from maz.network import create_network, init_params
from maz.selfplay import run_selfplay
from maz.train import ReplayBuffer, create_optimizer, train_on_buffer


# ---------- Config ----------
PEAK_LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 2048
EPOCHS_PER_WINDOW = 6
NUM_GAMES_PER_GEN = 4096
NUM_SIMULATIONS = 200
NUM_GENERATIONS = 200
TEMPERATURE = 1.0
SEED = 42
CHECKPOINT_DIR = "/content/drive/MyDrive/maz_checkpoints"
# ----------------------------


def main(checkpoint_dir=None, use_wandb=False, resume=True):
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    from maz.logger import Logger
    from maz.checkpoint import checkpoint_exists, load_checkpoint, save_checkpoint

    rng = jax.random.PRNGKey(SEED)

    # Initialize network
    net = create_network()
    rng, init_rng = jax.random.split(rng)
    variables = init_params(init_rng)

    # Estimate total training steps for LR schedule
    est_positions_per_gen = NUM_GAMES_PER_GEN * 20  # rough average game length
    est_steps_per_gen = max(1, (est_positions_per_gen // BATCH_SIZE) * EPOCHS_PER_WINDOW)
    total_steps = est_steps_per_gen * NUM_GENERATIONS
    optimizer = create_optimizer(PEAK_LR, WEIGHT_DECAY, total_steps)
    opt_state = optimizer.init(variables["params"])

    replay_buffer = ReplayBuffer(initial_capacity=6, max_capacity=30, grow_every=5)
    start_gen = 0

    # Resume from checkpoint
    if resume and checkpoint_exists(checkpoint_dir):
        print(f"Resuming from checkpoint in {checkpoint_dir}")
        variables, opt_state, replay_buffer, last_gen, rng = load_checkpoint(
            checkpoint_dir, variables, opt_state, rng,
        )
        start_gen = last_gen + 1
        print(f"Resumed at generation {start_gen}")

    # Logger
    config = {
        "peak_lr": PEAK_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs_per_window": EPOCHS_PER_WINDOW,
        "num_games_per_gen": NUM_GAMES_PER_GEN,
        "num_simulations": NUM_SIMULATIONS,
        "num_generations": NUM_GENERATIONS,
    }
    logger = Logger(use_wandb=use_wandb, config=config)

    print(f"Starting MAZ training: generations {start_gen}–{NUM_GENERATIONS - 1}, "
          f"{NUM_GAMES_PER_GEN} games/gen, {NUM_SIMULATIONS} sims/move, "
          f"batch_size={BATCH_SIZE}")
    print(f"Estimated {total_steps} total training steps")
    print()

    # Metric history for trend tracking
    history = {"value_loss": [], "policy_loss": [], "total_loss": [],
               "mean_length": [], "positions": [], "sp_time": [], "train_time": []}
    train_start = time.time()

    for gen in range(start_gen, NUM_GENERATIONS):
        gen_start = time.time()
        print(f"=== Generation {gen + 1}/{NUM_GENERATIONS} ===")

        # 1. Self-play
        sp_start = time.time()
        rng, sp_rng = jax.random.split(rng)
        games = run_selfplay(net, variables, sp_rng,
                             num_games=NUM_GAMES_PER_GEN,
                             num_simulations=NUM_SIMULATIONS,
                             temperature=TEMPERATURE)
        sp_elapsed = time.time() - sp_start

        lengths = [g.length for g in games]
        total_moves = sum(lengths)
        mean_len = total_moves / max(len(lengths), 1)
        draws = sum(1 for g in games if g.length > 0 and float(g.scores[0].sum()) == 0)
        decisive = len(games) - draws
        print(f"  Self-play: {len(games)} games, {total_moves} moves "
              f"(avg {mean_len:.1f}), {decisive} decisive / {draws} draws "
              f"[{sp_elapsed:.1f}s]")
        logger.log_selfplay(gen, games)

        history["mean_length"].append(mean_len)
        history["sp_time"].append(sp_elapsed)

        # 2. Add to replay buffer
        replay_buffer.add_generation(games)
        buf_positions = sum(
            g.length for gen_games in replay_buffer.generations for g in gen_games
        )
        print(f"  Buffer: {len(replay_buffer.generations)}/{replay_buffer.capacity} "
              f"generations, ~{buf_positions} positions")

        # 3. Train
        train_t0 = time.time()
        variables, opt_state, metrics = train_on_buffer(
            net, variables, optimizer, opt_state,
            replay_buffer,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_PER_WINDOW,
        )
        train_elapsed = time.time() - train_t0

        if metrics:
            v_loss = metrics["value_loss"]
            p_loss = metrics["policy_loss"]
            t_loss = metrics["total_loss"]
            history["value_loss"].append(v_loss)
            history["policy_loss"].append(p_loss)
            history["total_loss"].append(t_loss)
            history["positions"].append(metrics["num_positions"])
            history["train_time"].append(train_elapsed)

            # Show current + trend (last 5 gens)
            trend = ""
            if len(history["total_loss"]) >= 2:
                prev = np.mean(history["total_loss"][-6:-1])
                delta = t_loss - prev
                arrow = "v" if delta < 0 else "^"
                trend = f" ({arrow}{abs(delta):.4f})"
            print(f"  Train: loss={t_loss:.4f}{trend} "
                  f"(value={v_loss:.4f}, policy={p_loss:.4f}) "
                  f"| {metrics['num_positions']} pos, "
                  f"{metrics['num_steps']} steps [{train_elapsed:.1f}s]")
            logger.log_training(gen, metrics)

        # 4. Checkpoint
        try:
            save_checkpoint(checkpoint_dir, variables, opt_state,
                            replay_buffer, gen, rng)
        except Exception as e:
            print(f"  Warning: checkpoint save failed: {e}")

        gen_elapsed = time.time() - gen_start
        wall_elapsed = time.time() - train_start
        gens_done = gen - start_gen + 1
        gens_left = NUM_GENERATIONS - gen - 1
        eta = (wall_elapsed / gens_done) * gens_left if gens_done > 0 else 0
        print(f"  Gen time: {gen_elapsed:.1f}s | "
              f"Elapsed: {wall_elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
        print()

    # Final summary
    logger.finish()
    total_elapsed = time.time() - train_start
    print(f"Training complete! {NUM_GENERATIONS - start_gen} generations "
          f"in {total_elapsed/60:.1f}m")
    if history["total_loss"]:
        print(f"  Final loss: {history['total_loss'][-1]:.4f} "
              f"(value={history['value_loss'][-1]:.4f}, "
              f"policy={history['policy_loss'][-1]:.4f})")
        print(f"  Loss trend: {history['total_loss'][0]:.4f} -> "
              f"{history['total_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
