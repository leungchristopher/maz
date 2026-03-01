"""Orchestration: self-play → train → repeat."""

import jax
import jax.numpy as jnp

from maz.network import create_network, init_params
from maz.selfplay import run_selfplay
from maz.train import ReplayBuffer, create_optimizer, train_on_buffer


# ---------- Config ----------
PEAK_LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS_PER_WINDOW = 2
NUM_GAMES_PER_GEN = 128
NUM_SIMULATIONS = 50
NUM_GENERATIONS = 100
TEMPERATURE = 1.0
SEED = 42
CHECKPOINT_DIR = "/content/drive/MyDrive/maz_checkpoints"
# ----------------------------


def main(checkpoint_dir=None, use_wandb=False, resume=True):
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR

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

    replay_buffer = ReplayBuffer(initial_capacity=4, max_capacity=20, grow_every=5)
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
          f"{NUM_GAMES_PER_GEN} games/gen, {NUM_SIMULATIONS} sims/move")
    print(f"Estimated {total_steps} total training steps")
    print()

    gen_iter = range(start_gen, NUM_GENERATIONS)
    if tqdm is not None:
        gen_iter = tqdm(gen_iter, desc="Generations", unit="gen",
                        initial=start_gen, total=NUM_GENERATIONS)

    for gen in gen_iter:
        print(f"=== Generation {gen + 1}/{NUM_GENERATIONS} ===")

        # 1. Self-play
        print("Self-play...")
        rng, sp_rng = jax.random.split(rng)
        games = run_selfplay(net, variables, sp_rng,
                             num_games=NUM_GAMES_PER_GEN,
                             num_simulations=NUM_SIMULATIONS,
                             temperature=TEMPERATURE)

        total_moves = sum(g.length for g in games)
        print(f"  Generated {len(games)} games, {total_moves} total moves")
        logger.log_selfplay(gen, games)

        # 2. Add to replay buffer
        replay_buffer.add_generation(games)
        print(f"  Buffer: {len(replay_buffer.generations)} generations, "
              f"capacity {replay_buffer.capacity}")

        # 3. Train
        print("Training...")
        variables, opt_state, metrics = train_on_buffer(
            net, variables, optimizer, opt_state,
            replay_buffer,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_PER_WINDOW,
        )

        if metrics:
            print(f"  Loss: {metrics['total_loss']:.4f} "
                  f"(value: {metrics['value_loss']:.4f}, "
                  f"policy: {metrics['policy_loss']:.4f})")
            print(f"  Trained on {metrics['num_positions']} positions, "
                  f"{metrics['num_steps']} steps")
            logger.log_training(gen, metrics)

        # 4. Checkpoint
        try:
            save_checkpoint(checkpoint_dir, variables, opt_state,
                            replay_buffer, gen, rng)
            print(f"  Checkpoint saved to {checkpoint_dir}")
        except Exception as e:
            print(f"  Warning: checkpoint save failed: {e}")

        print()

    logger.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
