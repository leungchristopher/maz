"""Checkpoint save/load for resumable training across Colab sessions.

Uses orbax-checkpoint for JAX pytrees and pickle for ReplayBuffer.
All imports are lazy so the base package works without orbax installed.
"""

import os
import pickle
import shutil


def save_checkpoint(ckpt_dir, variables, opt_state, replay_buffer,
                    generation, rng, keep=2):
    """Save training state to disk.

    Args:
        ckpt_dir: root checkpoint directory
        variables: network variables dict (params + batch_stats)
        opt_state: optimizer state
        replay_buffer: ReplayBuffer instance
        generation: current generation number (0-indexed)
        rng: JAX PRNGKey
        keep: number of recent checkpoints to keep
    """
    import orbax.checkpoint as ocp

    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"gen_{generation:05d}"

    # Save JAX pytree (variables, opt_state, rng, generation)
    pytree = {
        "variables": variables,
        "opt_state": opt_state,
        "rng": rng,
        "generation": generation,
    }
    tree_dir = os.path.join(ckpt_dir, tag)
    checkpointer = ocp.PyTreeCheckpointer()
    if os.path.exists(tree_dir):
        shutil.rmtree(tree_dir)
    checkpointer.save(tree_dir, pytree)

    # Pickle ReplayBuffer state
    buf_path = os.path.join(ckpt_dir, f"{tag}_buffer.pkl")
    buf_state = {
        "generations": list(replay_buffer.generations),
        "capacity": replay_buffer.capacity,
        "generation_count": replay_buffer.generation_count,
        "initial_capacity": replay_buffer.generations.maxlen
                            if hasattr(replay_buffer.generations, 'maxlen')
                            else None,
        "max_capacity": replay_buffer.max_capacity,
        "grow_every": replay_buffer.grow_every,
    }
    with open(buf_path, "wb") as f:
        pickle.dump(buf_state, f)

    # Update latest.txt
    with open(os.path.join(ckpt_dir, "latest.txt"), "w") as f:
        f.write(str(generation))

    # Cleanup old checkpoints
    _cleanup_old(ckpt_dir, generation, keep)


def _cleanup_old(ckpt_dir, current_gen, keep):
    """Remove checkpoints older than the most recent `keep`."""
    keep_gens = set(range(max(0, current_gen - keep + 1), current_gen + 1))
    for entry in os.listdir(ckpt_dir):
        if not entry.startswith("gen_"):
            continue
        # Parse generation number from "gen_NNNNN" or "gen_NNNNN_buffer.pkl"
        parts = entry.replace("_buffer.pkl", "").split("_")
        try:
            gen_num = int(parts[1])
        except (IndexError, ValueError):
            continue
        if gen_num not in keep_gens:
            path = os.path.join(ckpt_dir, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


def load_checkpoint(ckpt_dir, variables_template, opt_state_template,
                    rng_template):
    """Load training state from the latest checkpoint.

    Args:
        ckpt_dir: root checkpoint directory
        variables_template: variables with correct structure (for restore)
        opt_state_template: optimizer state with correct structure
        rng_template: PRNGKey template

    Returns:
        (variables, opt_state, replay_buffer, generation, rng)
    """
    import orbax.checkpoint as ocp
    from maz.train import ReplayBuffer

    latest_path = os.path.join(ckpt_dir, "latest.txt")
    with open(latest_path) as f:
        gen = int(f.read().strip())

    tag = f"gen_{gen:05d}"

    # Restore JAX pytree
    template = {
        "variables": variables_template,
        "opt_state": opt_state_template,
        "rng": rng_template,
        "generation": 0,
    }
    tree_dir = os.path.join(ckpt_dir, tag)
    checkpointer = ocp.PyTreeCheckpointer()
    pytree = checkpointer.restore(tree_dir, item=template)

    variables = pytree["variables"]
    opt_state = pytree["opt_state"]
    rng = pytree["rng"]
    generation = int(pytree["generation"])

    # Restore ReplayBuffer
    buf_path = os.path.join(ckpt_dir, f"{tag}_buffer.pkl")
    with open(buf_path, "rb") as f:
        buf_state = pickle.load(f)

    replay_buffer = ReplayBuffer(
        initial_capacity=buf_state.get("max_capacity", 20),
        max_capacity=buf_state["max_capacity"],
        grow_every=buf_state["grow_every"],
    )
    replay_buffer.capacity = buf_state["capacity"]
    replay_buffer.generation_count = buf_state["generation_count"]
    from collections import deque
    replay_buffer.generations = deque(buf_state["generations"])

    return variables, opt_state, replay_buffer, generation, rng


def checkpoint_exists(ckpt_dir):
    """Check if a checkpoint exists in the given directory."""
    return os.path.isfile(os.path.join(ckpt_dir, "latest.txt"))
