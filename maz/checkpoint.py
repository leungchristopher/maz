import os
import pickle
import shutil
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp


def _pytree_to_numpy(pytree):
    return jax.tree.map(
        lambda x: x.__array__() if isinstance(x, jnp.ndarray) else x,
        pytree,
    )


def _numpy_to_jax(pytree):
    return jax.tree.map(
        lambda x: jnp.asarray(x) if hasattr(x, '__array__') else x,
        pytree,
    )


def save_checkpoint(ckpt_dir, variables, opt_state, replay_buffer,
                    generation, rng, keep=2):
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"gen_{generation:05d}"

    state = {
        "variables": _pytree_to_numpy(variables),
        "opt_state": _pytree_to_numpy(opt_state),
        "rng": _pytree_to_numpy(rng),
        "generation": generation,
        "buffer": {
            "generations": [
                (np.asarray(g[0]), np.asarray(g[1]), np.asarray(g[2]))
                if g is not None else None
                for g in replay_buffer.generations
            ],
            "raw_games": [
                [g._replace(
                    states=g.states.__array__() if isinstance(g.states, jnp.ndarray) else g.states,
                    policies=g.policies.__array__() if isinstance(g.policies, jnp.ndarray) else g.policies,
                    scores=g.scores.__array__() if isinstance(g.scores, jnp.ndarray) else g.scores,
                ) for g in gen_games]
                for gen_games in replay_buffer._raw_games
            ],
            "capacity": replay_buffer.capacity,
            "generation_count": replay_buffer.generation_count,
            "max_capacity": replay_buffer.max_capacity,
            "grow_every": replay_buffer.grow_every,
        },
    }

    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pkl")
    tmp_path = ckpt_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, ckpt_path)

    with open(os.path.join(ckpt_dir, "latest.txt"), "w") as f:
        f.write(str(generation))

    _cleanup_old(ckpt_dir, generation, keep)


def _cleanup_old(ckpt_dir, current_gen, keep):
    keep_gens = set(range(max(0, current_gen - keep + 1), current_gen + 1))
    for entry in os.listdir(ckpt_dir):
        if not entry.startswith("gen_"):
            continue
        parts = entry.replace("_buffer.pkl", "").replace(".pkl", "").split("_")
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
    from maz.train import ReplayBuffer
    from maz.selfplay import GameRecord

    latest_path = os.path.join(ckpt_dir, "latest.txt")
    with open(latest_path) as f:
        gen = int(f.read().strip())

    tag = f"gen_{gen:05d}"
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pkl")
    with open(ckpt_path, "rb") as f:
        state = pickle.load(f)

    variables = _numpy_to_jax(state["variables"])
    opt_state = _numpy_to_jax(state["opt_state"])
    rng = _numpy_to_jax(state["rng"])
    generation = int(state["generation"])

    buf = state["buffer"]
    replay_buffer = ReplayBuffer(
        initial_capacity=buf.get("max_capacity", 15),
        max_capacity=buf["max_capacity"],
        grow_every=buf["grow_every"],
    )
    replay_buffer.capacity = buf["capacity"]
    replay_buffer.generation_count = buf["generation_count"]
    replay_buffer.generations = deque(buf["generations"])
    replay_buffer._raw_games = deque(
        [[GameRecord(
            states=jnp.asarray(g.states),
            policies=jnp.asarray(g.policies),
            scores=jnp.asarray(g.scores),
            length=g.length,
        ) for g in gen_games]
         for gen_games in buf.get("raw_games", buf["generations"])]
    )

    return variables, opt_state, replay_buffer, generation, rng


def checkpoint_exists(ckpt_dir):
    return os.path.isfile(os.path.join(ckpt_dir, "latest.txt"))
