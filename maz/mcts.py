import functools

import chex
import jax
import jax.numpy as jnp

from maz.game import (
    COLS as NUM_ACTIONS,
    NUM_PLAYERS,
    GameState,
    step,
    get_valid_actions,
    encode_state,
    check_win,
)

MAX_NODES = 2048
C_PUCT = 2.0
DIRICHLET_ALPHA = 2.0
DIRICHLET_FRAC = 0.25
NUM_SIMULATIONS = 50
VIRTUAL_LOSS = 3.0


@chex.dataclass
class MCTSTree:
    visit_count: chex.Array
    value_sum: chex.Array
    prior: chex.Array
    children: chex.Array
    parent: chex.Array
    parent_action: chex.Array
    player: chex.Array
    virtual_loss: chex.Array
    valid_actions: chex.Array
    num_nodes: chex.Array


def batched_new_tree(n: int) -> MCTSTree:
    return MCTSTree(
        visit_count=jnp.zeros((n, MAX_NODES), dtype=jnp.int32),
        value_sum=jnp.zeros((n, MAX_NODES, NUM_PLAYERS), dtype=jnp.float32),
        prior=jnp.zeros((n, MAX_NODES, NUM_ACTIONS), dtype=jnp.float32),
        children=jnp.full((n, MAX_NODES, NUM_ACTIONS), -1, dtype=jnp.int32),
        parent=jnp.full((n, MAX_NODES), -1, dtype=jnp.int32),
        parent_action=jnp.full((n, MAX_NODES), -1, dtype=jnp.int32),
        player=jnp.zeros((n, MAX_NODES), dtype=jnp.int32),
        virtual_loss=jnp.zeros((n, MAX_NODES), dtype=jnp.float32),
        valid_actions=jnp.zeros((n, MAX_NODES, NUM_ACTIONS), dtype=jnp.bool_),
        num_nodes=jnp.zeros(n, dtype=jnp.int32),
    )


def expand_or_noop(tree, node_idx, state, prior, value, is_done):
    expanded = expand_node(tree, node_idx, state, prior, value)
    return jax.tree.map(
        lambda new, old: jnp.where(is_done, old, new),
        expanded, tree,
    )


def new_tree() -> MCTSTree:
    return MCTSTree(
        visit_count=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        value_sum=jnp.zeros((MAX_NODES, NUM_PLAYERS), dtype=jnp.float32),
        prior=jnp.zeros((MAX_NODES, NUM_ACTIONS), dtype=jnp.float32),
        children=jnp.full((MAX_NODES, NUM_ACTIONS), -1, dtype=jnp.int32),
        parent=jnp.full(MAX_NODES, -1, dtype=jnp.int32),
        parent_action=jnp.full(MAX_NODES, -1, dtype=jnp.int32),
        player=jnp.zeros(MAX_NODES, dtype=jnp.int32),
        virtual_loss=jnp.zeros(MAX_NODES, dtype=jnp.float32),
        valid_actions=jnp.zeros((MAX_NODES, NUM_ACTIONS), dtype=jnp.bool_),
        num_nodes=jnp.int32(0),
    )


def init_root(tree: MCTSTree, state: GameState,
              prior: chex.Array, rng: chex.PRNGKey) -> MCTSTree:
    valid = get_valid_actions(state)
    noise = jax.random.dirichlet(rng, jnp.full(NUM_ACTIONS, DIRICHLET_ALPHA))
    noisy_prior = (1 - DIRICHLET_FRAC) * prior + DIRICHLET_FRAC * noise
    noisy_prior = noisy_prior * valid
    noisy_prior = noisy_prior / (noisy_prior.sum() + 1e-8)

    tree = tree.replace(
        prior=tree.prior.at[0].set(noisy_prior),
        player=tree.player.at[0].set(state.current_player),
        valid_actions=tree.valid_actions.at[0].set(valid),
        num_nodes=jnp.int32(1),
    )
    return tree


def _ucb_score(tree: MCTSTree, node_idx: int, action: int) -> chex.Array:
    child_idx = tree.children[node_idx, action]
    has_child = child_idx >= 0

    parent_visits = tree.visit_count[node_idx].astype(jnp.float32)
    child_visits = jnp.where(has_child, tree.visit_count[child_idx].astype(jnp.float32), 0.0)
    child_vl = jnp.where(has_child, tree.virtual_loss[child_idx], 0.0)
    effective_visits = child_visits + child_vl

    player = tree.player[node_idx]
    child_value_sum = jnp.where(has_child, tree.value_sum[child_idx, player], 0.0)
    q = jnp.where(effective_visits > 0,
                   child_value_sum / effective_visits,
                   0.0)

    prior = tree.prior[node_idx, action]
    u = C_PUCT * prior * jnp.sqrt(parent_visits + 1) / (1 + effective_visits)

    valid = tree.valid_actions[node_idx, action]
    return jnp.where(valid, q + u, -jnp.inf)


def select_action(tree: MCTSTree, node_idx: int) -> chex.Array:
    scores = jax.vmap(lambda a: _ucb_score(tree, node_idx, a))(jnp.arange(NUM_ACTIONS))
    return jnp.argmax(scores)


def select_leaf(tree: MCTSTree, states: chex.Array) -> tuple:
    def cond_fn(carry):
        tree, node_idx, state, path, path_len = carry
        action = select_action(tree, node_idx)
        child = tree.children[node_idx, action]
        return child >= 0

    def body_fn(carry):
        tree, node_idx, state, path, path_len = carry
        action = select_action(tree, node_idx)
        child = tree.children[node_idx, action]

        tree = tree.replace(
            virtual_loss=tree.virtual_loss.at[child].add(VIRTUAL_LOSS)
        )

        new_state = step(state, action)

        path = path.at[path_len].set(child)
        path_len = path_len + 1

        return (tree, child, new_state, path, path_len)

    max_depth = 20
    path = jnp.full(max_depth, -1, dtype=jnp.int32)
    path = path.at[0].set(0)
    init_carry = (tree, jnp.int32(0), states, path, jnp.int32(1))

    tree, leaf_idx, leaf_state, path, path_len = jax.lax.while_loop(
        cond_fn, body_fn, init_carry
    )

    return tree, leaf_idx, leaf_state, path, path_len


def expand_node(tree: MCTSTree, node_idx: int, state: GameState,
                prior: chex.Array, value: chex.Array) -> MCTSTree:
    valid = get_valid_actions(state)
    masked_prior = prior * valid
    masked_prior = masked_prior / (masked_prior.sum() + 1e-8)

    tree = tree.replace(
        prior=tree.prior.at[node_idx].set(masked_prior),
        valid_actions=tree.valid_actions.at[node_idx].set(valid),
        player=tree.player.at[node_idx].set(state.current_player),
    )

    def alloc_child(carry, action):
        tree, next_id = carry
        is_valid = valid[action]
        child_state = step(state, action)
        child_valid = get_valid_actions(child_state)

        should_alloc = is_valid & (next_id < MAX_NODES)
        child_id = jnp.where(should_alloc, next_id, -1)
        new_next = jnp.where(should_alloc, next_id + 1, next_id)

        tree = tree.replace(
            children=tree.children.at[node_idx, action].set(child_id),
            parent=tree.parent.at[next_id].set(
                jnp.where(should_alloc, node_idx, tree.parent[next_id])
            ),
            parent_action=tree.parent_action.at[next_id].set(
                jnp.where(should_alloc, action, tree.parent_action[next_id])
            ),
            player=tree.player.at[next_id].set(
                jnp.where(should_alloc, child_state.current_player, tree.player[next_id])
            ),
            valid_actions=tree.valid_actions.at[next_id].set(
                jnp.where(should_alloc, child_valid, tree.valid_actions[next_id])
            ),
        )
        return (tree, new_next), None

    (tree, new_num_nodes), _ = jax.lax.scan(
        alloc_child, (tree, tree.num_nodes), jnp.arange(NUM_ACTIONS)
    )
    tree = tree.replace(num_nodes=new_num_nodes)
    return tree


def backpropagate(tree: MCTSTree, node_idx: int, value: chex.Array,
                  path: chex.Array, path_len: chex.Array) -> MCTSTree:
    def body_fn(i, tree):
        idx = path[path_len - 1 - i]
        is_valid = idx >= 0
        tree = tree.replace(
            visit_count=tree.visit_count.at[idx].add(
                jnp.where(is_valid, 1, 0)
            ),
            value_sum=tree.value_sum.at[idx].add(
                jnp.where(is_valid, value, jnp.zeros(NUM_PLAYERS))
            ),
            virtual_loss=tree.virtual_loss.at[idx].set(
                jnp.where(is_valid,
                           jnp.maximum(tree.virtual_loss[idx] - VIRTUAL_LOSS, 0.0),
                           tree.virtual_loss[idx])
            ),
        )
        return tree

    max_depth = 20
    tree = jax.lax.fori_loop(0, max_depth, body_fn, tree)
    return tree


def get_policy(tree: MCTSTree, temperature: float = 1.0) -> chex.Array:
    visits = jnp.zeros(NUM_ACTIONS, dtype=jnp.float32)

    def get_child_visits(a):
        child = tree.children[0, a]
        return jnp.where(child >= 0,
                         tree.visit_count[child].astype(jnp.float32),
                         0.0)

    visits = jax.vmap(get_child_visits)(jnp.arange(NUM_ACTIONS))

    use_argmax = temperature < 0.01
    logits = jnp.where(use_argmax, visits * 1e6, jnp.log(visits + 1e-8) / temperature)
    logits = jnp.where(visits > 0, logits, -jnp.inf)
    policy = jax.nn.softmax(logits)
    return policy


def search(state: GameState, net, variables, rng: chex.PRNGKey,
           num_simulations: int = NUM_SIMULATIONS,
           temperature: float = 1.0) -> chex.Array:
    tree = new_tree()

    obs = encode_state(state)[None]
    policy_logits, value = net.apply(variables, obs, train=False)
    root_prior = jax.nn.softmax(policy_logits[0])
    root_value = value[0]

    rng, noise_rng = jax.random.split(rng)
    tree = init_root(tree, state, root_prior, noise_rng)
    tree = expand_node(tree, 0, state, root_prior, root_value)
    tree = backpropagate(tree, 0, root_value,
                         jnp.array([0] + [-1] * 19, dtype=jnp.int32),
                         jnp.int32(1))

    for sim in range(num_simulations):
        rng, sim_rng = jax.random.split(rng)

        tree, leaf_idx, leaf_state, path, path_len = select_leaf(tree, state)

        leaf_done = leaf_state.done
        leaf_obs = encode_state(leaf_state)[None]
        leaf_logits, leaf_value = net.apply(variables, leaf_obs, train=False)
        leaf_prior = jax.nn.softmax(leaf_logits[0])
        leaf_val = leaf_value[0]

        from maz.game import get_scores
        actual_scores = get_scores(leaf_state)
        leaf_val = jnp.where(leaf_done, actual_scores, leaf_val)

        tree = jax.lax.cond(
            ~leaf_done,
            lambda t: expand_node(t, leaf_idx, leaf_state, leaf_prior, leaf_val),
            lambda t: t,
            tree,
        )

        tree = backpropagate(tree, leaf_idx, leaf_val, path, path_len)

    return get_policy(tree, temperature)


if __name__ == "__main__":
    from maz.game import init_state
    from maz.network import create_network, init_params

    rng = jax.random.PRNGKey(42)
    net = create_network()
    rng, init_rng = jax.random.split(rng)
    variables = init_params(init_rng)

    state = init_state()
    rng, search_rng = jax.random.split(rng)
    policy = search(state, net, variables, search_rng, num_simulations=10)
    print(f"MCTS policy: {policy}")
    print(f"Policy sum: {policy.sum()}")
    print("mcts test passed!")
