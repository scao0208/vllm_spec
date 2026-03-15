"""
EAGLE expanded tree structure definitions for Scenario A.

Generates tree structures with 10+ depth layers and 512+ total tokens,
mimicking how EAGLE builds speculative trees via top-K expansion at each level.
"""


def generate_eagle_tree(top_k, depth, total_tokens):
    """Generate EAGLE-style tree choices via breadth-first top-K expansion.

    Args:
        top_k: Maximum branching factor at each level.
        depth: Maximum tree depth.
        total_tokens: Target total number of tree nodes (excluding root).

    Returns:
        List of tuples representing tree node paths, sorted by (depth, path).
    """
    tree_choices = []

    # Level 1: top_k children of root
    level_nodes = []
    for k in range(top_k):
        if len(tree_choices) >= total_tokens:
            break
        node = (k,)
        tree_choices.append(node)
        level_nodes.append(node)

    # Subsequent levels: expand each parent with decaying branching factor
    for d in range(1, depth):
        if len(tree_choices) >= total_tokens:
            break

        next_level = []
        # Decay branching factor with depth to create realistic EAGLE shape
        # EAGLE focuses resources on likely paths (top-0 chain goes deepest)
        level_k = max(1, top_k - d)

        for parent in level_nodes:
            if len(tree_choices) >= total_tokens:
                break
            for k in range(level_k):
                if len(tree_choices) >= total_tokens:
                    break
                node = parent + (k,)
                tree_choices.append(node)
                next_level.append(node)

        level_nodes = next_level

    return tree_choices[:total_tokens]


def generate_deep_eagle_tree(top_k, depth, total_tokens):
    """Generate a deeper EAGLE tree that prioritizes depth over breadth.

    This is more representative of EAGLE-3's behavior where the most
    likely path (top-0 chain) goes very deep, with decreasing branching
    at each level.

    Strategy:
    - Level 1: top_k nodes
    - Level 2: top_k/2 children per parent (from level 1)
    - Level 3+: branching decays, but top-0 chain always extends
    - Remaining budget fills breadth at shallow levels
    """
    tree_choices = []
    level_nodes_by_depth = {}

    # Phase 1: Build the primary chain (top-0 at every level)
    chain = []
    for d in range(1, depth + 1):
        node = (0,) * d
        chain.append(node)

    # Phase 2: Build breadth at each level with decaying branching
    all_nodes = set()

    # Level 1: full top_k
    level_1 = [(k,) for k in range(top_k)]
    for n in level_1:
        all_nodes.add(n)
    level_nodes_by_depth[1] = level_1

    # Level 2+: decay branching, always include chain
    for d in range(2, depth + 1):
        level_k = max(1, top_k // d)
        new_level = []

        # Always include chain node
        chain_node = (0,) * d
        if chain_node not in all_nodes:
            new_level.append(chain_node)
            all_nodes.add(chain_node)

        # Expand parents from previous level
        prev_level = level_nodes_by_depth.get(d - 1, [])
        for parent in prev_level:
            if len(all_nodes) >= total_tokens:
                break
            for k in range(level_k):
                child = parent + (k,)
                if child not in all_nodes and len(all_nodes) < total_tokens:
                    new_level.append(child)
                    all_nodes.add(child)

        level_nodes_by_depth[d] = new_level

    # Sort by (depth, path) as required by vLLM
    tree_choices = sorted(all_nodes, key=lambda x: (len(x), x))
    return tree_choices[:total_tokens]


# Standard EAGLE tree from mc_sim_7b_63 (Monte Carlo optimized, 25 nodes, depth 5)
# Source: eagle/model/choices.py from EAGLE project
mc_sim_7b_63 = [
    (0,), (1,), (2,), (3,),
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0),
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (1, 0, 0),
    (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2),
    (0, 0, 0, 0, 0), (0, 0, 0, 0, 1),
]

# Pre-computed trees of various sizes
# ~64 tokens, 8 depth, top_k=6 (small baseline)
eagle_tree_64 = generate_deep_eagle_tree(top_k=6, depth=8, total_tokens=64)

# ~256 tokens, 12 depth, top_k=8
eagle_tree_256 = generate_deep_eagle_tree(top_k=8, depth=12, total_tokens=256)

# ~512 tokens, 14 depth, top_k=10
eagle_tree_512 = generate_deep_eagle_tree(top_k=10, depth=14, total_tokens=512)

# ~1024 tokens, 16 depth, top_k=12
eagle_tree_1024 = generate_deep_eagle_tree(top_k=12, depth=16, total_tokens=1024)

def generate_regular_tree(branching_factors):
    """Generate a regular tree with uniform branching at each level.

    This produces trees compatible with vLLM's propose_tree() which requires
    child_drafts_per_level[i] = num_nodes[i] / num_nodes[i-1] to be an integer.

    Args:
        branching_factors: List [k1, k2, ...] where ki is the number of
            children per parent at depth i. Tree depth = len(branching_factors).

    Returns:
        List of tuples representing tree node paths, sorted breadth-first.
    """
    tree = []
    parents = [()]  # virtual root
    for k in branching_factors:
        new_level = []
        for parent in parents:
            for c in range(k):
                node = parent + (c,)
                tree.append(node)
                new_level.append(node)
        parents = new_level
    return sorted(tree, key=lambda x: (len(x), x))


# Regular trees with uniform branching (compatible with propose_tree())
# regular_256: [4, 2, 2, 2, 2, 2] → 4+8+16+32+64+128 = 252 nodes, depth 6
regular_tree_256 = generate_regular_tree([4, 2, 2, 2, 2, 2])

# regular_512: [2, 2, 2, 2, 2, 2, 2, 2] → 2+4+8+16+32+64+128+256 = 510 nodes, depth 8
regular_tree_512 = generate_regular_tree([2, 2, 2, 2, 2, 2, 2, 2])


EAGLE_TREES = {
    "mc_sim_7b_63": mc_sim_7b_63,
    "generated_64": eagle_tree_64,
    "generated_256": eagle_tree_256,
    "generated_512": eagle_tree_512,
    "generated_1024": eagle_tree_1024,
    "regular_256": regular_tree_256,
    "regular_512": regular_tree_512,
}


def get_tree_stats(tree_choices):
    """Print statistics about a tree structure."""
    if not tree_choices:
        return {}
    depths = [len(node) for node in tree_choices]
    depth_counts = {}
    for d in depths:
        depth_counts[d] = depth_counts.get(d, 0) + 1
    return {
        "total_nodes": len(tree_choices),
        "max_depth": max(depths),
        "depth_counts": depth_counts,
    }


if __name__ == "__main__":
    for size, tree in EAGLE_TREES.items():
        stats = get_tree_stats(tree)
        print(f"\neagle_tree_{size}:")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Max depth: {stats['max_depth']}")
        print(f"  Nodes per depth:")
        for d in sorted(stats['depth_counts'].keys()):
            print(f"    Depth {d}: {stats['depth_counts'][d]}")
        # Show first few and last few nodes
        print(f"  First 5: {tree[:5]}")
        print(f"  Last 5: {tree[-5:]}")
