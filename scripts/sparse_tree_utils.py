"""
Sparse tree attention utilities for Scenario A.

Precomputes per-query ancestor indices from a static tree structure,
then gathers only the needed K/V for truly sparse tree attention.

Key idea: For a tree with N nodes and max depth D, each query token
only attends to its D+1 ancestors (including root and self), not all N
tree tokens. This reduces Stage 2 QK pairs from O(N^2) to O(N*D).

For regular_512 (binary tree, depth 8, 511 nodes including root):
  - Dense Stage 2: ~131K QK pairs (causal triangular)
  - Sparse Stage 2: ~4K QK pairs (ancestors only)
  - 32x reduction in tree-region compute
"""

import torch
from eagle_tree_choices import get_tree_stats


def precompute_ancestor_indices(tree_choices):
    """Precompute per-node ancestor index arrays from a static tree.

    Args:
        tree_choices: Sorted list of tuples representing tree node paths.
            E.g., [(0,), (1,), (0,0), (0,1), ...]. Root is implicit (index 0).

    Returns:
        ancestor_indices: [N, MAX_ANC] int64 tensor.
            Each row lists the tree-local KV indices of ancestors (including
            root=0 and self). Padded with 0 (root).
        ancestor_mask: [N, MAX_ANC] bool tensor.
            True for valid ancestor entries.
        ancestor_counts: [N] int32 tensor.
            Number of valid ancestors per node.

    Where N = len(tree_choices) + 1 (including root), and
    MAX_ANC = max_depth + 1 (root + all levels up to deepest node).
    """
    # Build node_to_idx map: root = 0, tree_choices[i] = i+1
    node_to_idx = {(): 0}  # root
    for i, node in enumerate(tree_choices):
        node_to_idx[node] = i + 1

    N = len(tree_choices) + 1  # total nodes including root
    stats = get_tree_stats(tree_choices)
    max_depth = stats["max_depth"]
    MAX_ANC = max_depth + 1  # root + max_depth levels

    ancestor_indices = torch.zeros(N, MAX_ANC, dtype=torch.int64)
    ancestor_mask = torch.zeros(N, MAX_ANC, dtype=torch.bool)
    ancestor_counts = torch.zeros(N, dtype=torch.int32)

    # Root (index 0): only ancestor is itself
    ancestor_indices[0, 0] = 0
    ancestor_mask[0, 0] = True
    ancestor_counts[0] = 1

    # Each tree node: ancestors = root + all prefixes + self
    for i, node in enumerate(tree_choices):
        idx = i + 1  # tree-local index (root=0, first node=1, ...)
        ancestors = [0]  # always includes root

        # Add all prefix ancestors: (node[0],), (node[0], node[1]), ...
        for prefix_len in range(1, len(node)):
            prefix = node[:prefix_len]
            ancestors.append(node_to_idx[prefix])

        # Add self
        ancestors.append(idx)

        n_anc = len(ancestors)
        ancestor_indices[idx, :n_anc] = torch.tensor(ancestors, dtype=torch.int64)
        ancestor_mask[idx, :n_anc] = True
        ancestor_counts[idx] = n_anc

    return ancestor_indices, ancestor_mask, ancestor_counts


def compute_dfs_permutation(tree_choices):
    """Compute DFS-order permutation so that subtree nodes are contiguous.

    In BFS order, nodes at the same depth are adjacent but share no ancestors.
    In DFS order, nodes in the same subtree are adjacent and share ancestor
    chains, making per-Q-block ancestor unions much tighter.

    Args:
        tree_choices: Sorted list of tuples representing tree node paths.
            Root is implicit (index 0).

    Returns:
        perm:     [N_tree] int64 tensor. perm[new_pos] = old_pos.
        inv_perm: [N_tree] int64 tensor. inv_perm[old_pos] = new_pos.
    """
    N_tree = len(tree_choices) + 1  # +1 for root

    # Build node_to_idx map: root = 0, tree_choices[i] = i+1
    node_to_idx = {(): 0}
    for i, node in enumerate(tree_choices):
        node_to_idx[node] = i + 1

    # Build parent → children adjacency list
    children = {i: [] for i in range(N_tree)}
    for i, node in enumerate(tree_choices):
        parent_path = node[:-1]  # e.g. (0,1,2) → parent (0,1)
        parent_idx = node_to_idx[parent_path]
        children[parent_idx].append(i + 1)

    # DFS traversal (iterative, left-to-right)
    perm = []
    stack = [0]  # start from root
    while stack:
        node_idx = stack.pop()
        perm.append(node_idx)
        # Push children in reverse so leftmost child is popped first
        for child in reversed(children[node_idx]):
            stack.append(child)

    perm = torch.tensor(perm, dtype=torch.int64)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(N_tree)

    return perm, inv_perm


def sparse_tree_attention(q, k, v, past_len, ancestor_indices, ancestor_mask,
                          scale):
    """Compute attention with sparse tree region using ancestor gather.

    Stage 1: Dense attention to past context [0, past_len).
    Stage 2: Sparse gather of only ancestor K/V in tree region.

    Args:
        q: [B, H, N_tree, D] query tensor (tree region queries).
        k: [B, H, N_KV, D] key tensor (past context + tree region).
        v: [B, H, N_KV, D] value tensor.
        past_len: Number of past context tokens.
        ancestor_indices: [N_tree, MAX_ANC] tree-local ancestor indices.
            These index into the tree region [0, N_tree).
        ancestor_mask: [N_tree, MAX_ANC] bool, True for valid entries.
        scale: Softmax scale factor.

    Returns:
        out: [B, H, N_tree, D] attention output.
    """
    B, H, N_tree, D = q.shape
    N_KV = k.shape[2]
    MAX_ANC = ancestor_indices.shape[1]

    # Move indices to same device
    ancestor_indices = ancestor_indices.to(q.device)
    ancestor_mask = ancestor_mask.to(q.device)

    # ---- Stage 1: Dense attention to past context ----
    if past_len > 0:
        k_past = k[:, :, :past_len, :]  # [B, H, past_len, D]
        v_past = v[:, :, :past_len, :]  # [B, H, past_len, D]

        # [B, H, N_tree, past_len]
        scores_past = torch.matmul(q, k_past.transpose(-2, -1)) * scale
    else:
        scores_past = q.new_empty(B, H, N_tree, 0)

    # ---- Stage 2: Sparse gather for tree ancestors only ----
    # Convert tree-local indices to global KV indices
    global_indices = ancestor_indices + past_len  # [N_tree, MAX_ANC]

    # Expand for batch/head dims: [1, 1, N_tree, MAX_ANC] -> broadcast
    idx_expanded = global_indices.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

    # Gather ancestor K/V: [B, H, N_tree, MAX_ANC, D]
    # Use gather on dim=2 (sequence dim)
    # First expand k/v index: need [B, H, N_tree*MAX_ANC, D] then reshape
    idx_flat = idx_expanded.reshape(B, H, N_tree * MAX_ANC, 1).expand(-1, -1, -1, D)
    k_tree = torch.gather(k, 2, idx_flat).reshape(B, H, N_tree, MAX_ANC, D)
    v_tree = torch.gather(v, 2, idx_flat).reshape(B, H, N_tree, MAX_ANC, D)

    # Compute QK for tree ancestors: [B, H, N_tree, MAX_ANC]
    scores_tree = torch.einsum('bhnd,bhnmd->bhnm', q, k_tree) * scale

    # Mask invalid ancestors
    anc_mask_expanded = ancestor_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_tree, MAX_ANC]
    scores_tree = scores_tree.masked_fill(~anc_mask_expanded, float('-inf'))

    # ---- Combined softmax + output ----
    # Concatenate scores: [B, H, N_tree, past_len + MAX_ANC]
    all_scores = torch.cat([scores_past, scores_tree], dim=-1)
    attn_weights = torch.softmax(all_scores, dim=-1)

    # Split attention weights
    w_past = attn_weights[:, :, :, :past_len]        # [B, H, N_tree, past_len]
    w_tree = attn_weights[:, :, :, past_len:]         # [B, H, N_tree, MAX_ANC]

    # Compute output
    if past_len > 0:
        out_past = torch.matmul(w_past, v_past)  # [B, H, N_tree, D]
    else:
        out_past = q.new_zeros(B, H, N_tree, D)

    out_tree = torch.einsum('bhnm,bhnmd->bhnd', w_tree, v_tree)  # [B, H, N_tree, D]

    return out_past + out_tree


def dense_tree_attention_reference(q, k, v, past_len, tree_choices, scale):
    """Reference dense tree attention for correctness comparison.

    Builds the full tree attention mask and computes standard attention.

    Args:
        q: [B, H, N_tree, D]
        k: [B, H, N_KV, D] where N_KV = past_len + N_tree
        v: [B, H, N_KV, D]
        past_len: Number of past context tokens.
        tree_choices: Sorted tree_choices list.
        scale: Softmax scale.

    Returns:
        out: [B, H, N_tree, D]
    """
    B, H, N_tree, D = q.shape
    N_KV = k.shape[2]

    # Compute full QK scores: [B, H, N_tree, N_KV]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Build attention mask
    # Past context: all visible (no masking needed)
    # Tree region: use tree_attn_mask

    # Build tree attention mask [N_tree, N_tree]
    tree_mask = torch.full((N_tree, N_tree), float('-inf'), device=q.device, dtype=q.dtype)

    # Build node_to_idx map
    node_to_idx = {(): 0}
    for i, node in enumerate(tree_choices):
        node_to_idx[node] = i + 1

    # Root (idx 0) attends to itself only
    tree_mask[0, 0] = 0.0

    # Each node attends to root, all ancestors, and self
    for i, node in enumerate(tree_choices):
        idx = i + 1
        # Attend to root
        tree_mask[idx, 0] = 0.0
        # Attend to all prefix ancestors
        for prefix_len in range(1, len(node)):
            prefix = node[:prefix_len]
            anc_idx = node_to_idx[prefix]
            tree_mask[idx, anc_idx] = 0.0
        # Attend to self
        tree_mask[idx, idx] = 0.0

    # Apply tree mask to tree region of scores
    # scores[:, :, :, past_len:] has shape [B, H, N_tree, N_tree]
    tree_mask_expanded = tree_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_tree, N_tree]
    scores[:, :, :, past_len:] = scores[:, :, :, past_len:] + tree_mask_expanded

    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    return out


def validate_sparse_vs_dense(tree_choices, past_len=64, B=2, H=4, D=128,
                              dtype=torch.float32, device='cuda'):
    """Validate sparse tree attention against dense reference.

    Args:
        tree_choices: Tree structure to test.
        past_len: Simulated past context length.
        B, H, D: Batch, heads, head dim.
        dtype: Data type (float32 for best numerical stability).
        device: Device.

    Returns:
        max_diff: Maximum absolute difference between sparse and dense outputs.
    """
    N_tree = len(tree_choices) + 1  # +1 for root
    N_KV = past_len + N_tree

    torch.manual_seed(42)
    q = torch.randn(B, H, N_tree, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N_KV, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N_KV, D, dtype=dtype, device=device)
    scale = 1.0 / (D ** 0.5)

    # Precompute ancestors
    ancestor_indices, ancestor_mask, ancestor_counts = precompute_ancestor_indices(tree_choices)

    # Sparse
    out_sparse = sparse_tree_attention(q, k, v, past_len, ancestor_indices,
                                       ancestor_mask, scale)

    # Dense reference
    out_dense = dense_tree_attention_reference(q, k, v, past_len, tree_choices,
                                               scale)

    max_diff = (out_sparse - out_dense).abs().max().item()
    mean_diff = (out_sparse - out_dense).abs().mean().item()

    stats = get_tree_stats(tree_choices)
    print(f"Tree: {stats['total_nodes']} nodes, depth {stats['max_depth']}")
    print(f"Past context: {past_len}")
    print(f"Shape: B={B}, H={H}, N_tree={N_tree}, D={D}")
    print(f"Max absolute diff: {max_diff:.2e}")
    print(f"Mean absolute diff: {mean_diff:.2e}")

    if max_diff < 1e-3:
        print("PASSED: Sparse matches dense (atol=1e-3)")
    elif max_diff < 1e-2:
        print("WARNING: Marginal match (1e-3 < diff < 1e-2)")
    else:
        print("FAILED: Outputs differ significantly")

    return max_diff


if __name__ == "__main__":
    import sys
    from eagle_tree_choices import EAGLE_TREES, regular_tree_512, mc_sim_7b_63

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # Run validation on multiple trees
        for name in ["mc_sim_7b_63", "regular_512"]:
            print(f"\n{'='*60}")
            print(f"Validating: {name}")
            print(f"{'='*60}")
            tree = EAGLE_TREES[name]
            validate_sparse_vs_dense(tree)

    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        # Print ancestor statistics
        for name in ["mc_sim_7b_63", "regular_256", "regular_512"]:
            tree = EAGLE_TREES[name]
            anc_idx, anc_mask, anc_counts = precompute_ancestor_indices(tree)
            stats = get_tree_stats(tree)
            N_tree = len(tree) + 1
            total_anc = int(anc_counts.sum().item())
            dense_pairs = N_tree * (N_tree + 1) // 2  # causal triangular
            print(f"\n{name}: {N_tree} nodes, depth {stats['max_depth']}")
            print(f"  Total ancestor pairs: {total_anc}")
            print(f"  Dense causal pairs:   {dense_pairs}")
            print(f"  Sparsity ratio:       {total_anc/dense_pairs:.4f} ({total_anc/dense_pairs*100:.1f}%)")
            print(f"  Avg ancestors/node:   {total_anc/N_tree:.1f}")
            print(f"  Max ancestors:        {int(anc_counts.max().item())}")
    elif len(sys.argv) > 1 and sys.argv[1] == "dfs":
        # Print DFS permutation info and subtree block stats
        for name in ["mc_sim_7b_63", "regular_512"]:
            tree = EAGLE_TREES[name]
            N_tree = len(tree) + 1
            perm, inv_perm = compute_dfs_permutation(tree)
            stats = get_tree_stats(tree)
            print(f"\n{name}: {N_tree} nodes, depth {stats['max_depth']}")
            print(f"  DFS first 20: {perm[:20].tolist()}")
            print(f"  DFS last 10:  {perm[-10:].tolist()}")

            # Compare block counts: BFS vs DFS ordering
            BLOCK = 32
            anc_idx, _, anc_counts = precompute_ancestor_indices(tree)
            num_q_blocks = (N_tree + BLOCK - 1) // BLOCK

            # BFS block counts
            bfs_total = 0
            for qb in range(num_q_blocks):
                qs, qe = qb * BLOCK, min((qb + 1) * BLOCK, N_tree)
                kv_set = set()
                for qp in range(qs, qe):
                    for a in range(int(anc_counts[qp].item())):
                        kv_set.add(int(anc_idx[qp, a].item()) // BLOCK)
                bfs_total += len(kv_set)

            # DFS block counts
            dfs_total = 0
            for qb in range(num_q_blocks):
                qs, qe = qb * BLOCK, min((qb + 1) * BLOCK, N_tree)
                kv_set = set()
                for new_q in range(qs, qe):
                    old_q = int(perm[new_q].item())
                    for a in range(int(anc_counts[old_q].item())):
                        old_anc = int(anc_idx[old_q, a].item())
                        new_anc = int(inv_perm[old_anc].item())
                        kv_set.add(new_anc // BLOCK)
                dfs_total += len(kv_set)

            dense_total = num_q_blocks * (num_q_blocks + 1) // 2
            print(f"  Total Stage 2 block loads:")
            print(f"    Dense (causal):     {dense_total}")
            print(f"    Sparse (BFS):       {bfs_total} ({(1-bfs_total/dense_total)*100:.0f}% reduction)")
            print(f"    Sparse (DFS/sub):   {dfs_total} ({(1-dfs_total/dense_total)*100:.0f}% reduction)")
            print(f"    DFS vs BFS:         {(1-dfs_total/bfs_total)*100:.0f}% fewer loads")

    else:
        print("Usage: python sparse_tree_utils.py [validate|stats|dfs]")
        print("  validate - Run correctness check (requires CUDA)")
        print("  stats    - Print ancestor statistics")
        print("  dfs      - Print DFS permutation and block load comparison")
