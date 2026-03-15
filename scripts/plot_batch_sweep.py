#!/usr/bin/env python3
"""Plot batch size sweep results: chain vs tree on HumanEval (64 prompts)."""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 11

# Data from batch_sweep_he.log
batch_chain = [2, 4, 8, 16, 32]
batch_tree  = [2, 4, 8, 16]  # 32 OOM

# ---- Throughput (tokens/s) ----
thru_chain = [66.30, 122.37, 205.34, 312.27, 405.84]
thru_tree  = [47.18,  63.06,  78.26,  85.64]

# ---- Iteration time (ms) ----
iter_chain = [216.09, 113.44, 67.60, 44.82, 34.18]
iter_tree  = [351.56, 262.27, 213.44, 194.99]

# ---- Per-iteration breakdown (ms) ----
core_chain = [0.0174, 0.0096, 0.0229, 0.0064, 0.0025]
core_tree  = [0.0493, 0.0437, 0.0309, 0.0267]
core80_chain = [x * 80 for x in core_chain]
core80_tree  = [x * 80 for x in core_tree]

attn_chain = [27.04, 16.48, 8.49, 4.43, 3.88]
mlp_chain  = [16.85,  9.29, 6.01, 5.65, 4.83]
other_chain= [172.20, 87.67, 53.10, 34.73, 25.47]

attn_tree  = [55.28, 53.90, 50.99, 49.16]
mlp_tree   = [100.88, 97.16, 94.44, 97.77]
other_tree = [195.39, 111.21, 68.01, 48.05]

# ---- Core attention % of total ----
core_pct_chain = [0.6, 0.7, 2.7, 1.1, 0.6]
core_pct_tree  = [1.1, 1.3, 1.2, 1.1]

# ---- Total time (s) ----
total_chain = [385.08, 209.40, 124.39, 82.01, 63.10]
total_tree  = [543.16, 411.24, 327.41, 302.81]

# ---- KV cache data (from CONTEXT vs TREE/CHAIN BREAKDOWN) ----
# Context (prefix) tokens, draft tokens, total KV tokens
ctx_chain  = [567, 1110, 2037, 3615, 6080]
draft_chain= [44,   86,  157,  280,  470]
kv_chain   = [610, 1196, 2194, 3896, 6551]

ctx_tree   = [588, 1170, 2250, 4348]
draft_tree = [973, 1910, 3725, 7159]
kv_tree    = [1561, 3080, 5975, 11506]

# KV cache size per token per GPU: 80 layers × 2 KV heads × 128 dim × 2 bytes × 2(K+V) = 81920 bytes = 80 KB
KB_PER_TOKEN = 80  # KB per token per GPU
kv_mb_chain = [t * KB_PER_TOKEN / 1024 for t in kv_chain]    # MB
kv_mb_tree  = [t * KB_PER_TOKEN / 1024 for t in kv_tree]
ctx_mb_chain = [t * KB_PER_TOKEN / 1024 for t in ctx_chain]
ctx_mb_tree  = [t * KB_PER_TOKEN / 1024 for t in ctx_tree]
draft_mb_chain = [t * KB_PER_TOKEN / 1024 for t in draft_chain]
draft_mb_tree  = [t * KB_PER_TOKEN / 1024 for t in draft_tree]

# Draft KV % of total KV
draft_kv_pct_chain = [d / t * 100 for d, t in zip(draft_chain, kv_chain)]
draft_kv_pct_tree  = [d / t * 100 for d, t in zip(draft_tree, kv_tree)]

# FLOP share: context vs draft (Q × KV attention FLOPs)
# context_flop_share = Q × context / (Q × total) = context / total (for each Q token)
# But actually from log: "Context region share" and "Tree/chain region share"
ctx_flop_pct_chain  = [92.8, 92.8, 92.8, 92.6, 92.5]
draft_flop_pct_chain= [7.2,  7.2,  7.2,  7.4,  7.5]
ctx_flop_pct_tree   = [37.6, 37.8, 37.4, 37.4]
draft_flop_pct_tree = [62.4, 62.2, 62.6, 62.6]

# Per-call latency (layer 0, single forward)
core_per_call_chain = [0.0333, 0.0363, 0.1562, 0.0778, 0.0503]
core_per_call_tree  = [0.0940, 0.1631, 0.2260, 0.3769]
attn_per_call_chain = [0.6477, 0.7761, 0.7234, 0.6760, 0.9953]
attn_per_call_tree  = [1.3181, 2.5154, 4.6556, 8.6763]
mlp_per_call_chain  = [0.4036, 0.4375, 0.5121, 0.8613, 1.2383]
mlp_per_call_tree   = [2.4054, 4.5343, 8.6234, 17.2543]


# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(18, 10), constrained_layout=True)
fig.suptitle('Batch Size Sweep: Chain vs Tree  (HumanEval, 64 prompts, Llama-3.1-70B, TP=4, FLy)',
             fontsize=14, fontweight='bold')
gs = fig.add_gridspec(2, 6)

# Row 1: Throughput, Iteration Time, Per-Iteration Breakdown
# Row 2: Core Attn %, KV Cache (stacked bar), Total Time

ax_thru  = fig.add_subplot(gs[0, 0:2])
ax_iter  = fig.add_subplot(gs[0, 2:4])
ax_break = fig.add_subplot(gs[0, 4:6])
ax_core  = fig.add_subplot(gs[1, 0:2])
ax_kv    = fig.add_subplot(gs[1, 2:4])
ax_total = fig.add_subplot(gs[1, 4:6])

BLUE = '#2196F3'
RED  = '#F44336'

# ---- 1. Throughput ----
ax = ax_thru
ax.plot(batch_chain, thru_chain, 'o-', color=BLUE, lw=2, ms=7, label='Chain')
ax.plot(batch_tree, thru_tree, 's-', color=RED, lw=2, ms=7, label='Tree')
for b, v in zip(batch_chain, thru_chain):
    ax.annotate(f'{v:.0f}', (b, v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
for b, v in zip(batch_tree, thru_tree):
    ax.annotate(f'{v:.0f}', (b, v), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
ax.set_xlabel('Batch Size'); ax.set_ylabel('Tokens/s')
ax.set_title('Throughput'); ax.set_xticks([2, 4, 8, 16, 32])
ax.legend(); ax.grid(True, alpha=0.3)

# ---- 2. Iteration Time ----
ax = ax_iter
ax.plot(batch_chain, iter_chain, 'o-', color=BLUE, lw=2, ms=7, label='Chain')
ax.plot(batch_tree, iter_tree, 's-', color=RED, lw=2, ms=7, label='Tree')
for b, v in zip(batch_chain, iter_chain):
    ax.annotate(f'{v:.0f}', (b, v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
for b, v in zip(batch_tree, iter_tree):
    ax.annotate(f'{v:.0f}', (b, v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
ax.set_xlabel('Batch Size'); ax.set_ylabel('ms')
ax.set_title('Iteration Time'); ax.set_xticks([2, 4, 8, 16, 32])
ax.legend(); ax.grid(True, alpha=0.3)

# ---- 3. Per-Iteration Breakdown (stacked bar, core attn in blue inside green) ----
ax = ax_break
width = 0.35
x_c = [i - width/2 for i in range(len(batch_chain))]
x_t = [i + width/2 for i in range(len(batch_tree))]

attn_proj_c = [a - c for a, c in zip(attn_chain, core80_chain)]
attn_proj_t = [a - c for a, c in zip(attn_tree, core80_tree)]
b1_c = other_chain
b2_c = [o + m for o, m in zip(other_chain, mlp_chain)]
b3_c = [b + p for b, p in zip(b2_c, attn_proj_c)]

ax.bar(x_c, other_chain, width, label='Other', color='#9E9E9E')
ax.bar(x_c, mlp_chain, width, bottom=b1_c, label='MLP/FFN', color='#FF9800')
ax.bar(x_c, attn_proj_c, width, bottom=b2_c, label='Attn Block (proj)', color='#4CAF50')
ax.bar(x_c, core80_chain, width, bottom=b3_c, label='Core Attention', color=BLUE)

b1_t = other_tree
b2_t = [o + m for o, m in zip(other_tree, mlp_tree)]
b3_t = [b + p for b, p in zip(b2_t, attn_proj_t)]
ax.bar(x_t, other_tree, width, color='#9E9E9E')
ax.bar(x_t, mlp_tree, width, bottom=b1_t, color='#FF9800')
ax.bar(x_t, attn_proj_t, width, bottom=b2_t, color='#4CAF50')
ax.bar(x_t, core80_tree, width, bottom=b3_t, color=BLUE)

all_batches = sorted(set(batch_chain + batch_tree))
ax.set_xticks(range(len(all_batches)))
ax.set_xticklabels([f'{b}\nC|T' for b in all_batches])
ax.set_xlabel('Batch Size (Chain|Tree)'); ax.set_ylabel('ms')
ax.set_title('Per-Iteration Breakdown')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

# ---- 4. Core Attention % of Total ----
ax = ax_core
ax.plot(batch_chain, core_pct_chain, 'o-', color=BLUE, lw=2, ms=7, label='Chain')
ax.plot(batch_tree, core_pct_tree, 's-', color=RED, lw=2, ms=7, label='Tree')
for b, v in zip(batch_chain, core_pct_chain):
    ax.annotate(f'{v:.1f}%', (b, v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
for b, v in zip(batch_tree, core_pct_tree):
    ax.annotate(f'{v:.1f}%', (b, v), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
ax.set_xlabel('Batch Size'); ax.set_ylabel('%')
ax.set_title('Core Attention % of Total Time')
ax.set_xticks([2, 4, 8, 16, 32])
ax.set_ylim(0, max(max(core_pct_chain), max(core_pct_tree)) * 2.0)
ax.legend(); ax.grid(True, alpha=0.3)

# ---- 5. KV Cache: context vs draft (stacked bar, MB per GPU) ----
ax = ax_kv
width = 0.35
x_c = [i - width/2 for i in range(len(batch_chain))]
x_t = [i + width/2 for i in range(len(batch_tree))]

ax.bar(x_c, ctx_mb_chain, width, label='Context KV (Chain)', color='#64B5F6')
ax.bar(x_c, draft_mb_chain, width, bottom=ctx_mb_chain, label='Draft KV (Chain)', color='#1565C0')

ax.bar(x_t, ctx_mb_tree, width, label='Context KV (Tree)', color='#EF9A9A')
ax.bar(x_t, draft_mb_tree, width, bottom=ctx_mb_tree, label='Draft KV (Tree)', color='#B71C1C')

# Annotate total MB
for x, total in zip(x_c, kv_mb_chain):
    ax.annotate(f'{total:.0f}', (x, total), textcoords="offset points", xytext=(0, 4), ha='center', fontsize=8)
for x, total in zip(x_t, kv_mb_tree):
    ax.annotate(f'{total:.0f}', (x, total), textcoords="offset points", xytext=(0, 4), ha='center', fontsize=8)

ax.set_xticks(range(len(all_batches)))
ax.set_xticklabels([f'{b}\nC|T' for b in all_batches])
ax.set_xlabel('Batch Size (Chain|Tree)'); ax.set_ylabel('MB per GPU')
ax.set_title('KV Cache Size (Context + Draft)')
ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3, axis='y')

# ---- 6. Total Time ----
ax = ax_total
ax.plot(batch_chain, total_chain, 'o-', color=BLUE, lw=2, ms=7, label='Chain')
ax.plot(batch_tree, total_tree, 's-', color=RED, lw=2, ms=7, label='Tree')
for b, v in zip(batch_chain, total_chain):
    ax.annotate(f'{v:.0f}s', (b, v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
for b, v in zip(batch_tree, total_tree):
    ax.annotate(f'{v:.0f}s', (b, v), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
ax.set_xlabel('Batch Size'); ax.set_ylabel('seconds')
ax.set_title('Total Time (64 prompts)')
ax.set_xticks([2, 4, 8, 16, 32])
ax.legend(); ax.grid(True, alpha=0.3)

plt.savefig('batch_sweep_he.png', dpi=150, bbox_inches='tight')
print('Saved batch_sweep_he.png')
