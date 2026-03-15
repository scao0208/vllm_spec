# Deep Analysis: HiSpec — Hierarchical Speculative Decoding for LLMs

**Paper:** arXiv 2510.01336 | Authors: Avinash Kumar, Sujay Sanghavi, Poulami Das (UT Austin)

---

## 1. Executive Summary

HiSpec introduces **intermediate verification** into speculative decoding using **early-exit (EE) models** to break the "verification wall" — the bottleneck where target model verification takes 2–10.3x longer than draft generation. Instead of waiting for the full target model to verify all draft tokens, HiSpec inserts an intermediate verifier at ~1/4 model depth that tentatively accepts/rejects tokens early, then periodically runs full target verification on the accumulated tentatively-accepted tokens. The key insight is that ~1/4 of model layers already produce 69% of tokens correctly, making them effective cheap filters. HiSpec achieves **1.28x average** and up to **2.01x** throughput improvement over baseline single-layer speculation across 6 models and 5 benchmarks, without accuracy degradation.

---

## 2. Detailed Analysis

### 2.1 The Verification Wall Problem

HiSpec's central motivation is quantified in Table 2 / Figure 2:

| Draft → Target | T_verif / T_gen |
|---|---|
| 1B → 70B (Llama3.1) | **6.0x** |
| 3B → 70B (Llama3.1) | **4.0x** |
| 8B → 70B (Llama3.1) | **2.4x** |
| 1B → 405B (Llama3.1) | **10.3x** |

### 2.2 Three-Layer Hierarchy Design

HiSpec uses a single EE model with three "virtual" models carved from different layer depths:

| Role | Layer Range | Purpose |
|---|---|---|
| Draft (L_d) | ~1/8 depth (e.g., Layer 3 for 32-layer model) | Fast token generation |
| Intermediate Verifier (L_i) | ~1/4 depth (e.g., Layer 8) | Early rejection of bad tokens |
| Target (L_f) | Full model (e.g., Layer 32) | Final ground-truth verification |

The workflow per iteration:
1. **Draft** generates N_d=2 tokens using layers 1→L_d
2. **Intermediate verify** checks each token against layers L_d+1→L_i
3. Repeat steps 1-2 until N_i=4 tokens tentatively accepted
4. **Target verify** runs full model on all tentatively-accepted tokens

### 2.3 KV Cache and Hidden State Reuse

The critical optimization: since draft, intermediate verifier, and target share the same model weights (just different layer subsets), HiSpec:
- **Buffers KV caches** generated during draft phase for reuse by intermediate verifier
- **Passes hidden states** from intermediate verifier directly to target verification (avoids recomputing layers 1→L_i)
- **Prunes** KV entries for rejected tokens after each verification step

This is fundamentally different from traditional speculative decoding (separate draft model), where no weight/KV sharing is possible.

### 2.4 Algorithm Walkthrough (Algorithm 2)

```
while not EOS:
    B_i = []                          # buffer for tentatively accepted tokens
    while |B_i| < N_i:               # accumulate until threshold
        specs = L_d.generate(ctx, N_d=2)    # draft 2 tokens
        for T_j in specs:
            p = L_i.predict(ctx)      # intermediate verify
            if T_j in top_predictions(p):
                B_i.append(T_j)       # tentatively accept
            else:
                B_i.append(L_i.generate(1))  # emit corrected token
                break                  # stop at first mismatch

    # Full target verification of accumulated buffer
    for U in B_i:
        q = L_f.predict(context)
        if U in top_predictions(q):
            context.append(U)         # commit
        else:
            context.append(L_f.generate(1))  # emit corrected token
            B_i = []                   # flush remaining
            break
```

Key design choices:
- **Stop-at-first-mismatch** in intermediate verification (no speculative continuation past rejection)
- **Dynamic target verification trigger** based on token count (N_i=4), not fixed rounds
- **Greedy top-k matching** rather than probabilistic rejection sampling

### 2.5 Intermediate Verifier Positioning

The 1/4 depth heuristic is empirically validated:
- Layer 8 of 32 (Llama3-8B) correctly produces 69% of tokens
- Exhaustive sweep of all (L_d, L_i) pairs for Llama3-8B confirms (3, 8) optimal
- Same ratio holds for Llama2-70B: (L_d=10, L_i=20) of 80 layers

---

## 3. Visual Breakdown

| Figure | Key Insight |
|---|---|
| **Fig 2** | Verification wall: 2–10.3x slower than draft. Scales with model size. |
| **Fig 3** | HiSpec overview: intermediate verify enables early rejection → faster next draft round |
| **Fig 4** | 1/4 model layers produce up to 69% correct tokens — justifies intermediate verifier position |
| **Fig 5** | Heatmap of throughput for all (L_d, L_i) pairs; optimal at ~(1/8, 1/4) depth |
| **Fig 6** | HiSpec Pareto-dominates LayerSkip baseline on both throughput AND acceptance rate |
| **Fig 7** | Ablation: lower N_d and N_i yield better throughput (shorter unverified chains) |
| **Fig 8** | Full 32x31 heatmap of all valid (draft, intermediate) combos for Llama3-8B |

---

## 4. Related Work Map

```
Speculative Decoding Landscape
├── Draft Acceleration (orthogonal to HiSpec)
│   ├── EAGLE/EAGLE-2/EAGLE-3 — trained draft heads, tree-based
│   ├── Medusa — multiple decoding heads
│   ├── LayerSkip — early-exit self-speculation
│   ├── AdaDecode — dynamic exit layer selection
│   ├── SWIFT — selective layer skipping
│   ├── Lookahead — parallel draft via modified attention mask
│   └── Kangaroo — early-exit + adapter + confidence threshold
│
├── Verification Acceleration (HiSpec's domain)
│   ├── SPRINTER — auxiliary verifier model (requires training, degrades accuracy)
│   ├── HiSpec — early-exit intermediate verification (no training, lossless) ◄━━
│   ├── PyramidSD (3-Model) — intermediate qualifier model
│   └── PPSD — pipeline-parallel draft+verify overlap
│
├── Hierarchical/Multi-stage
│   ├── HSD (Zhou et al. 2026) — hierarchical *verification algorithm* (different: focuses on
│   │   acceptance criteria, not intermediate models). 12% gain on EAGLE-3.
│   └── SpecMQuant — hierarchical with quantized intermediate stage
│
└── Orthogonal Techniques
    ├── Quantization (can combine with spec decoding)
    └── QuantSpec — quantized KV cache self-speculation
```

**Key distinction**: HiSpec is about adding an **intermediate verification stage** using existing model layers, not about improving the draft quality or the acceptance criteria.

---

## 5. Implementation Notes

### 5.1 Practical Considerations

**Requirements:**
- Needs an **early-exit model** (e.g., LayerSkip checkpoints on HuggingFace). Standard models without trained exit heads won't work well.
- EE model must have ≥2 exit points (draft + intermediate verifier)
- Pre-trained EE models (e.g., LayerSkip checkpoints) offer more flexibility than post-training modified ones

**Default hyperparameters:**
- L_d = 1/8 model depth, L_i = 1/4 model depth
- N_d = 2 (draft tokens per step)
- N_i = 4 (tentative acceptance window before target verification)

**Implementation framework:** HuggingFace Transformers (not optimized inference engines)

### 5.2 Limitations and Caveats

1. **No optimized inference engine integration**: All experiments use vanilla HuggingFace Transformers. Real-world deployment with CUDA graphs, FlashAttention, continuous batching would change the verify/draft ratio significantly.

2. **Single-request batch size only**: No evaluation of batched/concurrent request scenarios where verification parallelism could reduce the wall.

3. **EE model dependency**: Requires specially trained early-exit models. Can't use arbitrary draft+target model pairs (unlike EAGLE3 which uses a separately trained draft head).

4. **Greedy matching only**: Uses top-k matching rather than probabilistic rejection sampling. This means output isn't guaranteed to match the target model's sampling distribution exactly (though they claim "consistent" output via periodic target verification).

5. **No tree-based drafting**: Uses chain (sequential) drafting only. Tree-based approaches like EAGLE-3 could potentially be combined with HiSpec's intermediate verification idea.

### 5.3 Potential Integration Ideas

HiSpec's intermediate verification could theoretically be combined with EAGLE3:
1. Use EAGLE3 for drafting (better draft quality via trained feature extrapolation)
2. Add an intermediate verifier using early layers of the target model
3. Only run full target verification on tentatively-accepted tokens

This would require significant engineering to manage KV caches across the separate draft model and partial target model forward passes within a CUDA graph-based inference framework.
