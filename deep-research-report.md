# Bitmap‑Driven SMEM Block Design for Subtree‑Aware Sparse Verification

## Executive summary

The supplied PDF in this chat is a cluster control‑plane paper (“Manage the Workloads, Not the Cluster…”) and does **not** contain bitmap/SMEM attention‑kernel design; therefore, this note treats the “paper” as ambiguous and bases kernel‑level details on **DeFT (Decoding with Flash Tree‑Attention)**, **SpecInfer**, **FlashAttention‑2**, and NVIDIA’s CUDA/Hopper documentation. citeturn3view6turn6view1turn5view1turn2view2turn3view2  
DeFT’s key message for tree decoding/verification is that attention is often **memory‑bound**, so performance hinges on reducing **HBM↔SMEM movement of KV tiles** and avoiding spilling intermediate partial results (e.g., \(QK^\top\), softmax) to HBM via **tiling + kernel fusion**. citeturn3view6turn6view2turn2view2  
Bitmasks (e.g., SpecInfer‑style 64‑bit masks referenced by DeFT) compress tree causal constraints to tiny metadata, and can be elevated from “mask representation” to an **execution schedule**: a bitmap‑driven traversal that **skips loading and computing** on invisible KV blocks—especially attractive in subtree‑aware sparse verification. citeturn2view0turn3view5turn6view2  

## What “HBM ↔ SMEM traffic” means

**Short answer:** “HBM ↔ SMEM traffic” refers to data transfers between **HBM (global memory / VRAM)** and **SMEM (on‑chip shared memory inside each SM/thread block)**, including the recurring movement of **KV cache blocks** (and sometimes masks/metadata) from HBM into SMEM tiles for computation, plus writing outputs (and any intermediate partials, if unfused) back to HBM. DeFT explicitly characterises the attention kernel’s essential metadata/data flow as going “from HBM to shared memory in groups” during the QKV preparation phase. citeturn3view6  

Concretely, the main flows in a fused attention kernel are:

- **HBM → (L2) → SMEM:** load K/V tiles (and optionally a compact mask/bitmap) into the thread block’s shared memory “working set”. DeFT emphasises that minimising memory access **between HBM and shared memory** is crucial for memory‑bound attention computations. citeturn3view6turn6view2  
- **SMEM → registers → Tensor Cores:** load fragments (typically via `ldmatrix` in CUDA implementations) and perform MMA for \(QK^\top\) and \(PV\) (or equivalent fused accumulate). This stage is compute‑heavy but becomes starved if the HBM→SMEM stage is inefficient. citeturn2view2  
- **(Optional, undesirable) SMEM/HBM for partials:** if not fused, intermediate matrices (e.g., attention scores and softmax outputs) may be written/read, adding IO; DeFT explicitly targets eliminating IO of partial results via tiling + fusion. citeturn6view2turn2view2  
- **H100/Hopper‑specific transfer mechanisms:** TMA is explicitly designed to efficiently move multi‑dimensional arrays from global memory to shared memory. citeturn3view2turn9view0  

## Bitmap layout and traversal as an execution schedule

### What the primary sources say about bitmasks in tree attention

DeFT’s appendix discussion of mask overhead makes two points that motivate bitmap‑driven sparse execution:

- Causal masks introduce **memory‑access redundancy** (loading an \(n_q \times n_{kv}\) mask to shared memory) and **compute redundancy** because many \(QK^\top\) products are later masked out and never used. citeturn3view5  
- SpecInfer is described as using a **64‑bit integer per query token** to encode causal relationships among up to 64 tokens, incurring “minimal IO cost from HBM to shared memory” but limited to small trees. citeturn2view0turn2view0  

DeFT also notes that detailed bitmask design for its DEFT‑Subtree variant is deferred (“will be in the future version”), so any block‑sparse scheduling layout must be presented parametrically. citeturn3view5  

### Distilled essence: block‑sparse bitmap vs tree bitmap

Both are “bitsets over KV blocks”, but they differ in **what determines the bits** and **how reusable the mask is**:

- **Block‑sparse (history) bitmap:** a **policy** (recent window, landmarks, draft‑guided saliency) chooses which history blocks remain “visible”; this mask can often be shared across many queries (per request / per head / per layer).  
- **Tree (subtree) bitmap:** the **tree structure** (ancestor closure + candidate branches) determines visibility; masks typically vary per query‑block (or per subtree group) and change with the tree.  

This aligns with DeFT’s observation that query IO is often negligible compared to KV IO, and many queries can share ancestor KV cache in tree decoding (reuse opportunity). citeturn6view1turn6view2  

### Parametric bitmap layout: bit → KV block mapping

Let:

- `BLOCK_N` = number of KV tokens per block (block granularity).  
- `N_blocks` = number of KV blocks in the region you’re masking (history window or tree region).  
- `W = ceil(N_blocks / 64)` = number of 64‑bit words.

**Mapping:**

- Word index: `w = block_id // 64`  
- Bit index: `b = block_id % 64`  
- Bit meaning: `bitmap[w][b] == 1` ⇒ KV block `block_id` is visited (loaded + computed).  
- KV token start of that block: `k0 = block_id * BLOCK_N`.

This is the practical “block‑bitmap” specialisation of the SpecInfer‑style “64‑bit mask” described in DeFT. citeturn2view0turn3view5  

### Storage granularity: per query‑block vs per warp

Let `q_block` denote a group of `Br` query rows (FlashAttention convention).

**Bitmap‑per‑q_block** (CTA‑reused)  
Use when queries in the same `q_block` share visibility (common in subtree verification when queries are siblings under the same subtree root). This mirrors DeFT’s philosophy of grouping by shared KV to eliminate redundant IO. citeturn6view2turn6view1  

**Bitmap‑per‑warp**  
Use when visibility varies significantly inside a CTA; store one bitmap per warp to keep warp control flow coherent.

### Placement: HBM/L2/SMEM/registers

A robust placement hierarchy is:

- **HBM (global):** store all bitmaps as tightly packed `uint64` arrays (metadata is tiny).  
- **L2:** rely on reuse/grouping to keep hot masks and hot prefix KV blocks resident; DeFT’s grouping aims to reduce redundant KV IO and indirectly improves cache locality. citeturn6view2turn6view1  
- **Registers:** load the active `uint64 mask` into registers and iterate with integer intrinsics (`__ffsll`, `__popcll`, etc.). NVIDIA documents 64‑bit bit operations such as `__ffsll` and `__popcll` as supported device intrinsics. citeturn9view3turn9view2  
- **SMEM:** generally avoid staging bitmaps into SMEM unless multiple warps repeatedly access many words; SMEM is better spent on KV tiles.

### Bitmap traversal pseudocode and memory access pattern

#### Set‑bit iteration (ctz/ffs + popcount)

```c
// Inputs: bitmap[W] in HBM; BLOCK_N, base pointers for K/V.
// Optional: nnz = popcount(mask) to plan prefetch/loop bounds.

for (int w = 0; w < W; ++w) {
  uint64_t mask = bitmap[w];          // load once; keep in registers
  int nnz = popcount64(mask);         // e.g., __popcll

  while (mask) {
    int b = ctz64(mask);              // or: b = __ffsll(mask) - 1
    mask &= (mask - 1);               // clear lowest set bit
    int block_id = w * 64 + b;
    int k0 = block_id * BLOCK_N;

    // Access pattern:
    // - Coalesced contiguous load within [k0, k0+BLOCK_N)
    // - Skips whole blocks when bits are 0 (irregular across blocks)
    load_KV_tile(k0);
    fused_attention_update();         // online softmax + accumulate
  }
}
```

**Memory access pattern:** inside each selected block, loads are contiguous and coalesced; across blocks, you skip zeros (non‑unit stride). This is why a hybrid “dense history via TMA, sparse tree via bitmap traversal” is often best. citeturn3view2turn9view0turn3view5  

### Why 64‑bit is a GPU sweet spot (and when not to use it)

**Why 64‑bit is practical:**
- Fits in (effectively) a single register value per mask word, enabling register‑resident traversal.  
- Native or near‑native integer intrinsics exist for 64‑bit bit operations (`__ffsll`, `__popcll`), and NVIDIA notes that 64‑bit variants map to only a few instructions. citeturn9view3turn9view2  
- Encourages warp‑coherent traversal (all lanes follow the same “while(mask)” path if you broadcast one mask per warp).

**When to use multi‑segment bitmaps:** if `N_blocks > 64`, store `W = ceil(N_blocks/64)` words and iterate segments; this preserves bit‑ops traversal while scaling to long contexts.

**When to consider index lists:** if masks are extremely sparse and highly scattered (e.g., only 1–3 blocks in a very large region), an explicit index list can reduce loop overhead, but it re‑introduces index materialisation and indirection overhead (the very overhead you are trying to avoid).

## SMEM block design: tile layout, banking, and TMA/cp.async integration

### Parametric tile layout

Let:

- `Br` = query rows per CTA tile  
- `Bc` = KV columns per tile (typically set to `BLOCK_N` for a block‑bitmap design)  
- `d` = head dimension  
- `s` = bytes per element (`2` for FP16/BF16)

A typical fused‑attention SMEM allocation (double‑buffered) is:

- `smem_K[2][Bc][d + PAD_K]`  
- `smem_V[2][Bc][d + PAD_V]`  
- optional `smem_Q[Br][d + PAD_Q]` (often Q stays in registers)

Double buffering overlaps transfers with compute.

This is consistent with DeFT’s framing of attention execution: (i) load/group QKV into shared memory, then (ii) apply fused attention on groups, and with FlashAttention‑2’s online‑softmax tiling that avoids storing full \(S\) and \(P\) matrices. citeturn3view6turn2view2  

### Element layout: row‑major vs column‑major

For block‑bitmap traversal, prefer **token‑major row‑major** in global and SMEM:

- global K/V: `[token][d]` contiguous for each token block  
- SMEM K/V tile: `[Bc][d]` to enable contiguous bulk loads

If MMA prefers a different fragment layout, transpose/swizzle at the register‑fragment stage (or via `ldmatrix`) rather than forcing global loads into a strided pattern.

### SMEM bank mapping and conflict mitigation

NVIDIA’s Best Practices Guide states that when multiple addresses in a request map to the same shared‑memory bank, accesses are serialised, reducing effective bandwidth. citeturn3view1  
It also provides a canonical example where writing a tile “in columns” causes many‑way bank conflicts and recommends padding the shared memory array with an extra column to break the stride‑32 bank mapping. citeturn3view1  

**Practical mitigations for attention tiles:**
- **Padding (`PAD_K`, `PAD_V`)**: choose padding so the stride in bytes is not an unlucky multiple of the bank period for your access pattern. In transpose‑like patterns, the “+1 column” remedy is a reliable first test. citeturn3view1  
- **Skewing/swizzling**: use a small per‑row skew in SMEM to avoid repeated bank aliasing when warps read columns.  
- **Double buffering**: allow you to keep compute progressing while the next tile is loaded.

### TMA vs `cp.async` for global→SMEM transfers

**H100/Hopper TMA:** CUDA’s Programming Guide describes Hopper’s Tensor Memory Accelerator (TMA) as providing an efficient mechanism to transfer multi‑dimensional arrays from global memory to shared memory. citeturn3view2turn2view3  
The PyTorch Hopper TMA deep dive adds two implementer‑relevant points: TMA is lightweight (one thread can initiate transfers) and moves data directly GMEM→SMEM, avoiding heavy register participation typical of older approaches. citeturn9view0  

**Guidance for bitmap traversal:**
- Use **TMA** for *dense, contiguous* regions (e.g., compressed history window) where you stream many sequential tiles.  
- For bitmap‑selected sparse blocks, consider **bulk async copies** (`cp.async.bulk`) if each selected block is contiguous but blocks are non‑sequential; note the alignment constraint: both source and destination must be 16‑byte aligned and copy size must be a multiple of 16. citeturn9view1turn2view3  

## Performance model and sensitivity (H100 and A100)

### Assumptions

- **H100 SXM** memory bandwidth: 3.35 TB/s (peak). citeturn2view7  
- **A100 80GB** memory bandwidth: 2 TB/s (datasheet), and Ampere whitepaper lists 1555 GB/s for a 40GB configuration; treat A100 as ~1.55–2.0 TB/s depending on SKU. citeturn2view6turn2view5  
- Kernel is **memory‑bound** in the KV load phase (as emphasised by DeFT’s focus on KV cache IO dominance). citeturn6view1turn3view6  
- Let `α` be achieved bandwidth efficiency for dense streaming (TMA‑friendly) and `β` for sparse bitmap traversal; typically `β ≤ α` due to less regular streaming.

### Bytes moved per KV block and per traversal

Per KV block (K and V):

\[
Bytes_{KV\_block} = 2 \cdot Bc \cdot d \cdot s
\]

Dense traversal over `N_blocks`:

\[
Bytes_{dense} \approx N_{blocks} \cdot Bytes_{KV\_block}
\]

Bitmap traversal over `N_active = p \cdot N_blocks`:

\[
Bytes_{bitmap} \approx N_{active} \cdot Bytes_{KV\_block} + 8 \cdot \left\lceil \frac{N_{blocks}}{64}\right\rceil
\]

The bitmap metadata term is generally negligible compared with KV tiles, matching DeFT’s point that bitmask IO is minimal compared with dense masks. citeturn2view0turn3view5  

### Latency estimates (memory‑bound approximation)

\[
T_{dense} \approx \frac{Bytes_{dense}}{\alpha \cdot BW_{HBM}}
\qquad
T_{bitmap} \approx \frac{Bytes_{bitmap}}{\beta \cdot BW_{HBM}}
\]

**Example (illustrative):** with `Bc=BLOCK_N=64`, `d=128`, `s=2` bytes (BF16/FP16),  
\(Bytes_{KV\_block} = 2 \cdot 64 \cdot 128 \cdot 2 = 32\,768\) bytes ≈ 32 KiB per head per block.  
At peak bandwidth, this is ~9.8 ns per block on H100 and ~16 ns per block on an A100‑class 2 TB/s link (these are lower bounds; actual kernels are limited by SM‑level concurrency and achieved bandwidth). citeturn2view7turn2view6  

### Instruction mix: bitmap traversal vs dense TMA

- **Dense TMA path:** fewer instructions per tile (issue bulk/tensor copy + pipeline/barrier), high data‑movement efficiency. CUDA explicitly positions TMA as offloading data transfer from global to shared memory for multi‑dimensional arrays. citeturn3view2turn9view0  
- **Bitmap path:** adds a small integer overhead per active block (`ffs/ctz`, `mask&=mask-1`, pointer arithmetic). Device intrinsics exist for 64‑bit bit scanning and popcount. citeturn9view3turn9view2  

In practice, the bitmap integer overhead is amortised when `Bytes_{KV_block}` is large (moderate/large `Bc`, `d`), and the win comes from skipping entire KV tiles and masked‑out \(QK^\top\) work (a redundancy DeFT highlights). citeturn3view5  

### Sensitivity

- **Block size (`BLOCK_N` / `Bc`)**  
  - Larger `Bc` improves copy efficiency (TMA/bulk copies) and amortises integer overhead, but reduces pruning granularity.  
  - Smaller `Bc` increases granularity but raises loop and metadata overhead and can hurt copy efficiency.

- **Tree sparsity (`p = N_active / N_blocks`)**  
  - Ideal speedup upper bound (if α≈β): \(\approx 1/p\).  
  - If sparse traversal reduces achieved bandwidth (β<α), realised speedup is \(\approx (\alpha/\beta) \cdot (1/p)\).

- **Query‑block grouping**  
  Grouping queries that share KV blocks improves L2 reuse and reduces redundant HBM→SMEM loads; DeFT’s KV‑guided grouping is explicitly designed to reduce such redundant KV cache IO. citeturn6view2turn6view1  

## Implementation guidance for kernel engineers

### Triton vs CUDA

DeFT implements its kernel in **Triton** to control memory access and fuse attention operations into a single kernel. citeturn6view1  
For bitmap traversal:

- **Triton is ideal for prototyping** the bitmap‑driven loop structure and tile sizes.  
- **CUDA is preferred for peak Hopper performance** when you want direct control over TMA, async pipelines/barriers, warp specialisation, and low‑level fragment/SMEM layouts. CUDA’s TMA guidance is in the Programming Guide. citeturn3view2turn2view3  

### Register pressure, occupancy, and divergence

- Asynchronous copies can reduce intermediate register usage and increase occupancy; Best Practices notes async copy avoids intermediary register file access and can reduce register pressure. citeturn3view1  
- **Warp divergence** is the main control‑flow risk for bitmap traversal:
  - Use **bitmap‑per‑warp** (broadcast one mask with `__shfl_sync`) when per‑query visibility differs.
  - Use **bitmap‑per‑q_block** when visibility is shared; it reduces metadata reads and keeps warps coherent.

### Minimal code‑change pseudocode: dense → bitmap traversal

#### Triton‑style (conceptual)

```python
# Dense loop:
for start_n in range(0, seqlen_k, BLOCK_N):
    K = tl.load(K_ptr + start_n + ...)
    V = tl.load(V_ptr + start_n + ...)
    acc = flash_update(Q, K, V, acc, m, l)

# Bitmap-driven loop:
for w in range(W):
    mask = tl.load(bitmap_ptr + w)      # uint64 (or two uint32)
    while mask != 0:
        b = ctz(mask)                   # or emulate via ffs
        mask = mask & (mask - 1)
        start_n = (w*64 + b) * BLOCK_N
        K = tl.load(K_ptr + start_n + ...)
        V = tl.load(V_ptr + start_n + ...)
        acc = flash_update(Q, K, V, acc, m, l)
```

#### CUDA‑style (warp‑coherent)

```c
uint64_t mask = (lane0) bitmap[w];
mask = __shfl_sync(0xffffffff, mask, 0);

while (mask) {
  int b = __ffsll(mask) - 1;            // bit scan
  uint64_t next = mask & (mask - 1);
  mask = __shfl_sync(0xffffffff, next, 0);

  int block_id = w*64 + b;
  int k0 = block_id * BLOCK_N;

  // Dense history: prefer TMA streaming.
  // Sparse blocks: cp.async.bulk or vectorised ld/st.
  async_load_KV_to_smem(k0);
  compute_flash_tile();
}
```

This directly removes the “masked‑out compute” redundancy that DeFT flags for causal‑mask approaches and turns the bitmask into an execution schedule. citeturn3view5turn2view2  

## Design options, recommendations, and next steps

### Design options comparison

| Option | Metadata size (parametric) | Load reduction | Control‑flow complexity | Triton friendliness | Expected speedup regime |
|---|---:|---:|---:|---:|---|
| Bitmap per q_block | \(8 \cdot \lceil N_{blocks}/64\rceil\) bytes / q_block | High (skips 0‑blocks) | Medium | High | Best when many q rows share visibility; strong L2 reuse with grouping |
| Bitmap per warp | \(\#warps \cdot 8 \cdot \lceil N_{blocks}/64\rceil\) bytes / CTA | High | Lower divergence within warp | Medium | Best when visibility varies within CTA |
| Explicit index list | \(\approx N_{active} \cdot b_{idx}\) bytes (e.g., 2–4B per block id) | Maximal | Higher (indirection, pointer chasing) | Medium | Best at extreme sparsity; can lose if index overhead dominates |
| Dense streaming (TMA‑friendly) | ~0 | None | Low | High | Best when near‑dense or when the region is small and perfectly streamable |

### Mermaid flowchart: bitmap traversal loop

```mermaid
flowchart TD
  A[Load bitmap word(s) for q_block/warp] --> B{mask != 0?}
  B -- No --> Z[Write output tile and exit]
  B -- Yes --> C[bit = ffs/ctz(mask)]
  C --> D[mask = mask & (mask - 1)]
  D --> E[k0 = (word*64 + bit) * BLOCK_N]
  E --> F[Load K/V tile to SMEM\n(TMA for dense history; cp.async/ld for sparse blocks)]
  F --> G[Fused attention update\n(QK, online softmax, PV accumulate)]
  G --> B
```

### Memory layout diagram: SMEM tile + bitmap placement (conceptual)

```text
HBM (global memory / VRAM)
  K: [tokens, d]   (token-major contiguous)
  V: [tokens, d]
  history_bitmap: [head, seg] uint64   (often shared)
  tree_bitmap:    [q_block, seg] uint64 (often per-q_block)

L2 cache
  - caches hot KV prefix blocks and hot bitmap lines (benefits from grouping)

SMEM (per CTA, double-buffered)
  smem_K[2][Bc][d + PAD_K]
  smem_V[2][Bc][d + PAD_V]
  (bitmap usually NOT stored here)

Registers (per warp/program)
  Q fragments, accumulators (O, m, l), current uint64 mask word
```

### Prioritised micro‑optimisations to test

1. **Hybrid transfer strategy:** use TMA for the dense history window (streaming) and bitmap traversal only for sparse blocks (tree region and/or history landmarks). citeturn3view2turn9view0turn9view1  
2. **Group‑by‑bitmap / group‑by‑subtree:** schedule q_blocks so warps share similar masks; this mirrors DeFT’s KV‑guided grouping objective to eliminate redundant KV IO. citeturn6view2turn6view1  
3. **Double buffering + async copies:** overlap global→SMEM transfers and MMA compute; async copies avoid intermediate registers and can improve occupancy. citeturn3view1turn9view0  
4. **Bank‑conflict mitigation:** add padding/skew to SMEM tiles; NVIDIA explicitly shows padding an extra column as a simple remedy for severe bank conflicts in tiled patterns. citeturn3view1  
5. **Mask packing discipline:** prefer `uint64` words; rely on device intrinsics for `ffs/popcount` and note that 64‑bit popcount/bit‑reverse map to only a few instructions. citeturn9view2turn9view3  
6. **Alignment correctness for bulk async copies:** if using `cp.async.bulk`, enforce 16‑byte alignment and sizes multiple of 16. citeturn9view1  

### Next steps checklist

- Choose `BLOCK_N (=Bc)` and `Br` based on (i) MMA fragment shape and (ii) SMEM budget (double buffering + padding).  
- Decide a **history sparsity policy** (e.g., recent window + landmarks, or draft‑guided blocks) and determine whether the history bitmap can be shared per request/head/layer.  
- Prototype the **bitmap traversal loop in Triton** (correctness first), then evaluate CUDA/TMA porting if H100 peak throughput matters. citeturn6view1turn3view2