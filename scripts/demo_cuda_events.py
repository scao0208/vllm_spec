"""
Demo: CUDA Events vs wall-clock timing for GPU profiling.

Shows how CUDA events measure real GPU execution time
with near-zero overhead, without blocking the GPU pipeline.
"""

import time
import torch

torch.manual_seed(42)
device = "cuda:0"


def fake_attention(Q, K, V):
    """Simulate attention: softmax(QK^T / sqrt(d)) V"""
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (d ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


# ── Scenario: profile 10 "verify" attention calls ──
# Simulate chain (Q=23) vs tree (Q=512) with context=1000

print("=" * 60)
print("CUDA Events Demo: Profiling Attention Forward")
print("=" * 60)

for label, q_len in [("Chain (Q=23)", 23), ("Tree (Q=512)", 512)]:
    context_len = 1000
    d_head = 128
    total_kv = context_len + q_len

    Q = torch.randn(q_len, d_head, device=device)
    K = torch.randn(total_kv, d_head, device=device)
    V = torch.randn(total_kv, d_head, device=device)

    # Warmup
    for _ in range(3):
        fake_attention(Q, K, V)
    torch.cuda.synchronize()

    # ── Method 1: CUDA Events (what we'll use) ──
    events = []
    for i in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()                    # GPU timestamp — no sync
        result = fake_attention(Q, K, V)  # kernel runs
        end.record()                      # GPU timestamp — no sync
        events.append((start, end))
        # Note: NO synchronize here — GPU keeps running at full speed

    # Sync ONCE at the end, compute all 10 times in bulk
    torch.cuda.synchronize()
    gpu_times = [s.elapsed_time(e) for s, e in events]

    # ── Method 2: Wall-clock (for comparison) ──
    wall_times = []
    for i in range(10):
        torch.cuda.synchronize()          # must sync before timing
        t0 = time.perf_counter()
        result = fake_attention(Q, K, V)
        torch.cuda.synchronize()          # must sync after timing
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000)

    # ── Method 3: Wall-clock WITHOUT sync (wrong!) ──
    wrong_times = []
    for i in range(10):
        t0 = time.perf_counter()
        result = fake_attention(Q, K, V)  # async! returns immediately
        t1 = time.perf_counter()
        wrong_times.append((t1 - t0) * 1000)

    print(f"\n{label}  (context={context_len}, total_kv={total_kv})")
    print("-" * 60)
    print(f"  CUDA Events (real GPU time):  {sum(gpu_times)/10:.4f} ms avg")
    print(f"  Wall-clock + sync (correct):  {sum(wall_times)/10:.4f} ms avg")
    print(f"  Wall-clock no sync (WRONG):   {sum(wrong_times)/10:.4f} ms avg")
    print(f"  ^ This is just kernel launch overhead, not execution time")

print(f"\n{'=' * 60}")
print("Key takeaway:")
print("  CUDA events = real GPU time, near-zero overhead, no sync needed")
print("  Wall-clock + sync = correct but blocks GPU pipeline (slow)")
print("  Wall-clock no sync = measures CPU launch time, not GPU time")
print(f"{'=' * 60}")
