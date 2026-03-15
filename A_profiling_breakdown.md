# Speculative Decoding Iteration Profiling Breakdown

**Target**: Llama-3.1-70B-Instruct (TP=4)
**Mode**: STANDALONE chain (num_steps=5, topk=1, num_draft_tokens=6), CUDA graphs enabled
**FLy**: entropy_threshold=0.3, window_size=6
**Profiling**: nsys + NVTX markers with `torch.cuda.synchronize()` + `--cuda-graph-trace=node`
**GPU**: 4× NVIDIA RTX PRO 6000 Blackwell (TP=4), dtype=float16

## Configurations


| Config | Draft Model           | Dataset   | Prompts | max_num_seqs |
| ------ | --------------------- | --------- | ------- | ------------ |
| **8B** | Llama-3.1-8B-Instruct | HumanEval | 10–20   | 2            |
| **3B** | Llama-3.2-3B-Instruct | ShareGPT  | 10      | 2            |
| **1B** | Llama-3.2-1B-Instruct | ShareGPT  | 10      | 2            |


## Phase-Level Breakdown (per iteration avg)


| Phase               | 8B (ms)   | 8B %      | 3B (ms)   | 3B %      | 1B (ms)   | 1B %      | Description                                   |
| ------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------------------------------------------- |
| **P1_Draft**        | **20.49** | **33.5%** | **15.55** | **24.6%** | **7.80**  | **15.4%** | Draft model × 5 sequential steps + tree build |
| **P2_Verify**       | **34.43** | **56.2%** | **42.35** | **67.0%** | **39.70** | **78.3%** | 70B forward + sampling                        |
| P2a_Prep            | 0.27      | 0.4%      | 0.52      | 0.8%      | 0.44      | 0.9%      | prepare_for_verify + batch setup              |
| **P2b_TargetFwd**   | **32.76** | **53.5%** | **34.42** | **54.5%** | **32.44** | **64.0%** | 70B model forward (CUDA graph replay)         |
| P2c_VerifySample    | 1.37      | 2.2%      | 7.37      | 11.7%     | 6.78      | 13.4%     | spec_info.verify + acceptance check           |
| **P3_DraftExtend**  | **6.25**  | **10.2%** | **5.25**  | **8.3%**  | **3.10**  | **6.1%**  | Draft extend forward (prepare hidden states)  |
| **SpecDecode_Iter** | **61.21** | **100%**  | **63.22** | **100%**  | **50.67** | **100%**  | Full iteration                                |


**Verify / Draft ratio**:

- 8B: 34.43 / 20.49 = **1.68×** | Draft total (P1+P3) = 26.74 ms (78% of verify)
- 3B: 42.35 / 15.55 = **2.72×** | Draft total (P1+P3) = 20.80 ms (49% of verify)
- 1B: 39.70 / 7.80 = **5.09×** | Draft total (P1+P3) = 10.90 ms (27% of verify)

## Kernel-Level Breakdown (GPU device 0, per iteration)


| Category                    | 8B (ms)   | 8B %      | 3B (ms)   | 3B %      | 1B (ms)   | 1B %      |
| --------------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| **GEMM** (QKV/O proj + MLP) | **38.34** | **67.0%** | **30.93** | **60.2%** | **25.43** | **64.8%** |
| **NCCL** (TP allreduce)     | **12.11** | **21.2%** | **14.84** | **28.9%** | **9.63**  | **24.6%** |
| **Attention** (flashinfer)  | **3.32**  | **5.8%**  | **3.09**  | **6.0%**  | **2.28**  | **5.8%**  |
| Norm + Activation + RoPE    | 2.07      | 3.6%      | 1.86      | 3.6%      | 1.34      | 3.4%      |
| KV cache + Other            | 1.41      | 2.5%      | 0.64      | 1.3%      | 0.54      | 1.4%      |
| **GPU kernel total**        | **57.25** |           | **51.36** |           | **39.22** |           |


## End-to-End Metrics


| Metric                | 8B Draft | 3B Draft | 1B Draft |
| --------------------- | -------- | -------- | -------- |
| Throughput (tokens/s) | —        | 115.06   | 132.49   |
| Time per token (ms)   | —        | 8.69     | 7.55     |
| Avg acceptance length | —        | 4.02     | 3.70     |
| Iter time (ms)        | 61.21    | 63.22    | 50.67    |


## Key Findings

1. **GEMM dominates** (~60–67%) across all draft sizes — linear projections (QKV, O, Gate/Up/Down) are the main cost, not attention or communication.
2. **NCCL communication is 2nd** (~21–29%) — TP=4 requires two allreduce per layer. Smaller draft models have fewer layers, reducing NCCL overhead.
3. **Attention is only ~6%** — chain mode verifies just 6 tokens; KV cache is long but query count is small, so attention compute is minimal.
4. **Smaller drafts shift the bottleneck to verify** — 1B draft is so cheap (7.80 ms) that verify dominates at 78% of iteration time (vs 56% with 8B). The verify/draft ratio goes from 1.68× (8B) to 5.09× (1B).
5. **1B achieves best throughput** despite lower acceptance (3.70 vs 4.02) — the 17× cheaper draft cost (7.80 vs 20.49 ms P1 + 3.10 vs 6.25 ms P3) more than compensates for the 8% lower acceptance rate.
6. **P2c_VerifySample is higher with ShareGPT** (6.78–7.37 ms vs 1.37 ms with HumanEval) — likely due to longer sequences and more complex verification sampling with the ShareGPT dataset.
7. **Draft is not cheap with 8B** — P1+P3 = 26.74 ms (78% of verify), but drops to 10.90 ms (27%) with 1B.

## Profiling Notes

- **NVTX without `cuda.synchronize()`** gives distorted results: P1_Draft appears ~6.6 ms (undercounted) and P2_Verify appears ~51.8 ms (overcounted), yielding a false 7.9× ratio. Root cause: `cuda_graph_runner.replay()` returns asynchronously (CPU doesn't wait for GPU), while `spec_info.verify()` calls `.tolist()` which forces a full stream sync, draining leftover draft GPU work into verify's timing.
- `**--disable-cuda-graph**` also distorts results: draft's 5 sequential steps suffer ~5× kernel launch overhead while verify's single pass suffers ~1×, making draft appear as the bottleneck.
- `**--cuda-graph-trace=node**` is required to see individual kernels within CUDA graph replay. It conflicts with `--cuda-memory-usage=true`; use `-t cuda,nvtx` without `osrt` and memory tracing.

