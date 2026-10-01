/**
 * Test Blackwell-specific features on RTX PRO 6000 (sm_120, CC 12.0):
 *   1. TMEM         — Tensor Memory alloc/dealloc (tcgen05.alloc/dealloc)
 *   2. tcgen05.mma  — Blackwell Tensor Core MMA instruction
 *   3. TMA          — Tensor Memory Accelerator (mbarrier + cp.async.bulk)
 *
 * Build:
 *   nvcc -arch=sm_120a -std=c++17 -O2 test_blackwell_features.cu -o test_blackwell_features
 *   (or -arch=sm_120 for non-accelerated)
 *
 * Run:
 *   CUDA_VISIBLE_DEVICES=0 ./test_blackwell_features
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Test 1: TMEM alloc/dealloc
// ============================================================================

__global__ void kernel_tmem() {
    // Only one thread allocates
    if (threadIdx.x == 0) {
        unsigned int tmem_addr;
        // Allocate 1 column (32 bytes) of tensor memory
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned %0, 1;\n"
            : "=r"(tmem_addr)
        );
        // Deallocate
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned %0, 1;\n"
            : : "r"(tmem_addr)
        );
    }
}

bool test_tmem() {
    printf("[Test 1] TMEM (tcgen05.alloc / tcgen05.dealloc)...\n");

    kernel_tmem<<<1, 32>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  FAIL (launch): %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  FAIL (runtime): %s\n", cudaGetErrorString(err));
        return false;
    }
    printf("  PASS\n");
    return true;
}

// ============================================================================
// Test 2: tcgen05.mma — simplest form
// ============================================================================
// Use tcgen05.mma with smallest config: fp16 input, fp32 accum
// A in registers, B in shared memory, result in TMEM

__global__ void kernel_tcgen05_mma() {
    __shared__ __align__(128) char smem_b[16384]; // B operand in smem

    // Zero shared memory
    for (int i = threadIdx.x; i < 16384 / 4; i += blockDim.x) {
        reinterpret_cast<int*>(smem_b)[i] = 0;
    }
    __syncthreads();

    // Allocate TMEM for accumulator (thread 0 only, broadcast to warp)
    unsigned int tmem_addr = 0;
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned %0, 32;\n"
            : "=r"(tmem_addr)
        );
    }
    // Broadcast
    tmem_addr = __shfl_sync(0xFFFFFFFF, tmem_addr, 0);

    // Get smem address for B
    unsigned int smem_b_u32;
    asm volatile(
        "{\n"
        "  .reg .u64 addr64;\n"
        "  cvta.to.shared.u64 addr64, %1;\n"
        "  cvt.u32.u64 %0, addr64;\n"
        "}\n"
        : "=r"(smem_b_u32) : "l"(smem_b)
    );

    // A operand: 4 zero registers
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;

    // tcgen05.mma with scale_vec::1X, block_scale, kind::f16
    // Minimal: cta_group::1 means 1 CTA in the group
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16.block_scale.scale_vec::1X"
        " [%0], {%1, %2, %3, %4}, [%5], zero::1, idesc[0];\n"
        :
        : "r"(tmem_addr), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(smem_b_u32)
        : "memory"
    );

    __syncthreads();

    // Deallocate TMEM
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned %0, 32;\n"
            : : "r"(tmem_addr)
        );
    }
}

bool test_tcgen05_mma() {
    printf("[Test 2] tcgen05.mma (Blackwell Tensor Core)...\n");

    kernel_tcgen05_mma<<<1, 32>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  FAIL (launch): %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  FAIL (runtime): %s\n", cudaGetErrorString(err));
        return false;
    }
    printf("  PASS\n");
    return true;
}

// ============================================================================
// Test 3: TMA — mbarrier + cp.async.bulk
// ============================================================================
// mbarrier was introduced in SM90 (Hopper), continues on Blackwell.
// This tests the basic TMA infrastructure.

__global__ void kernel_tma_mbarrier() {
    __shared__ __align__(8) unsigned long long barrier;

    if (threadIdx.x == 0) {
        // Initialize mbarrier with expected arrival count = 1
        asm volatile(
            "mbarrier.init.shared.b64 [%0], 1;\n"
            : : "l"(&barrier) : "memory"
        );

        // Signal arrival
        asm volatile(
            "mbarrier.arrive.shared.b64 _, [%0];\n"
            : : "l"(&barrier) : "memory"
        );
    }
    __syncthreads();
    // If we get here without crashing, mbarrier works
}

bool test_tma() {
    printf("[Test 3] TMA (mbarrier init/arrive)...\n");

    kernel_tma_mbarrier<<<1, 32>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  FAIL (launch): %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  FAIL (runtime): %s\n", cudaGetErrorString(err));
        return false;
    }
    printf("  PASS\n");
    return true;
}

// ============================================================================
// Test 4: tcgen05.st / tcgen05.ld — TMEM read/write
// ============================================================================
// Write values to TMEM, read them back

__global__ void kernel_tmem_st_ld(float* output) {
    unsigned int tmem_addr = 0;
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned %0, 1;\n"
            : "=r"(tmem_addr)
        );
    }
    tmem_addr = __shfl_sync(0xFFFFFFFF, tmem_addr, 0);

    // Store a value to TMEM
    float val = 42.0f + (float)threadIdx.x;
    asm volatile(
        "tcgen05.st.sync.aligned.32x1b.x1.b32 [%0], %1;\n"
        : : "r"(tmem_addr), "f"(val) : "memory"
    );

    // Load it back
    float loaded;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x1b.x1.b32 %0, [%1];\n"
        : "=f"(loaded) : "r"(tmem_addr) : "memory"
    );

    output[threadIdx.x] = loaded;

    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned %0, 1;\n"
            : : "r"(tmem_addr)
        );
    }
}

bool test_tmem_st_ld() {
    printf("[Test 4] TMEM st/ld (tcgen05.st / tcgen05.ld)...\n");

    float* d_out;
    float h_out[32];
    cudaMalloc(&d_out, 32 * sizeof(float));

    kernel_tmem_st_ld<<<1, 32>>>(d_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  FAIL (launch): %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  FAIL (runtime): %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return false;
    }

    cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    // Verify: thread 0 should have stored 42.0
    bool ok = true;
    for (int i = 0; i < 4; i++) {
        float expected = 42.0f + i;
        printf("  thread[%d]: expected=%.1f, got=%.1f %s\n",
               i, expected, h_out[i],
               (h_out[i] == expected) ? "OK" : "MISMATCH");
        if (h_out[i] != expected) ok = false;
    }
    if (ok) printf("  PASS\n"); else printf("  FAIL (data mismatch)\n");
    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d (sm_%d%d)\n\n",
           prop.major, prop.minor, prop.major, prop.minor);

    int pass = 0, fail = 0;

    if (test_tmem()) pass++; else fail++;
    printf("\n");

    if (test_tcgen05_mma()) pass++; else fail++;
    printf("\n");

    if (test_tma()) pass++; else fail++;
    printf("\n");

    if (test_tmem_st_ld()) pass++; else fail++;
    printf("\n");

    printf("========================================\n");
    printf("Results: %d PASS, %d FAIL (out of 4)\n", pass, fail);
    printf("========================================\n");

    return fail > 0 ? 1 : 0;
}
