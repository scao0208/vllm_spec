/**
 * Test TMA + mbarrier on Blackwell (sm_120)
 *
 * Build: nvcc -arch=sm_120 -std=c++17 -O2 test_tma_bulk.cu -o test_tma_bulk
 * Run:   CUDA_VISIBLE_DEVICES=4 ./test_tma_bulk
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ unsigned int to_smem_u32(const void* ptr) {
    unsigned int r;
    asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
                 : "=r"(r) : "l"(ptr));
    return r;
}

// Test A: mbarrier full cycle
__global__ void kernel_mbarrier_full() {
    __shared__ __align__(8) uint64_t bar;
    unsigned int bar_u32 = to_smem_u32(&bar);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" : : "r"(bar_u32) : "memory");
        asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" : : "r"(bar_u32) : "memory");
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int ready;
        asm volatile(
            "{ .reg .pred p;\n"
            "  mbarrier.try_wait.parity.shared.b64 p, [%1], 0;\n"
            "  selp.u32 %0, 1, 0, p; }\n"
            : "=r"(ready) : "r"(bar_u32) : "memory");
        printf("  mbarrier try_wait: %s\n", ready ? "ready" : "NOT ready");
    }
}

// Test B: TMA cp.async.bulk
// For complete_tx::bytes mode, mbarrier.init count = 0 (no thread arrivals),
// and the TMA engine signals completion by adding tx_count bytes to the barrier.
// The barrier flips phase when arrived_count == expected_count AND tx_count == 0.
// So we init with count=1, then:
//   1. cp.async.bulk adds 1024 to pending tx_count
//   2. When transfer completes, 1024 is subtracted from tx_count
//   3. Thread calls mbarrier.arrive to decrement count
//   4. Phase flips when count=0 and tx_count=0
__global__ void kernel_tma_real(const float* __restrict__ src, float* __restrict__ dst) {
    __shared__ __align__(128) float smem[256];
    __shared__ __align__(8) uint64_t bar;

    unsigned int smem_u32 = to_smem_u32(smem);
    unsigned int bar_u32  = to_smem_u32(&bar);

    if (threadIdx.x == 0) {
        // Init with expected count = 1 (one thread will arrive after issuing TMA)
        asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" : : "r"(bar_u32) : "memory");

        // Issue TMA: adds 1024 bytes to pending tx
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0], [%1], 1024, [%2];\n"
            : : "r"(smem_u32), "l"(src), "r"(bar_u32) : "memory");

        // Thread arrives (expected_count decremented: 1->0)
        // TMA will signal tx completion separately
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], 1024;\n"
            : : "r"(bar_u32) : "memory");
    }
    __syncthreads();

    // Wait for phase flip
    if (threadIdx.x == 0) {
        int iters = 0;
        unsigned int done = 0;
        while (!done && iters < 10000000) {
            asm volatile(
                "{ .reg .pred p;\n"
                "  mbarrier.try_wait.parity.shared.b64 p, [%1], 0;\n"
                "  selp.u32 %0, 1, 0, p; }\n"
                : "=r"(done) : "r"(bar_u32) : "memory");
            iters++;
        }
        printf("  cp.async.bulk: done=%u iters=%d\n", done, iters);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        dst[i] = smem[i];
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Test A
    printf("[Test A] mbarrier init + arrive + try_wait...\n");
    kernel_mbarrier_full<<<1, 32>>>();
    cudaError_t err = cudaDeviceSynchronize();
    printf("  %s\n\n", err == cudaSuccess ? "PASS" : cudaGetErrorString(err));

    // Test B
    float h_src[256], h_dst[256];
    for (int i = 0; i < 256; i++) h_src[i] = (float)i;
    float *d_src, *d_dst;
    cudaMalloc(&d_src, 1024); cudaMalloc(&d_dst, 1024);
    cudaMemcpy(d_src, h_src, 1024, cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0, 1024);

    printf("[Test B] TMA cp.async.bulk (1KB global -> shared)...\n");
    kernel_tma_real<<<1, 32>>>(d_src, d_dst);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  FAIL: %s\n", cudaGetErrorString(err));
    } else {
        cudaMemcpy(h_dst, d_dst, 1024, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < 256; i++) {
            if (h_dst[i] != (float)i) {
                printf("  MISMATCH [%d]: %.0f vs %.0f\n", i, (float)i, h_dst[i]);
                ok = false; break;
            }
        }
        if (ok) printf("  Data: [0]=%.0f [127]=%.0f [255]=%.0f\n", h_dst[0], h_dst[127], h_dst[255]);
        printf("  %s\n", ok ? "PASS" : "FAIL (data)");
    }

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
}
