/**
 * Test Hopper WGMMA on Blackwell (sm_120) — expected to FAIL
 * WGMMA is sm_90a, deprecated on Blackwell.
 *
 * Build: nvcc -arch=sm_120 -std=c++17 -O2 test_wgmma.cu -o test_wgmma
 * Run:   CUDA_VISIBLE_DEVICES=0 ./test_wgmma
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Try to use wgmma.mma_async (Hopper SM90a instruction)
// This should fail to compile for sm_120 since wgmma is deprecated.
__global__ void kernel_wgmma() {
    // wgmma requires warpgroup (128 threads)
    // Minimal: 64x8x16 fp16 -> fp32
    float d0 = 0.0f, d1 = 0.0f;
    unsigned int a0 = 0, a1 = 0;
    unsigned long long desc_b = 0;

    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16"
        " {%0, %1}, {%2, %3}, %4, 1, 1, 0, 0;\n"
        : "+f"(d0), "+f"(d1)
        : "r"(a0), "r"(a1), "l"(desc_b)
    );
}

int main() {
    printf("This test checks if wgmma compiles for sm_120.\n");
    printf("If you see this message, wgmma compiled successfully.\n");

    kernel_wgmma<<<1, 128>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  FAIL (launch): %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  FAIL (runtime): %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("  PASS\n");
    return 0;
}
