/**
 * Test TMEM alloc/dealloc on Blackwell
 *
 * Build: nvcc -arch=sm_120a -std=c++17 -O2 test_tmem.cu -o test_tmem
 * Run:   CUDA_VISIBLE_DEVICES=0 ./test_tmem
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void kernel_tmem_alloc_dealloc() {
    __shared__ int tmem_addr_smem[1];

    // tcgen05.alloc: full warp needed, result written to shared memory
    // num_cols must be in [32..512] and multiple of 32
    if (threadIdx.x < 32) {
        unsigned int smem_addr;
        asm volatile(
            "{\n"
            "  .reg .u64 addr64;\n"
            "  cvta.to.shared.u64 addr64, %1;\n"
            "  cvt.u32.u64 %0, addr64;\n"
            "}\n"
            : "=r"(smem_addr) : "l"(tmem_addr_smem)
        );

        // Allocate 32 columns (minimum)
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            : : "r"(smem_addr), "n"(32)
        );
    }
    __syncthreads();

    int taddr = tmem_addr_smem[0];
    if (threadIdx.x == 0) {
        printf("  TMEM allocated at addr = %d\n", taddr);
    }

    // Deallocate (num_cols must match alloc = 32, and be multiple of 32)
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
            : : "r"(taddr), "n"(32)
        );
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("  TMEM deallocated OK\n");
    }
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    printf("[Test] TMEM alloc/dealloc (tcgen05)...\n");
    kernel_tmem_alloc_dealloc<<<1, 128>>>();
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
