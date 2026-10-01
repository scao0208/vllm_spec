/**
 * Test 3: TMA (mbarrier) on Blackwell
 * mbarrier is sm_90+ feature, should work on sm_120.
 *
 * Build: nvcc -arch=sm_120 -std=c++17 -O2 test_tma.cu -o test_tma
 * Run:   CUDA_VISIBLE_DEVICES=0 ./test_tma
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void kernel_mbarrier() {
    __shared__ __align__(8) uint64_t barrier;

    if (threadIdx.x == 0) {
        unsigned int smem_addr;
        asm volatile(
            "{\n"
            "  .reg .u64 addr64;\n"
            "  cvta.to.shared.u64 addr64, %1;\n"
            "  cvt.u32.u64 %0, addr64;\n"
            "}\n"
            : "=r"(smem_addr) : "l"(&barrier)
        );

        // Init mbarrier with expected count = 1
        asm volatile(
            "mbarrier.init.shared.b64 [%0], 1;\n"
            : : "r"(smem_addr) : "memory"
        );

        // Arrive
        asm volatile(
            "mbarrier.arrive.shared.b64 _, [%0];\n"
            : : "r"(smem_addr) : "memory"
        );

        printf("  mbarrier init + arrive OK\n");
    }
    __syncthreads();
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    printf("[Test] TMA mbarrier (sm_90+ feature)...\n");
    kernel_mbarrier<<<1, 32>>>();
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
