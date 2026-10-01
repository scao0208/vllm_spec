/**
 * Test 2: TMEM load (tcgen05.ld) on Blackwell (sm_120a)
 *
 * Build: nvcc -arch=sm_120a -std=c++17 -O2 test_tmem_ld.cu -o test_tmem_ld
 * Run:   CUDA_VISIBLE_DEVICES=0 ./test_tmem_ld
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void kernel_tmem_ld(float* output) {
    __shared__ int tmem_addr_smem[1];

    // Allocate 8 columns of TMEM
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
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            : : "r"(smem_addr), "n"(8)
        );
    }
    __syncthreads();

    int taddr = tmem_addr_smem[0];

    // Load 8 FP32 values from TMEM (should be uninitialized/zero)
    // Address encoding: (row << 16) | col, where row = warp_id * 32
    if (threadIdx.x < 32) {
        int warp_id = threadIdx.x / 32;
        int row = warp_id * 32;
        int col = 0;
        int addr = taddr + (row << 16) + col;

        float tmp[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
            : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
              "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
            : "r"(addr)
        );

        asm volatile("tcgen05.wait::ld.sync.aligned;\n");

        // Write first 8 values from thread 0
        if (threadIdx.x == 0) {
            for (int i = 0; i < 8; i++) output[i] = tmp[i];
        }
    }
    __syncthreads();

    // Deallocate
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
            : : "r"(taddr), "n"(8)
        );
    }
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    float* d_out;
    float h_out[8];
    cudaMalloc(&d_out, 8 * sizeof(float));
    cudaMemset(d_out, 0xFF, 8 * sizeof(float)); // fill with NaN sentinel

    printf("[Test] TMEM ld (tcgen05.ld)...\n");
    kernel_tmem_ld<<<1, 128>>>(d_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  FAIL (launch): %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  FAIL (runtime): %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        return 1;
    }

    cudaMemcpy(h_out, d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  TMEM values: ");
    for (int i = 0; i < 8; i++) printf("%.2f ", h_out[i]);
    printf("\n  PASS\n");

    cudaFree(d_out);
    return 0;
}
