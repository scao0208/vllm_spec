/**
 * Definitively determine mma.sync m16n8k16 A fragment mapping
 * Set exactly ONE thread's A register to non-zero and see where it shows up
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t pack_half2(half a, half b) {
    half2 h = make_half2(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

// Set ONE specific thread's ONE A register to {1,0}, all else zero.
// B = all 1s. Then D[m,n] = A[m, k_low] where k_low is the first half
// of the pair owned by that thread/register combo.
__global__ void probe_a_kernel(float* out, int target_tid, int target_reg) {
    int tid = threadIdx.x;
    half one = __float2half(1.0f);
    half zero = __float2half(0.0f);

    uint32_t a0 = pack_half2(zero, zero);
    uint32_t a1 = pack_half2(zero, zero);
    uint32_t a2 = pack_half2(zero, zero);
    uint32_t a3 = pack_half2(zero, zero);

    // Set target thread's target register to {1, 0}
    if (tid == target_tid) {
        uint32_t val = pack_half2(one, zero);
        if (target_reg == 0) a0 = val;
        if (target_reg == 1) a1 = val;
        if (target_reg == 2) a2 = val;
        if (target_reg == 3) a3 = val;
    }

    // B = all 1s → D[m,n] = sum_k A[m,k]*1 = sum_k A[m,k]
    uint32_t b0 = pack_half2(one, one);
    uint32_t b1 = pack_half2(one, one);

    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));

    // Store full 16x8 D matrix (each thread writes its 4 elements)
    int row0 = tid / 4;
    int col_pair = tid % 4;
    out[row0*8 + col_pair*2]     = d0;
    out[row0*8 + col_pair*2 + 1] = d1;
    out[(row0+8)*8 + col_pair*2]     = d2;
    out[(row0+8)*8 + col_pair*2 + 1] = d3;
}

int main() {
    printf("mma.sync m16n8k16 A fragment mapping probe\n");
    printf("Setting one thread's one A register to {1,0}, B=all 1s\n\n");

    float *d_out, h_out[128];
    cudaMalloc(&d_out, 128 * sizeof(float));

    // Probe: for a few key threads, set each register and see which D[m,n]=1
    // Since we set {1,0}, exactly one k is non-zero (the low half of the pair).
    // D[m,n] = A[m,k] * B[k,n] summed. With B=all 1s and only one A entry = 1,
    // D[m,n] = 1 for all n in that row m.

    int probe_threads[] = {0, 1, 2, 3, 4, 8, 16};
    int num_probes = 7;

    for (int pi = 0; pi < num_probes; pi++) {
        int t = probe_threads[pi];
        for (int reg = 0; reg < 4; reg++) {
            cudaMemset(d_out, 0, 128*4);
            probe_a_kernel<<<1, 32>>>(d_out, t, reg);
            cudaDeviceSynchronize();
            cudaMemcpy(h_out, d_out, 128*4, cudaMemcpyDeviceToHost);

            // Find which rows have D[row, 0] == 1.0
            printf("Thread %2d, a%d={1,0}: ", t, reg);
            bool found = false;
            for (int m = 0; m < 16; m++) {
                if (h_out[m*8] == 1.0f) {
                    printf("D[%d,*]=1 → A[%d, k=?]", m, m);
                    found = true;
                    break;
                }
            }
            if (!found) printf("no D entry = 1.0");
            printf("\n");
        }
    }

    // Now probe B similarly
    printf("\nmma.sync m16n8k16 B fragment mapping probe\n");
    printf("Setting one thread's one B register, A=all 1s\n\n");

    cudaFree(d_out);
    return 0;
}
