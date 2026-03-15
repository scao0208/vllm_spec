/**
 * Minimal mma.sync m16n8k16 diagnostic
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t pack_half2(half a, half b) {
    half2 h = make_half2(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

// Tests 1-4: uniform B, A=all 1s
__global__ void test_b1_kernel(float* out, int test_id) {
    half one = __float2half(1.0f);
    half two = __float2half(2.0f);
    half zero = __float2half(0.0f);
    uint32_t a_one = pack_half2(one, one);
    uint32_t a0 = a_one, a1 = a_one, a2 = a_one, a3 = a_one;
    uint32_t b0, b1;
    if (test_id == 1) { b0 = pack_half2(one, one);  b1 = pack_half2(one, one); }
    if (test_id == 2) { b0 = pack_half2(one, one);  b1 = pack_half2(two, two); }
    if (test_id == 3) { b0 = pack_half2(one, one);  b1 = pack_half2(zero, zero); }
    if (test_id == 4) { b0 = pack_half2(zero, zero); b1 = pack_half2(one, one); }
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
    if (threadIdx.x == 0) { out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3; }
}

// Test 5: varying B, A=all 1s
__global__ void test_varying_b_kernel(float* out) {
    int tid = threadIdx.x;
    half one = __float2half(1.0f);
    uint32_t a_one = pack_half2(one, one);
    uint32_t a0 = a_one, a1 = a_one, a2 = a_one, a3 = a_one;
    uint32_t b0 = pack_half2(__float2half((float)(tid*2+1)), __float2half((float)(tid*2+2)));
    uint32_t b1 = pack_half2(__float2half((float)(100+tid*2+1)), __float2half((float)(100+tid*2+2)));
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
    if (tid == 0) { out[0] = d0; out[1] = d1; out[2] = d2; out[3] = d3; }
    if (tid == 1) { out[4] = d0; out[5] = d1; out[6] = d2; out[7] = d3; }
}

// Test 6: exact replica of debug_mma_kernel from test_quest_mma.cu
// A: row 0 = all 1.0, rest = 0. B: B[k,n] = k+1 for all n.
// Expected: D[0,n] = sum(1..16) = 136, D[m>0,n] = 0
__global__ void test6_kernel(float* out) {
    int tid = threadIdx.x;
    int row0 = tid / 4;
    int col_group = tid % 4;

    half one = __float2half(1.0f);
    half zero = __float2half(0.0f);

    uint32_t a0 = (row0 == 0) ? pack_half2(one, one) : pack_half2(zero, zero);
    uint32_t a1 = (row0 == 0) ? pack_half2(one, one) : pack_half2(zero, zero);
    uint32_t a2 = pack_half2(zero, zero);
    uint32_t a3 = pack_half2(zero, zero);

    // B[k,n] = k+1. Since b maps tid_in_group→K, groupID→N:
    // b0 = {B[col_group*2, groupID], B[col_group*2+1, groupID]} = {col_group*2+1, col_group*2+2}
    // b1 = {B[8+col_group*2, groupID], B[8+col_group*2+1, groupID]} = {8+col_group*2+1, 8+col_group*2+2}
    uint32_t b0 = pack_half2(__float2half((float)(col_group*2+1)), __float2half((float)(col_group*2+2)));
    uint32_t b1 = pack_half2(__float2half((float)(8+col_group*2+1)), __float2half((float)(8+col_group*2+2)));

    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));

    // Store D[16,8] matrix
    out[row0*8 + col_group*2]     = d0;
    out[row0*8 + col_group*2 + 1] = d1;
    out[(row0+8)*8 + col_group*2]     = d2;
    out[(row0+8)*8 + col_group*2 + 1] = d3;
}

// Test 7: same as test 6 but with A row 0 = 1s for ALL threads (not conditional)
// This isolates whether the A conditional is the problem
__global__ void test7_kernel(float* out) {
    int tid = threadIdx.x;
    int row0 = tid / 4;
    int col_group = tid % 4;

    half one = __float2half(1.0f);
    half zero = __float2half(0.0f);

    // ALL threads get a0=a1={1,1}. Different from test6 where only groupID=0 does.
    uint32_t a0 = pack_half2(one, one);
    uint32_t a1 = pack_half2(one, one);
    uint32_t a2 = pack_half2(one, one);
    uint32_t a3 = pack_half2(one, one);

    // Same B as test 6
    uint32_t b0 = pack_half2(__float2half((float)(col_group*2+1)), __float2half((float)(col_group*2+2)));
    uint32_t b1 = pack_half2(__float2half((float)(8+col_group*2+1)), __float2half((float)(8+col_group*2+2)));

    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));

    out[row0*8 + col_group*2]     = d0;
    out[row0*8 + col_group*2 + 1] = d1;
    out[(row0+8)*8 + col_group*2]     = d2;
    out[(row0+8)*8 + col_group*2 + 1] = d3;
}

int main() {
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    float *d_out, h_out[128];
    cudaMalloc(&d_out, 128 * sizeof(float));

    // Tests 1-4
    struct { int id; float expect; const char* desc; } tests[] = {
        {1, 16.0f, "b0={1,1} b1={1,1}"},
        {2, 24.0f, "b0={1,1} b1={2,2}"},
        {3,  8.0f, "b0={1,1} b1={0,0}"},
        {4,  8.0f, "b0={0,0} b1={1,1}"},
    };
    for (auto& t : tests) {
        test_b1_kernel<<<1, 32>>>(d_out, t.id);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, 16, cudaMemcpyDeviceToHost);
        printf("Test %d: %s → d0=%.1f (expect %.1f) %s\n",
               t.id, t.desc, h_out[0], t.expect, h_out[0] == t.expect ? "PASS" : "FAIL");
    }

    // Test 5
    printf("\nTest 5: varying B, A=all 1s\n");
    cudaMemset(d_out, 0, 128*4);
    test_varying_b_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 32, cudaMemcpyDeviceToHost);
    printf("  Thread 0: d0=%.1f (expect 872)\n", h_out[0]);

    // Test 6: row0=1s, varying B
    printf("\nTest 6: A row0=1s rest=0, B[k,n]=k+1\n");
    cudaMemset(d_out, 0, 128*4);
    test6_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 128*4, cudaMemcpyDeviceToHost);
    printf("  D[0,0..7]: ");
    for (int j = 0; j < 8; j++) printf("%.1f ", h_out[j]);
    printf(" (expect 136.0)\n");

    // Test 7: ALL rows=1s, varying B
    printf("\nTest 7: A=all 1s, B[k,n]=k+1 (same B, different A)\n");
    cudaMemset(d_out, 0, 128*4);
    test7_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 128*4, cudaMemcpyDeviceToHost);
    printf("  D[0,0..7]: ");
    for (int j = 0; j < 8; j++) printf("%.1f ", h_out[j]);
    printf(" (expect 136.0)\n");

    cudaFree(d_out);
    return 0;
}
