// Test file to verify __stwt behavior with nvcc
//
// Compile with nvcc to inspect PTX output:
//   nvcc --ptx -arch=sm_75 stwt_intrinsic_test.cu -o stwt_intrinsic_test.ptx
//
// Then inspect the .ptx file to see what instructions nvcc generates
// for __stwt(uint4*, uint4).
//
// Alternatively, compile to SASS:
//   nvcc -cubin -arch=sm_75 stwt_intrinsic_test.cu -o stwt_intrinsic_test.cubin
//   cuobjdump -sass stwt_intrinsic_test.cubin
//
// To run the runtime validation test:
//   nvcc -arch=sm_75 stwt_intrinsic_test.cu -o stwt_intrinsic_test && ./stwt_intrinsic_test

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================
// Test 1: Basic __stwt(uint4*, uint4) — the overload we need
// ============================================================
__global__ void test_stwt_uint4(uint4* dst, uint4 val) {
  __stwt(dst + threadIdx.x, val);
}

// ============================================================
// Test 2: __stwt with computed uint4 value (not just a parameter)
// ============================================================
__global__ void test_stwt_uint4_computed(uint4* dst, uint32_t base) {
  uint4 val;
  val.x = base;
  val.y = base + 1;
  val.z = base + 2;
  val.w = base + 3;
  __stwt(dst + threadIdx.x, val);
}

// ============================================================
// Test 3: __stwt in a loop (matches the NCCL usage pattern)
//   for (uint8_t i = 0; i < 4; i++) {
//     __stwt((uint4*)&q[idx] + i, ((uint4*)gfd)[i]);
//   }
// ============================================================
struct FakeGfd {
  uint4 chunks[4];  // 64 bytes total, like ncclGinProxyGfd_t
};

__global__ void test_stwt_uint4_loop(uint4* queue, FakeGfd* gfd) {
  // Mimics the NCCL postGfd pattern
  int idx = threadIdx.x;
  #pragma unroll
  for (uint8_t i = 0; i < 4; i++) {
    __stwt((uint4*)&queue[idx * 4] + i, ((uint4*)gfd)[i]);
  }
}

// ============================================================
// Test 4: Documented __stwt(__half*, __half) for PTX comparison
// ============================================================
__global__ void test_stwt_half(__half* dst, __half val) {
  __stwt(dst + threadIdx.x, val);
}

// ============================================================
// Test 5: Documented __stwt(__half2*, __half2) for PTX comparison
// ============================================================
__global__ void test_stwt_half2(__half2* dst, __half2 val) {
  __stwt(dst + threadIdx.x, val);
}

// ============================================================
// Test 6: Our hand-written PTX implementation for comparison
// ============================================================
__device__ __forceinline__ void stwt_handwritten(uint4* ptr, uint4 val) {
  asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               : "memory");
}

__global__ void test_stwt_handwritten(uint4* dst, uint4 val) {
  stwt_handwritten(dst + threadIdx.x, val);
}

// ============================================================
// Test 7: Both side-by-side for direct PTX comparison
// ============================================================
__global__ void test_stwt_comparison(uint4* dst_nvcc, uint4* dst_hand, uint4 val) {
  int tid = threadIdx.x;
  // nvcc's __stwt
  __stwt(dst_nvcc + tid, val);
  // Our handwritten version
  stwt_handwritten(dst_hand + tid, val);
}

// ============================================================
// Test 8: Other integer vector types — do they compile with nvcc?
// Uncomment each to test what nvcc accepts
// ============================================================

// uint2:
__global__ void test_stwt_uint2(uint2* dst, uint2 val) {
  __stwt(dst + threadIdx.x, val);
}

// uint1 (single uint):
__global__ void test_stwt_uint1(unsigned int* dst, unsigned int val) {
  __stwt(dst + threadIdx.x, val);
}

// int4:
__global__ void test_stwt_int4(int4* dst, int4 val) {
  __stwt(dst + threadIdx.x, val);
}

// int2:
__global__ void test_stwt_int2(int2* dst, int2 val) {
  __stwt(dst + threadIdx.x, val);
}

// float4:
__global__ void test_stwt_float4(float4* dst, float4 val) {
  __stwt(dst + threadIdx.x, val);
}

// float2:
__global__ void test_stwt_float2(float2* dst, float2 val) {
  __stwt(dst + threadIdx.x, val);
}

// float:
__global__ void test_stwt_float(float* dst, float val) {
  __stwt(dst + threadIdx.x, val);
}

// double:
__global__ void test_stwt_double(double* dst, double val) {
  __stwt(dst + threadIdx.x, val);
}

// unsigned long long:
__global__ void test_stwt_ull(unsigned long long* dst, unsigned long long val) {
  __stwt(dst + threadIdx.x, val);
}

// ============================================================
// Runtime validation: verify nvcc __stwt and handwritten produce
// identical results
// ============================================================
__global__ void write_with_nvcc_stwt(uint4* dst, uint4 val) {
  __stwt(dst, val);
}

__global__ void write_with_hand_stwt(uint4* dst, uint4 val) {
  stwt_handwritten(dst, val);
}

int main() {
  printf("=== __stwt(uint4*, uint4) intrinsic test ===\n\n");

  uint4* d_nvcc;
  uint4* d_hand;
  uint4 h_nvcc, h_hand;

  cudaMalloc(&d_nvcc, sizeof(uint4));
  cudaMalloc(&d_hand, sizeof(uint4));

  uint4 val = make_uint4(0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0);

  // Clear
  cudaMemset(d_nvcc, 0, sizeof(uint4));
  cudaMemset(d_hand, 0, sizeof(uint4));

  // Write with nvcc's __stwt
  write_with_nvcc_stwt<<<1, 1>>>(d_nvcc, val);
  cudaDeviceSynchronize();

  // Write with our handwritten version
  write_with_hand_stwt<<<1, 1>>>(d_hand, val);
  cudaDeviceSynchronize();

  // Read back
  cudaMemcpy(&h_nvcc, d_nvcc, sizeof(uint4), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_hand, d_hand, sizeof(uint4), cudaMemcpyDeviceToHost);

  printf("Input:       x=0x%08X y=0x%08X z=0x%08X w=0x%08X\n",
         val.x, val.y, val.z, val.w);
  printf("nvcc __stwt: x=0x%08X y=0x%08X z=0x%08X w=0x%08X\n",
         h_nvcc.x, h_nvcc.y, h_nvcc.z, h_nvcc.w);
  printf("hand PTX:    x=0x%08X y=0x%08X z=0x%08X w=0x%08X\n",
         h_hand.x, h_hand.y, h_hand.z, h_hand.w);

  bool match = (h_nvcc.x == h_hand.x && h_nvcc.y == h_hand.y &&
                h_nvcc.z == h_hand.z && h_nvcc.w == h_hand.w);
  bool correct = (h_nvcc.x == val.x && h_nvcc.y == val.y &&
                  h_nvcc.z == val.z && h_nvcc.w == val.w);

  printf("\nnvcc values correct: %s\n", correct ? "YES" : "NO");
  printf("nvcc == handwritten: %s\n", match ? "YES" : "NO");

  // Test with multiple values
  printf("\n--- Multi-value test ---\n");
  uint4 test_vals[] = {
    make_uint4(0, 0, 0, 0),
    make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
    make_uint4(1, 2, 3, 4),
    make_uint4(0x80000000, 0x7FFFFFFF, 0x00000001, 0xFFFFFFFE),
  };

  bool all_match = true;
  for (int t = 0; t < 4; t++) {
    cudaMemset(d_nvcc, 0, sizeof(uint4));
    cudaMemset(d_hand, 0, sizeof(uint4));

    write_with_nvcc_stwt<<<1, 1>>>(d_nvcc, test_vals[t]);
    write_with_hand_stwt<<<1, 1>>>(d_hand, test_vals[t]);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_nvcc, d_nvcc, sizeof(uint4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_hand, d_hand, sizeof(uint4), cudaMemcpyDeviceToHost);

    bool m = (h_nvcc.x == h_hand.x && h_nvcc.y == h_hand.y &&
              h_nvcc.z == h_hand.z && h_nvcc.w == h_hand.w);
    printf("  test[%d]: nvcc={%08X,%08X,%08X,%08X} hand={%08X,%08X,%08X,%08X} match=%s\n",
           t, h_nvcc.x, h_nvcc.y, h_nvcc.z, h_nvcc.w,
           h_hand.x, h_hand.y, h_hand.z, h_hand.w,
           m ? "YES" : "NO");
    if (!m) all_match = false;
  }

  printf("\nAll tests match: %s\n", all_match ? "PASS" : "FAIL");

  cudaFree(d_nvcc);
  cudaFree(d_hand);

  return all_match ? 0 : 1;
}
