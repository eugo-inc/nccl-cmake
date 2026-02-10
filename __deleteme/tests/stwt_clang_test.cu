// Test file to verify __stwt behavior with clang
//
// Compile with clang:
//   /opt/llvm_toolchain/bin/clang++ --cuda-gpu-arch=sm_75 -L/usr/local/cuda/lib64 -lcudart -lcudart_static -ldl -lrt -lpthread stwt_clang_test.cu -o stwt_clang_test
//
// Or use the CUDA path setup:
//   clang++ --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_75 -L/usr/local/cuda/lib64 -lcudart stwt_clang_test.cu -o stwt_clang_test

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================
// Our clang-compatible __stwt(uint4*, uint4) implementation
// ============================================================
#if defined(__clang__) && defined(__CUDA__)
__device__ __forceinline__ void __stwt(uint4* ptr, uint4 val) {
  asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
               :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               : "memory");
}
#endif

// ============================================================
// Test 1: Basic __stwt(uint4*, uint4) — should work with our guard
// ============================================================
__global__ void test_stwt_uint4(uint4* dst, uint4 val) {
  if (threadIdx.x == 0) {
    __stwt(dst, val);
  }
}

// ============================================================
// Test 2: __stwt in a loop (matches the NCCL usage pattern)
// ============================================================
struct FakeGfd {
  uint4 chunks[4];  // 64 bytes total
};

__global__ void test_stwt_uint4_loop(uint4* queue, FakeGfd* gfd) {
  if (threadIdx.x == 0) {
    int idx = 0;
    #pragma unroll
    for (uint8_t i = 0; i < 4; i++) {
      __stwt((uint4*)&queue[idx * 4] + i, gfd->chunks[i]);
    }
  }
}

// ============================================================
// Test 3: Documented __stwt(__half*, __half) — should exist in clang CUDA
// ============================================================
__global__ void test_stwt_half(__half* dst, __half val) {
  if (threadIdx.x == 0) {
    __stwt(dst, val);
  }
}

// ============================================================
// Test 4: Compare behavior vs normal store
// ============================================================
__global__ void test_normal_store(uint4* dst, uint4 val) {
  if (threadIdx.x == 0) {
    *dst = val;
  }
}

// ============================================================
// Runtime validation
// ============================================================
int main() {
  printf("=== __stwt(uint4*, uint4) clang CUDA test ===\n\n");

  // Check CUDA availability
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    printf("ERROR: No CUDA devices available: %s\n", cudaGetErrorString(err));
    printf("This test requires a CUDA GPU to run.\n");
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Using device: %s (compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

  uint4* d_stwt_buf;
  uint4* d_normal_buf;
  uint4 h_stwt, h_normal;

  err = cudaMalloc(&d_stwt_buf, sizeof(uint4));
  if (err != cudaSuccess) {
    printf("ERROR: cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  
  err = cudaMalloc(&d_normal_buf, sizeof(uint4));
  if (err != cudaSuccess) {
    printf("ERROR: cudaMalloc failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_stwt_buf);
    return 1;
  }

  uint4 test_vals[] = {
    make_uint4(0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0),
    make_uint4(0, 0, 0, 0),
    make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
    make_uint4(1, 2, 3, 4),
    make_uint4(0x80000000, 0x7FFFFFFF, 0x00000001, 0xFFFFFFFE),
  };

  bool all_pass = true;

  for (int t = 0; t < 5; t++) {
    uint4 val = test_vals[t];
    
    // Clear buffers
    cudaMemset(d_stwt_buf, 0xCC, sizeof(uint4));
    cudaMemset(d_normal_buf, 0xCC, sizeof(uint4));

    // Test __stwt
    test_stwt_uint4<<<1, 32>>>(d_stwt_buf, val);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("ERROR: __stwt kernel failed: %s\n", cudaGetErrorString(err));
      all_pass = false;
      continue;
    }

    // Test normal store
    test_normal_store<<<1, 32>>>(d_normal_buf, val);
    cudaDeviceSynchronize();

    // Read back
    cudaMemcpy(&h_stwt, d_stwt_buf, sizeof(uint4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_normal, d_normal_buf, sizeof(uint4), cudaMemcpyDeviceToHost);

    bool stwt_ok = (h_stwt.x == val.x && h_stwt.y == val.y && h_stwt.z == val.z && h_stwt.w == val.w);
    bool normal_ok = (h_normal.x == val.x && h_normal.y == val.y && h_normal.z == val.z && h_normal.w == val.w);
    bool match = (h_stwt.x == h_normal.x && h_stwt.y == h_normal.y && 
                  h_stwt.z == h_normal.z && h_stwt.w == h_normal.w);

    printf("Test[%d]:\n", t);
    printf("  Input:  {%08X, %08X, %08X, %08X}\n", val.x, val.y, val.z, val.w);
    printf("  __stwt: {%08X, %08X, %08X, %08X} %s\n", 
           h_stwt.x, h_stwt.y, h_stwt.z, h_stwt.w, stwt_ok ? "✓" : "✗");
    printf("  normal: {%08X, %08X, %08X, %08X} %s\n", 
           h_normal.x, h_normal.y, h_normal.z, h_normal.w, normal_ok ? "✓" : "✗");
    printf("  Match:  %s\n\n", match ? "YES" : "NO");

    if (!stwt_ok || !match) all_pass = false;
  }

  // Test loop pattern
  printf("Testing loop pattern (4x __stwt):\n");
  FakeGfd h_gfd;
  FakeGfd* d_gfd;
  uint4* d_queue;
  uint4 h_queue[4];

  h_gfd.chunks[0] = make_uint4(0x11111111, 0x22222222, 0x33333333, 0x44444444);
  h_gfd.chunks[1] = make_uint4(0x55555555, 0x66666666, 0x77777777, 0x88888888);
  h_gfd.chunks[2] = make_uint4(0x99999999, 0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC);
  h_gfd.chunks[3] = make_uint4(0xDDDDDDDD, 0xEEEEEEEE, 0xFFFFFFFF, 0x00000000);

  cudaMalloc(&d_gfd, sizeof(FakeGfd));
  cudaMalloc(&d_queue, 4 * sizeof(uint4));
  cudaMemcpy(d_gfd, &h_gfd, sizeof(FakeGfd), cudaMemcpyHostToDevice);
  cudaMemset(d_queue, 0, 4 * sizeof(uint4));

  test_stwt_uint4_loop<<<1, 32>>>(d_queue, d_gfd);
  err = cudaDeviceSynchronize();
  
  if (err != cudaSuccess) {
    printf("ERROR: Loop test kernel failed: %s\n", cudaGetErrorString(err));
    all_pass = false;
  } else {
    cudaMemcpy(h_queue, d_queue, 4 * sizeof(uint4), cudaMemcpyDeviceToHost);
    
    bool loop_ok = true;
    for (int i = 0; i < 4; i++) {
      bool chunk_ok = (h_queue[i].x == h_gfd.chunks[i].x &&
                       h_queue[i].y == h_gfd.chunks[i].y &&
                       h_queue[i].z == h_gfd.chunks[i].z &&
                       h_queue[i].w == h_gfd.chunks[i].w);
      printf("  Chunk[%d]: {%08X, %08X, %08X, %08X} %s\n",
             i, h_queue[i].x, h_queue[i].y, h_queue[i].z, h_queue[i].w,
             chunk_ok ? "✓" : "✗");
      if (!chunk_ok) loop_ok = false;
    }
    printf("  Loop test: %s\n\n", loop_ok ? "PASS" : "FAIL");
    if (!loop_ok) all_pass = false;
  }

  cudaFree(d_queue);
  cudaFree(d_gfd);
  cudaFree(d_stwt_buf);
  cudaFree(d_normal_buf);

  // Test __half variant (should exist in clang)
  printf("Testing documented __stwt(__half*, __half):\n");
  __half* d_half;
  __half h_half_result;
  __half h_half_val = __float2half(3.14159f);

  cudaMalloc(&d_half, sizeof(__half));
  cudaMemset(d_half, 0, sizeof(__half));
  
  test_stwt_half<<<1, 32>>>(d_half, h_half_val);
  err = cudaDeviceSynchronize();
  
  if (err != cudaSuccess) {
    printf("ERROR: __half test failed: %s\n", cudaGetErrorString(err));
    printf("  (This may indicate clang doesn't have __stwt for __half either)\n");
  } else {
    cudaMemcpy(&h_half_result, d_half, sizeof(__half), cudaMemcpyDeviceToHost);
    float f_expected = __half2float(h_half_val);
    float f_result = __half2float(h_half_result);
    bool half_ok = (f_result == f_expected);
    printf("  Expected: %f, Got: %f %s\n", f_expected, f_result, half_ok ? "✓" : "✗");
  }
  
  cudaFree(d_half);

  printf("\n=== Summary ===\n");
  printf("Overall: %s\n", all_pass ? "PASS ✓" : "FAIL ✗");
  
#if defined(__clang__)
  printf("\nCompiled with: clang\n");
  printf("Clang version: %s\n", __VERSION__);
#else
  printf("\nCompiled with: nvcc or other\n");
#endif

  return all_pass ? 0 : 1;
}
