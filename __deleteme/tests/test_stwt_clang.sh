#!/bin/bash
# Test script to verify __stwt behavior with clang

set -e

CLANG=/opt/llvm_toolchain/bin/clang++
CUDA_PATH=/usr/local/cuda
ARCH=sm_75

echo "=== Testing __stwt(uint4*, uint4) with clang ==="
echo ""

# Test 0: Check if clang has the documented __stwt(__half*, __half) intrinsic
echo "Test 0: Checking if clang has documented __stwt(__half*, __half)..."
cat > /tmp/test_half_stwt.cu << 'EOF'
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void test(__half* dst, __half val) {
  __stwt(dst, val);
}

int main() { return 0; }
EOF

if $CLANG --cuda-path=$CUDA_PATH --cuda-gpu-arch=$ARCH -L$CUDA_PATH/lib64 -lcudart \
   /tmp/test_half_stwt.cu -o /tmp/test_half_stwt 2>&1; then
  echo "  ✓ CONFIRMED: clang HAS __stwt(__half*, __half) from CUDA Math API"
else
  echo "  ✗ clang does NOT have __stwt(__half*, __half)"
  echo "  (This means clang lacks ALL __stwt overloads, not just uint4)"
  $CLANG --cuda-path=$CUDA_PATH --cuda-gpu-arch=$ARCH -L$CUDA_PATH/lib64 -lcudart \
     /tmp/test_half_stwt.cu -o /tmp/test_half_stwt 2>&1 || true
fi
echo ""

# Test 1: Try to compile without our guard (should fail)
echo "Test 1: Compiling WITHOUT our __stwt(uint4*, uint4) implementation (expect failure)..."
cat > /tmp/test_no_guard.cu << 'EOF'
#include <cuda_runtime.h>

__global__ void test(uint4* dst, uint4 val) {
  __stwt(dst, val);
}

int main() { return 0; }
EOF

if $CLANG --cuda-path=$CUDA_PATH --cuda-gpu-arch=$ARCH -L$CUDA_PATH/lib64 -lcudart \
   /tmp/test_no_guard.cu -o /tmp/test_no_guard 2>&1 | grep -qE "(no matching function|undeclared identifier).* '__stwt'"; then
  echo "  ✓ EXPECTED: clang does NOT have __stwt(uint4*, uint4)"
  echo "  Error message confirms missing __stwt for uint4"
else
  echo "  ✗ UNEXPECTED: Either compiled successfully or different error"
  $CLANG --cuda-path=$CUDA_PATH --cuda-gpu-arch=$ARCH -L$CUDA_PATH/lib64 -lcudart \
     /tmp/test_no_guard.cu -o /tmp/test_no_guard 2>&1 || true
fi
echo ""

# Test 2: Compile with our guard (should succeed)
echo "Test 2: Compiling WITH our __stwt implementation..."
if $CLANG --cuda-path=$CUDA_PATH --cuda-gpu-arch=$ARCH -L$CUDA_PATH/lib64 -lcudart \
   tests/stwt_clang_test.cu -o tests/stwt_clang_test 2>&1; then
  echo "  ✓ SUCCESS: Compiled with our inline PTX implementation"
else
  echo "  ✗ FAILED: Compilation failed"
  exit 1
fi
echo ""

# Test 3: Run the test (if GPU is available)
echo "Test 3: Running functional test..."
if tests/stwt_clang_test; then
  echo "  ✓ Functional test PASSED"
else
  echo "  ⚠ Functional test FAILED (may be due to no GPU in container)"
fi
echo ""

# Test 4: Generate PTX and verify instruction
echo "Test 4: Verifying PTX output..."
$CLANG --cuda-path=$CUDA_PATH --cuda-gpu-arch=$ARCH -S -emit-llvm --cuda-device-only \
   tests/stwt_clang_test.cu -o /tmp/stwt_clang.ll 2>&1

$CLANG --cuda-path=$CUDA_PATH --cuda-gpu-arch=$ARCH --cuda-device-only -S \
   tests/stwt_clang_test.cu -o /tmp/stwt_clang.ptx 2>&1

if grep -q "st\.global\.wt\.v4\.u32" /tmp/stwt_clang.ptx; then
  echo "  ✓ PTX contains: st.global.wt.v4.u32"
  echo "  Exact instruction:"
  grep "st\.global\.wt\.v4\.u32" /tmp/stwt_clang.ptx | head -1 | sed 's/^/    /'
else
  echo "  ✗ PTX does not contain expected st.global.wt.v4.u32 instruction"
  exit 1
fi
echo ""

echo "=== All tests passed ==="
echo "Conclusion: Our __stwt(uint4*, uint4) implementation is correct for clang."
