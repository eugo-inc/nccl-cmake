/*************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMPILER_GCC_H
#define NCCL_COMPILER_GCC_H

// Helper macros to convert C++ memory ordering to GCC atomic ordering
#define NCCL_CONVERT_ORDER(order) \
  ((order) == std::memory_order_relaxed ? __ATOMIC_RELAXED : \
   (order) == std::memory_order_consume ? __ATOMIC_CONSUME : \
   (order) == std::memory_order_acquire ? __ATOMIC_ACQUIRE : \
   (order) == std::memory_order_release ? __ATOMIC_RELEASE : \
   (order) == std::memory_order_acq_rel ? __ATOMIC_ACQ_REL : \
   (order) == std::memory_order_seq_cst ? __ATOMIC_SEQ_CST : \
   __ATOMIC_SEQ_CST)

// @EUGO_CHANGE: @begin: Clang atomic instruction generation fix for packed structs
//
// PROBLEM:
// When taking a pointer to a member of a __attribute__((packed)) struct (e.g., 
// ncclGinProxyGfd_t), the compiler assumes alignment=1 (byte-aligned) because 
// packed forces minimum alignment. Even though we use __atomic_load_n (which 
// should inline for lock-free types like uint64_t), clang cannot emit inline 
// LDAR/STLR instructions for potentially misaligned pointers. Instead, it emits 
// calls to the library functions __atomic_load(size, src, dest, order) and 
// __atomic_store(size, dest, val, order), which require linking libatomic.
//
// SOLUTION:
// Wrap the pointer with __builtin_assume_aligned(ptr, sizeof(*ptr)) to tell
// the compiler "this pointer IS naturally aligned to its type's size." This
// is safe because:
//   1. The actual data IS properly aligned (heap-allocated via cudaMalloc/malloc)
//   2. The packed attribute is redundant (struct is already naturally aligned)
//   3. __builtin_assume_aligned is just an optimizer hint, doesn't dereference
//
// The (void*) cast is needed because __builtin_assume_aligned expects const void*,
// but some callers pass volatile uint32_t*. After assume_aligned, we cast back
// to __typeof__(ptr) to restore the original type (including volatile).
//
// POINTER TRANSFORMATION (key change):
//   BEFORE: (ptr)
//   AFTER:  (__typeof__(ptr))__builtin_assume_aligned((void*)(ptr), sizeof(*(ptr)))
//
// BEFORE (original upstream macros):
//   #define COMPILER_ATOMIC_LOAD(ptr, order) \
//     __atomic_load_n((ptr), NCCL_CONVERT_ORDER(order))
//   #define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) do { \
//     __atomic_load((ptr), (dest), NCCL_CONVERT_ORDER(order)); /* library call! */ \
//   } while(0)
//   #define COMPILER_ATOMIC_STORE(ptr, val, order) \
//     __atomic_store_n((ptr), (val), NCCL_CONVERT_ORDER(order))
//
// AFTER (with __builtin_assume_aligned wrapper, inline LDAR/STLR instructions):
#define COMPILER_ATOMIC_LOAD(ptr, order) \
  // @EUGO_CHANGE: Changed from __atomic_load_n(ptr, order) to __atomic_load_n((__typeof__(ptr))__builtin_assume_aligned((void*)(ptr), sizeof(*(ptr))), order) to fix clang codegen for atomics on packed struct members. See detailed explanation above.
  __atomic_load_n((__typeof__(ptr))__builtin_assume_aligned((void*)(ptr), sizeof(*(ptr))), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) do { \
  // @EUGO_CHANGE: Changed from __atomic_load(ptr, dest, order) to *(dest) = __atomic_load_n((__typeof__(ptr))__builtin_assume_aligned((void*)(ptr), sizeof(*(ptr))), order) to fix clang codegen for atomics on packed struct members. See detailed explanation above.
  *(dest) = __atomic_load_n((__typeof__(ptr))__builtin_assume_aligned((void*)(ptr), sizeof(*(ptr))), NCCL_CONVERT_ORDER(order)); \
} while(0)
#define COMPILER_ATOMIC_STORE(ptr, val, order) \
  // @EUGO_CHANGE: Changed from __atomic_store_n(ptr, val, order) to __atomic_store_n((__typeof__(ptr))__builtin_assume_aligned((void*)(ptr), sizeof(*(ptr))), (val), order) to fix clang codegen for atomics on packed struct members. See detailed explanation above.
   __atomic_store_n((__typeof__(ptr))__builtin_assume_aligned((void*)(ptr), sizeof(*(ptr))), (val), NCCL_CONVERT_ORDER(order))
  __atomic_store_n((__typeof__(ptr))__builtin_assume_aligned((void*)(ptr), sizeof(*(ptr))), (val), NCCL_CONVERT_ORDER(order))
//
// ADDITIONAL FIX in COMPILER_ATOMIC_LOAD_DEST:
// Changed from __atomic_load(ptr, dest, order) [3-argument generic form] to
// *(dest) = __atomic_load_n(ptr, order) [2-argument natural-size form + assign].
// The _n form returns the value (allowing inline code generation), while the
// generic form writes to dest via pointer (forcing a library call).
//
// RESULT:
// - Zero library dependencies (no libatomic needed)
// - Inline atomic instructions for lock-free types (uint32_t, uint64_t)
// - Works with both regular pointers and pointers from packed structs
// - Preserves volatile semantics via __typeof__
// @EUGO_CHANGE: @end: Clang atomic instruction generation fix for packed structs
#define COMPILER_ATOMIC_EXCHANGE(ptr, val, order) \
  __atomic_exchange_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired, success_order, failure_order) \
  __atomic_compare_exchange_n((ptr), (expected), (desired), true, NCCL_CONVERT_ORDER(success_order), NCCL_CONVERT_ORDER(failure_order))

#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) __atomic_fetch_add((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_ADD_FETCH(ptr, val, order) __atomic_add_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_SUB_FETCH(ptr, val, order) __atomic_sub_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))

#define COMPILER_PREFETCH(addr) __builtin_prefetch((addr))

#define COMPILER_POPCOUNT32(x) __builtin_popcount(x)
#define COMPILER_POPCOUNT64(x) __builtin_popcountll(x)

#define COMPILER_EXPECT(x, v) __builtin_expect((x), (v))

// Find First Set (FFS) - returns index of first set bit (1-indexed), 0 if no bits set
#define COMPILER_FFS(x) __builtin_ffs(x)
#define COMPILER_FFSL(x) __builtin_ffsl(x)
#define COMPILER_FFSLL(x) __builtin_ffsll(x)

// Count Leading Zeros (CLZ) - undefined behavior if x == 0
#define COMPILER_CLZ(x) __builtin_clz(x)
#define COMPILER_CLZL(x) __builtin_clzl(x)
#define COMPILER_CLZLL(x) __builtin_clzll(x)

// Byte Swap
#define COMPILER_BSWAP16(x) __builtin_bswap16(x)
#define COMPILER_BSWAP32(x) __builtin_bswap32(x)
#define COMPILER_BSWAP64(x) __builtin_bswap64(x)

// Compiler hints
#define COMPILER_ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned((ptr), (alignment))

#endif // NCCL_COMPILER_GCC_H
