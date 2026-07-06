---
name: eugo-upstream-merge-reference
description: Deep-reference companion to eugo-upstream-merge for the nccl-cmake fork - the per-file conflict catalog, category rules (BUILD/CODEGEN/HEADER/SOURCE/DOCA), the new-upstream-feature checklist, post-merge validation, xla_clang.patch upkeep, and the known pitfalls (__stwt, DOCA hyphen vs underscore, NCCL_USE_CMAKE, host_table.cu). Activates on "nccl conflict catalog", "per-file nccl resolution", "generate.py conflict", "DOCA headers", "__stwt", "xla_clang.patch", "nccl post-merge validation".
---

# eugo-upstream-merge-reference (nccl-cmake)

The merge RECIPE (branching, triage script, resolution order, decision
framework, adoption) lives in the eugo-upstream-merge skill. This skill is the
complete per-file / per-category catalog.

## Key files unique to this fork

- `CMakeLists.txt` — top-level CMake build (replaces upstream's `Makefile`)
- `src/CMakeLists.txt` — all `libnccl` sources in the flat `NCCL_SRC_FILES` list
- `src/device/CMakeLists.txt` — device code generation + compilation
- `xla_clang.patch` — canonical record of clang compatibility changes
- `eugo_src_diff_helper.sh` — diffs source file lists vs upstream
  (`eugo_src_diff_helper_llm_prompt.md` is its instructions)
- `ir/` — LLVM IR generation directory (eugo-specific)

## Per-file resolution table

| File | Resolution strategy |
|---|---|
| `CMakeLists.txt` (root) | Keep ours entirely. Cross-check upstream's for new `add_definitions()` / feature flags to port. |
| `src/CMakeLists.txt` | Keep ours (flat list). Add new upstream sources to `NCCL_SRC_FILES`. |
| `src/device/CMakeLists.txt` | Keep ours. Check for new generate.py arguments. Key differences: our variable names (`COLLDEVICE_GENSRC_FILES` vs `files`); we set `NCCL_USE_CMAKE=1` once via `set(ENV{NCCL_USE_CMAKE} "1")` in root CMakeLists.txt, not per-command. |
| `src/device/generate.py`, `src/device/symmetric/generate.py` | Merge carefully: accept new collectives/algos, keep the clang patches (below). |
| `src/include/nccl_device/*.h`, `.../impl/*.h` | Accept upstream (they carry `NCCL_CHECK_CUDACC` now). |
| `src/init.cc` | Accept upstream's new code, keep our `CUDAToolkit_VERSION_*` macros. |
| `src/ras/*.cc` | Accept upstream's NDEBUG removal, keep our `CUDAToolkit_VERSION_*` macros. |
| `ext-profiler/example/CMakeLists.txt` | Accept upstream — usually trivial. |
| `src/misc/`, `src/plugin/`, `src/nccl_device/`, `src/scheduler/`, `src/transport/` CMakeLists.txt | Accept upstream. Reference-only: both sides created them, but our build never invokes them (flat list wins). Don't perfect their resolution. |
| `src/transport/net_ib/gdaki/CMakeLists.txt` | **Keep ours.** Contains the DOCA header-copy logic, load-bearing for device compilation (must be invoked via `add_subdirectory` or inlined — currently inlined in `src/CMakeLists.txt`). |

## BUILD conflicts (CMakeLists.txt)

Upstream added their own CMake files as of v2.29.x, using `add_subdirectory()`
+ `PARENT_SCOPE` per directory. Ours is a flat source list. These always
conflict; the table above decides each file. When upstream adds new
`.cc`/`.cu` files, add them to `NCCL_SRC_FILES` in `src/CMakeLists.txt`,
alphabetically sorted within the matching comment section:

```cmake
# === @begin: ./subdirectory/ (`@/src/CMakeLists.txt`) ===
existing_file.cc
new_upstream_file.cc    # Added from upstream vX.Y.Z
# === @end: ./subdirectory/ (`@/src/CMakeLists.txt`) ===
```

Run `./eugo_src_diff_helper.sh` after the merge to detect: new upstream files
(add), renamed files (update entry), deleted files (remove), and new
subdirectories (add a new comment section). If upstream added new
subdirectories (e.g. `src/gin/`, `src/os/`, `src/rma/`), also add them to the
directory mappings inside `eugo_src_diff_helper.sh` itself.

## CODEGEN conflicts (generate.py)

Clang patches to ALWAYS preserve in `src/device/generate.py` (and
`symmetric/generate.py` where applicable):

```python
# 1. File extensions: .cu (NOT .cu.cc). Upstream converged on .cu, but verify:
return "%s.cu" % paste(...)       # CORRECT
return "%s.cu.cc" % paste(...)    # WRONG - reject

# 2. host_table output file: host_table.cu (NOT host_table.cc) so clang
# compiles it as CUDA. If upstream emits "host_table.cc", check whether our
# build handles it or we still need .cu.

# 3. No `const` on device function tables (clang linker requirement):
out("__device__ ncclDevFuncPtr_t ncclDevFuncTable[] = {\n")        # CORRECT
out("__device__ ncclDevFuncPtr_t const ncclDevFuncTable[] = {\n")  # WRONG

# 4. Separate pointer variables for void* casts (clang global-init workaround):
out("/*%4d*/ void* %s_ptr = (void*)%s;\n" % (index, sym, sym))  # OURS
out("/*%4d*/ %s_ptr,\n" % (index, sym))                          # ...in array
# UPSTREAM inline form compiles with nvcc but not clang:
out("/*%4d*/ (void*)%s,\n" % (index, sym))                       # reject
```

Accept from upstream: new collective operations (additions to `all_colls`,
`algos_of_coll`, `coll_camel_to_lower`, `enumerate_func_rows()`,
`best_kernel()`); new algorithms (e.g. `GinHier_MCRing` in symmetric); the
`NCCL_USE_CMAKE` conditional that skips `rules.mk` generation; new arrays
(e.g. `ncclDevKernelRequirements[]`) — but apply the pointer-variable pattern
if they contain `(void*)` casts; the `ncclGetGitVersion()` extern declaration.

Indentation: keep upstream's 2-space Python style going forward (our old
4-space causes merge noise and mixed-indent syntax errors).

## HEADER conflicts (src/include/nccl_device)

These headers guard CUDA device code. Our historical fix was
`#ifdef __CUDACC__`; upstream's v2.29.x fix is `NCCL_CHECK_CUDACC`, defined in
`src/include/nccl_device/utility.h`:

```cpp
#ifndef NCCL_CHECK_CUDACC
    #if defined(__clang__)
        #ifdef __CUDACC__
            #define NCCL_CHECK_CUDACC 1
        #else
            #define NCCL_CHECK_CUDACC 0
        #endif
    #else
        #if __CUDACC__
            #define NCCL_CHECK_CUDACC 1
        #else
            #define NCCL_CHECK_CUDACC 0
        #endif
    #endif
#endif
```

Decisions (all: take upstream):

- **`NCCL_CHECK_CUDACC`**: accept upstream's pattern — a superset of our
  `#ifdef` fix, maintained upstream. Replace our `#ifdef __CUDACC__` edits
  with `#if NCCL_CHECK_CUDACC`. Never leave a bare `#if __CUDACC__`.
- **declval**: accept upstream's `nccl::utility::declval<T>()` over our
  `cuda::std::declval<T>()` (CCCL). If it ever fails under clang, fall back
  to `cuda::std::declval` with an `@EUGO_CHANGE` annotation.
- **`__forceinline__`**: accept upstream's conditional `NCCL_DEVICE_INLINE` /
  `NCCL_HOST_DEVICE_INLINE` definitions (clang path uses
  `__attribute__((always_inline))`). Never use `__forceinline__` without a
  clang guard.
- **Renames**: upstream renamed `mem_barrier.{h,cc}` (+ `__funcs.h`,
  `__types.h`) to `lsa_barrier.*` — accept, and update `NCCL_SRC_FILES`.

Mechanical pattern per header: accept upstream's version and all new upstream
code; remove our superseded `#ifdef __CUDACC__` lines; verify no bare
`#if __CUDACC__` remains.

## SOURCE conflicts (C++ files)

- **Version macros**: upstream uses `CUDA_MAJOR`/`CUDA_MINOR` (Makefile-set);
  our CMake build defines `CUDAToolkit_VERSION_MAJOR`/`_MINOR` via
  `target_compile_definitions` in `src/CMakeLists.txt` (from
  `find_package(CUDAToolkit)`). Keep ours; replace upstream references.
  Affects `src/init.cc` (version string) and `src/ras/client_support.cc`
  (version reporting).
- **NDEBUG in `src/ras/*.cc`**: upstream removed its hardcoded
  `#define NDEBUG` lines as of v2.29.x — accept the clean deletion and drop
  our now-moot `@EUGO_CHANGE` NDEBUG comment blocks. Never hardcode
  `#define NDEBUG` in sources (toolchain sets it).
- General rule: accept all new upstream functionality; re-apply our macro
  renames; re-apply still-relevant `@EUGO_CHANGE` annotations; remove
  annotations whose upstream code has been fixed.

## DOCA GPUNetIO headers and CUDA file extensions

Upstream's GDAKI networking (`src/transport/net_ib/gdaki/doca-gpunetio/`)
brings `.cuh` device headers plus a naming trap: the physical directory is
`doca-gpunetio` (hyphen) but includes use `doca_gpunetio` (underscore), e.g.
`gin_gdaki.h` (at `src/include/nccl_device/gin/gdaki/`) does
`#include "doca_gpunetio/doca_gpunetio_device.h"`. Headers must be COPIED to
an underscore path — do not "fix" the includes (breaks upstream compat).

- Upstream copies to
  `${CMAKE_BINARY_DIR}/include/nccl_device/gin/gdaki/doca_gpunetio/` and adds
  that parent dir to include paths.
- **We copy to `${CMAKE_BINARY_DIR}/include/doca_gpunetio/`** because
  `${CMAKE_BINARY_DIR}/include` is already on our include path (for
  configured headers) — fewer include paths, same resolution: quoted include
  misses next to `gin_gdaki.h`, then hits our copy via the include path.
- The copy logic is **inlined in `src/CMakeLists.txt`** (section "DOCA
  GPUNetIO Header Copying"); `src/transport/net_ib/gdaki/CMakeLists.txt` is
  kept as reference. All DOCA sources are in the flat `NCCL_SRC_FILES`.
- Headers copied (from `.../gdaki/doca-gpunetio/include/`): top-level
  `doca_gpunetio_device.h` (+ `_host.h`, `_config.h`); `common/`
  `doca_gpunetio_verbs_def.h` + `_dev.h`; `device/` five `.cuh` files
  (`..._dev_verbs_common.cuh`, `_counter.cuh`, `_cq.cuh`, `_onesided.cuh`,
  `_qp.cuh`).
- Host side: `gin_host_gdaki.cc` includes `doca_gpunetio_host.h`, so
  `src/transport/net_ib/gdaki/doca-gpunetio/include` must be on the `nccl`
  target's include path.
- Missing copy symptom:
  `fatal error: 'doca_gpunetio/doca_gpunetio_device.h' file not found`.

### .cu vs .cc/.cpp rules

A source file MUST be `.cu` (compiled as CUDA by clang) if ANY of: it contains
`__device__`/`__global__`/`__host__` qualifiers; it includes `.cuh` headers;
it uses `<<<...>>>` launch syntax; it uses device built-ins (`__syncthreads()`,
`threadIdx`, ...). Files that merely call the CUDA runtime API or
`dlopen()` CUDA symbols stay host code. Current state: all 13
`doca-gpunetio/src/*.cpp` are pure host code (`doca_verbs_cuda_wrapper.cpp` is
a dlopen wrapper) and correctly remain `.cpp` — do NOT rename them. The
`.cuh` files are compiled only through the device header chain
(`gin_device_api.h` -> `gin_gdaki.h` -> `doca_gpunetio_device.h`) which flows
into generate.py-generated `.cu` files built by the `nccl_colldevice` target;
`src/device/symmetric/*.cuh` (6 files) likewise reach only generated `.cu`.

Audit greps for every merge:

```bash
grep -rn '\.cuh' src/ --include="*.cc" --include="*.cpp"          # .cuh included by host files?
grep -rln '__device__\|__global__' src/ --include="*.cc" --include="*.cpp"
find src/ -name "*.cuh" | sort                                     # new .cuh files?
```

Any hit in the first two means that file must be renamed to `.cu` and its
`NCCL_SRC_FILES` entry updated.

## New-upstream-feature checklist

- **New source files** -> `NCCL_SRC_FILES`; detect with
  `./eugo_src_diff_helper.sh`.
- **New subdirectories** (`src/gin/`, `src/rma/`, `src/os/`,
  `src/scheduler/`, ...) -> new comment section in `NCCL_SRC_FILES`; do NOT
  add a per-dir CMakeLists for our build.
- **New CUDA device sources** shipped as `.cc`/`.cpp` -> rename to `.cu`
  (rules above) and list the `.cu` name.
- **New `.cuh` headers** -> find their includers: included by `.cc`/`.cpp`
  -> rename those to `.cu`; included only via the device header chain -> no
  rename; under `doca-gpunetio/include/` -> extend the DOCA copy logic.
- **New collectives** (additions to `all_colls`) propagate to generate.py,
  symmetric/generate.py, C++ enqueue/transport code, and headers.
- **New CMake `option()` / cache vars** -> do not carry the option: evaluate,
  keep only the selected path live, comment the unused branch as
  `# @NVIDIA_ORIGINAL:`, explain with `# @EUGO_CHANGE:`, and set any
  associated `-D` macro via `NCCL_COMMON_COMPILE_DEFINITIONS`. Example:

  ```cmake
  # @NVIDIA_ORIGINAL: option(EMIT_LLVM_IR "Generate LLVM IR" OFF)
  # @NVIDIA_ORIGINAL: if(EMIT_LLVM_IR)
  # @EUGO_CHANGE: We always emit LLVM IR, so the option is removed and
  # the body is unconditionally included.
  add_subdirectory(ir)
  # @NVIDIA_ORIGINAL: endif()

  # @EUGO_CHANGE: Upstream guards this behind an option(); we always enable it.
  list(APPEND NCCL_COMMON_COMPILE_DEFINITIONS EMIT_LLVM_IR=1)
  ```

- **New compile definitions**: check upstream's root CMakeLists AND
  `Makefile` for new `-D` flags; port to `NCCL_COMMON_COMPILE_DEFINITIONS`.
- **New dependencies**: new `find_package()` / library links -> add to
  `NCCL_DEPENDENCIES` if needed.

## What NOT to do

- Do NOT import upstream's `add_subdirectory()` + `PARENT_SCOPE` pattern into
  our `src/CMakeLists.txt`.
- Do NOT accept a bare `#if __CUDACC__` — must be `#if NCCL_CHECK_CUDACC`.
- Do NOT add `const` back to `ncclDevFuncTable[]` / `ncclDevKernelList[]`.
- Do NOT use `__forceinline__` without a clang guard.
- Do NOT hardcode `#define NDEBUG` in source files.
- Do NOT import `enhcompat.cc` (it overrides cudart symbols with crashing
  stubs; we link libcudart directly).
- Do NOT rename `doca-gpunetio/src/*.cpp` to `.cu` — pure host code.
- Do NOT forget the DOCA header copying step.

## Post-merge gate

```bash
./eugo_src_diff_helper.sh
cmake -B eugo_build -S . -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=clang++ && ninja -C eugo_build
grep -rn '<<<<<<<\|=======\|>>>>>>>' src/ CMakeLists.txt
find src/device -name "*.cu.cc"                          # must be empty
grep -n 'const ncclDevFuncTable\|const ncclDevKernelList' src/device/generate.py  # must be empty
```

Then symbol parity vs the PyPI wheel, the install-tree comparison vs the
official package layout, and the smoke tests — all in eugo-build-and-test.
Finally: adoption is a separate step (bump
`protomolecule/dependencies/native/cuda_nccl/meta.json` commit pin) — covered
by eugo-upstream-merge.

### xla_clang.patch upkeep

After merging, regenerate the patch if our clang changes evolved:

```bash
git diff upstream/master -- src/device/generate.py src/device/common.h \
  src/device/symmetric/generate.py > xla_clang.patch.new
# Review and replace xla_clang.patch if substantially changed
```

## Pitfalls that recur

1. New upstream sources invisible to our explicit list — run
   `./eugo_src_diff_helper.sh` every merge.
2. generate.py indentation mixing (prefer upstream 2-space).
3. `NCCL_USE_CMAKE` env contract between root CMakeLists and generate.py —
   breaks codegen SILENTLY if upstream renames the check.
4. Renames shown as delete+add — check
   `git log --diff-filter=R upstream/master`.
5. New plugin source files may need `SKIP_UNITY_BUILD_INCLUSION` properties
   to avoid ODR violations in unity builds (we don't enable unity — see
   eugo-cmake-review — but keep the property discipline).
6. DOCA: physical dir `doca-gpunetio` (hyphen) vs include path
   `doca_gpunetio` (underscore); we copy headers to
   `${CMAKE_BINARY_DIR}/include/doca_gpunetio/`. Missing copy =
   `fatal error: 'doca_gpunetio/doca_gpunetio_device.h' file not found`.
7. Not every `.cpp` near CUDA code needs `.cu` — only rename if it really has
   `__device__`/`__global__` or includes `.cuh`. Grep before renaming.
8. `__stwt(uint4*, uint4)` — clang 23+ ships it natively in
   `__clang_cuda_intrinsics.h`; the fork's temporary inline-PTX workaround in
   `src/include/nccl_device/gin/proxy/gin_proxy.h` was REMOVED 2026-07-06
   (redefinition error otherwise). Do NOT re-add our unguarded copy on merges;
   upstream master's own fallback is guarded `__clang_major__ < 21` (inert for
   us) and is fine to take verbatim (see `__deleteme/STWT_INVESTIGATION.md`).

## Sync history

| Date | Upstream | Branch | Notes |
|---|---|---|---|
| 2026-02-08 | v2.29.3-1 | `NVIDIA-upstream-master-02-26` | 29 conflicts. Upstream added CMake build (`add_subdirectory` style), `NCCL_CHECK_CUDACC`, DOCA GPUNetIO headers. Key issues: DOCA header copy path (`build/include/doca_gpunetio/` vs upstream's nested path), missing `DOCA_VERBS_USE_IBV_WRAPPER` define, `CUmemGenericAllocationHandle` nullptr->0 fix, `ptr__funcs.h` bad merge artifacts (orphaned templates), `__stwt` missing in clang 20 (inline PTX added, later removed). |

After each merge, update: the category rules here if upstream introduced new
conflict patterns; the feature checklist if new categories emerged; the
per-file table if new files need resolution guidance; and the sync-history
row.

## Related

- eugo-upstream-merge (recipe, triage, decision framework, adoption),
  eugo-cmake-review (annotation and invariant checks on the resolved tree),
  eugo-build-and-test (build, symbol parity, install-tree comparison).
