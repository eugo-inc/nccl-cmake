# NCCL Upstream Merge Agent Guide

> Dual-purpose document: human-readable merge playbook **and** AI agent instructions for resolving upstream NVIDIA/nccl merge conflicts into the eugo-inc/nccl-cmake fork.

---

## 1. Fork Identity & Non-Negotiable Rules

This fork exists to build NCCL with **clang only** (no nvcc, no gcc) using **CMake only** (no Makefiles for compilation). Every merge resolution must preserve these invariants:

| Rule | Detail |
|---|---|
| **R1: Clang-only compilation** | All host and device code compiles with clang. No nvcc-specific syntax, no gcc extensions that clang rejects. |
| **R2: CMake-only build** | `CMakeLists.txt` is the source of truth. Upstream `Makefile` logic is informational only—never imported verbatim. |
| **R3: Flat source list in `src/CMakeLists.txt`** | All library sources are listed in a single `NCCL_SRC_FILES` variable in `src/CMakeLists.txt`, organized by subdirectory comments. We do NOT use per-subdirectory `CMakeLists.txt` + `PARENT_SCOPE` for the main `libnccl`. |
| **R4: `@EUGO_CHANGE` / `@NVIDIA_ORIGINAL` annotations** | Every deviation from upstream must be annotated with `# @EUGO_CHANGE:` explaining why. When commenting out upstream code, prefix with `# @NVIDIA_ORIGINAL:` so the original intent is preserved for future merges. |
| **R5: Preserve `xla_clang.patch` changes** | The patch in `xla_clang.patch` defines our clang compatibility delta. Its transformations must survive every merge. |
| **R6: Reduce CMake options to selected path** | When upstream introduces `option()`, `set(... CACHE ...)`, or `if()`/`else()` conditional blocks, evaluate the option, pick the path we want, comment out the unused branch with `# @NVIDIA_ORIGINAL:`, and keep the selected path live. Set any associated `-D` macros via `NCCL_COMMON_COMPILE_DEFINITIONS`. See Section 5 "New CMake options from upstream" for the full procedure and examples. |

---

## 2. Repository Layout

```
origin   = https://github.com/eugo-inc/nccl-cmake.git   (our fork)
upstream = https://github.com/NVIDIA/nccl.git            (NVIDIA's repo)

Branches:
  master               — our stable branch (clang + CMake)
  NVIDIA-master-cmake  — working branch for CMake integration
  NVIDIA-master        — pristine mirror of upstream master
```

### Key files unique to our fork
- `CMakeLists.txt` — top-level CMake build (replaces upstream's `Makefile`)
- `src/CMakeLists.txt` — all `libnccl` sources in flat list
- `src/device/CMakeLists.txt` — device code generation + compilation
- `xla_clang.patch` — canonical record of clang compatibility changes
- `eugo_src_diff_helper.sh` — script to diff source file lists vs upstream
- `eugo_src_diff_helper_llm_prompt.md` — instructions for the diff script
- `ir/` — LLVM IR generation directory (eugo-specific)

---

## 3. Merge Workflow (Step by Step)

### 3.1 Preparation

```bash
# Fetch upstream
git fetch upstream

# Create a merge branch from our master
git checkout master
git checkout -b NVIDIA-upstream-merge-$(date +%m-%d)

# Merge upstream master
git merge upstream/master
```

### 3.2 Triage Conflicts

Run this to categorize conflicts:

```bash
git diff --name-only --diff-filter=U | while read f; do
  case "$f" in
    *CMakeLists.txt)          echo "BUILD:   $f" ;;
    src/device/generate.py|src/device/symmetric/generate.py)
                              echo "CODEGEN: $f" ;;
    src/include/nccl_device/*) echo "HEADER:  $f" ;;
    *.cc|*.cu|*.c|*.h)        echo "SOURCE:  $f" ;;
    *)                        echo "OTHER:   $f" ;;
  esac
done
```

### 3.3 Resolve Conflicts by Category

Resolve in this order (dependencies flow downward):

1. **HEADER** — nccl_device headers (most mechanical)
2. **CODEGEN** — generate.py files (requires understanding both codegen pipelines)
3. **BUILD** — CMakeLists.txt files (depends on knowing new source files from steps 1-2)
4. **SOURCE** — C++ source files (mostly local macro name changes)
5. **OTHER** — documentation, config, etc.

### 3.4 Post-Merge Validation

```bash
# 1. Run the source diff helper to catch missing files
./eugo_src_diff_helper.sh

# 2. Build
cmake -B build -DCMAKE_CUDA_COMPILER=clang++ -DCMAKE_CXX_COMPILER=clang++
cmake --build build

# 3. Check for orphaned conflict markers
grep -rn '<<<<<<\|======\|>>>>>>' src/ CMakeLists.txt

# 4. Verify xla_clang.patch changes are intact
# Check .cu extensions (not .cu.cc) in device files
find src/device -name "*.cu.cc" | head  # should return nothing

# Check const removal in generate.py
grep -n 'const ncclDevFuncTable' src/device/generate.py  # should find nothing
grep -n 'const ncclDevKernelList' src/device/generate.py  # should find nothing

# 5. Compare install tree against upstream's official package layout
# See section 8 "Comparing install tree against upstream" for full details
tree <our_install_prefix> | sed 's|<our_install_prefix>|/usr|' | sort > our_tree.txt
# Compare against reference tree from `dnf repoquery -l libnccl-devel`
diff ref_tree.txt our_tree.txt
```

---

## 4. Conflict Resolution Rules by Category

### 4.1 CMakeLists.txt Conflicts (BUILD)

**Context:** Upstream NVIDIA has started adding their own `CMakeLists.txt` files (as of v2.29.x). Their architecture uses `add_subdirectory()` + `PARENT_SCOPE` variables per directory. Ours uses a flat source list in `src/CMakeLists.txt`. These will always conflict.

#### Resolution Rules

| Situation | Action |
|---|---|
| **`src/CMakeLists.txt`** | Always keep our flat-list architecture. Extract new source files from upstream's changes and add them to `NCCL_SRC_FILES` in the correct comment section. |
| **`src/device/CMakeLists.txt`** | Keep our version. Key differences: our variable names (`COLLDEVICE_GENSRC_FILES` vs `files`), we set `NCCL_USE_CMAKE=1` via `set(ENV{NCCL_USE_CMAKE} "1")` in root CMakeLists.txt, not per-command. |
| **`CMakeLists.txt` (root)** | Keep ours entirely. Cross-check upstream's for new `add_definitions()` or feature flags that need porting. |
| **Subdirectory CMakeLists.txt** (`src/misc/`, `src/plugin/`, etc.) | These are upstream's. They conflict with ours because both sides created them. We don't use them (our flat list is in `src/CMakeLists.txt`), but we keep them in the tree for reference. Accept upstream's version, but they are NOT authoritative for our build. |

#### How to add new source files

When upstream adds new `.cc`/`.cu` files, add them to `src/CMakeLists.txt` in `NCCL_SRC_FILES`:

```cmake
# === @begin: ./subdirectory/ (`@/src/CMakeLists.txt`) ===
existing_file.cc
new_upstream_file.cc    # Added from upstream vX.Y.Z
# === @end: ./subdirectory/ (`@/src/CMakeLists.txt`) ===
```

Keep files alphabetically sorted within each section.

#### Checklist for new upstream files

Run `eugo_src_diff_helper.sh` after merge to detect:
- **New files in upstream** → add to `NCCL_SRC_FILES` in `src/CMakeLists.txt`
- **Renamed files** → update the entry in `NCCL_SRC_FILES`
- **Deleted files** → remove from `NCCL_SRC_FILES`
- **New subdirectories** (e.g., `src/gin/`, `src/rma/`, `src/os/`) → add a new comment section in `NCCL_SRC_FILES`

---

### 4.2 Device Code / generate.py Conflicts (CODEGEN)

**Context:** `src/device/generate.py` and `src/device/symmetric/generate.py` generate CUDA kernel source files. Our fork patches these for clang compatibility. Upstream frequently adds new collective operations and kernel variants here.

#### Critical eugo changes to ALWAYS preserve in `generate.py`

```python
# 1. File extensions: .cu (NOT .cu.cc)
# Upstream may still use .cu now (they converged), but verify:
return "%s.cu" % paste(...)       # CORRECT
return "%s.cu.cc" % paste(...)    # WRONG — reject

# 2. host_table output file: host_table.cu (NOT host_table.cc)
# This is needed for clang to compile it as CUDA
names = impl_names + ["host_table.cc", "device_table.cu"]
# We need host_table to be treated as CUDA code, so if upstream uses .cc,
# check if our build handles it or if we still need .cu

# 3. Remove `const` from device function tables (clang linker requirement)
out("__device__ ncclDevFuncPtr_t ncclDevFuncTable[] = {\n")      # CORRECT
out("__device__ ncclDevFuncPtr_t const ncclDevFuncTable[] = {\n") # WRONG

# 4. Separate pointer variables for void* casts (clang global init workaround)
# OURS (correct for clang):
out("/*%4d*/ void* %s_ptr = (void*)%s;\n" % (index, sym, sym))
# ...later in array:
out("/*%4d*/ %s_ptr,\n" % (index, sym))

# UPSTREAM (compiles with nvcc but not clang):
out("/*%4d*/ (void*)%s,\n" % (index, sym))
```

#### What to accept from upstream

- **New collective operations** (e.g., `AllGatherV`): Accept additions to `all_colls`, `algos_of_coll`, `coll_camel_to_lower`, `enumerate_func_rows()`, `best_kernel()`
- **New algorithms** (e.g., `GinHier_MCRing` in symmetric): Accept new algo lists and kernel generation logic
- **`NCCL_USE_CMAKE` conditional for `rules.mk`**: Accept — we want `rules.mk` generation skipped when `NCCL_USE_CMAKE=1`
- **New arrays** (e.g., `ncclDevKernelRequirements[]`): Accept, but apply the pointer-variable pattern if they contain `(void*)` casts
- **`ncclGetGitVersion()`**: Accept the extern declaration

#### Indentation

Our fork uses 4-space indentation in Python. Upstream uses 2-space. **Keep upstream's indentation** going forward to minimize future conflicts. If our changes are minimal (just the clang patches), maintaining upstream's style reduces merge noise.

---

### 4.3 nccl_device Header Conflicts (HEADER)

**Context:** These headers contain CUDA device code with preprocessor guards for `__CUDACC__`. Our fork changed `#if __CUDACC__` to `#ifdef __CUDACC__` for clang compatibility. As of v2.29.x, upstream introduced their own fix: `NCCL_CHECK_CUDACC`.

#### The `NCCL_CHECK_CUDACC` decision

Upstream now defines in `src/include/nccl_device/utility.h`:

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

**Resolution: Accept upstream's `NCCL_CHECK_CUDACC` pattern.** It is a superset of our `#ifdef` fix and is maintained upstream, reducing future merge conflicts. Replace all our `#ifdef __CUDACC__` changes with `#if NCCL_CHECK_CUDACC` from upstream.

#### The `declval` decision

| Approach | Code |
|---|---|
| Ours | `cuda::std::declval<T>()` (from CCCL/libcu++) |
| Upstream | `nccl::utility::declval<T>()` (their own impl in `utility.h`) |

**Resolution: Accept upstream's `nccl::utility::declval`.** As of v2.29.x, upstream's implementation handles clang correctly. Using theirs minimizes diff. If it doesn't compile with clang, fall back to `cuda::std::declval` with a `@EUGO_CHANGE` annotation.

#### The `__forceinline__` decision

Upstream now conditionally uses `__attribute__((always_inline))` for clang:

```cpp
#if defined(__clang__)
  #define NCCL_DEVICE_INLINE __device__ __attribute__((always_inline)) inline
#else
  #define NCCL_DEVICE_INLINE __device__ __forceinline__
#endif
```

**Resolution: Accept upstream's definitions.** This is their official clang support.

#### File renames (mem_barrier → lsa_barrier)

Upstream renamed:
- `mem_barrier.h` → `lsa_barrier.h`
- `mem_barrier__funcs.h` → `lsa_barrier__funcs.h`
- `mem_barrier__types.h` → `lsa_barrier__types.h`
- `mem_barrier.cc` → `lsa_barrier.cc`

**Resolution:** Accept the renames. Update `NCCL_SRC_FILES` in `src/CMakeLists.txt` accordingly.

#### Mechanical resolution pattern for all HEADER files

For each header file with `#ifdef __CUDACC__` vs `#if NCCL_CHECK_CUDACC` conflicts:

1. **Accept upstream's version** (`#if NCCL_CHECK_CUDACC`)
2. **Accept all new upstream code** (new functions, structs, etc.)
3. **Remove our `#ifdef __CUDACC__` lines** (superseded by upstream's fix)
4. **Verify no remaining `#if __CUDACC__`** (bare, unfixed) lines exist

---

### 4.4 C++ Source Conflicts (SOURCE)

#### Macro naming: `CUDA_MAJOR`/`CUDA_MINOR` vs `CUDAToolkit_VERSION_MAJOR`/`CUDAToolkit_VERSION_MINOR`

**Context:** Upstream uses `CUDA_MAJOR`/`CUDA_MINOR` (set by their Makefile). Our CMake build defines `CUDAToolkit_VERSION_MAJOR`/`CUDAToolkit_VERSION_MINOR` (from `find_package(CUDAToolkit)`). See `src/CMakeLists.txt` where we set these via `target_compile_definitions`.

**Resolution:** Keep our macro names. When upstream references `CUDA_MAJOR`/`CUDA_MINOR`, replace with our equivalents. This affects:
- `src/init.cc` (version string)
- `src/ras/client_support.cc` (version reporting)

#### NDEBUG handling in `src/ras/*.cc`

**Context:** Upstream originally had `#define NDEBUG` hardcoded in several RAS files. Our fork commented these out (toolchain should set NDEBUG). As of v2.29.x, upstream also removed these.

**Resolution:** If upstream removed the `#define NDEBUG` lines entirely, accept theirs (clean deletion). Remove our `@EUGO_CHANGE` comment blocks about NDEBUG if the upstream code no longer has them.

#### General rule for SOURCE conflicts

1. Accept all new upstream functionality (new functions, refactored logic, new features)
2. Re-apply our macro renames (`CUDAToolkit_VERSION_*`)
3. Re-apply any `@EUGO_CHANGE` annotations that are still relevant
4. Remove `@EUGO_CHANGE` annotations whose upstream code has been fixed

---

### 4.5 DOCA GPUNetIO Headers & CUDA File Extensions

**Context:** Upstream added GDAKI (GPU Direct Async Kernel-Initiated) networking support via `src/transport/net_ib/gdaki/doca-gpunetio/`. This introduces `.cuh` (CUDA header) files and requires special header-copying logic. Upstream handles this via `add_subdirectory()` chain: `transport/CMakeLists.txt` → `net_ib/CMakeLists.txt` → `gdaki/CMakeLists.txt`. Since we use a flat source list, we must port the header-copying logic separately.

#### DOCA Header Copying (CRITICAL)

The include chain requires headers to be at a specific path:

```
gin_device_api.h  →  gdaki/gin_gdaki.h  →  "doca_gpunetio/doca_gpunetio_device.h"
                                             └── includes all 5 .cuh device headers
```

`gin_gdaki.h` (at `src/include/nccl_device/gin/gdaki/`) uses `#include "doca_gpunetio/doca_gpunetio_device.h"` — note the **underscore** (`doca_gpunetio`). The actual source directory uses a **hyphen** (`doca-gpunetio`).

**Header destination path decision:**

Upstream copies headers to: `${CMAKE_BINARY_DIR}/include/nccl_device/gin/gdaki/doca_gpunetio/`

This requires their `target_include_directories()` to add `${CMAKE_BINARY_DIR}/include/nccl_device/gin/gdaki/` as an include path, so that `#include "doca_gpunetio/..."` resolves to `build/include/nccl_device/gin/gdaki/doca_gpunetio/...`.

**Our architecture:** We already have `${CMAKE_BINARY_DIR}/include` in the include path (for `nccl.h` and other configured headers). For simplicity and consistency with our flat-list architecture, we copy headers to: `${CMAKE_BINARY_DIR}/include/doca_gpunetio/`

This allows `#include "doca_gpunetio/..."` to resolve without adding additional nested include paths. The quoted include directive first searches relative to the including file, then falls back to the include path list.

**Include resolution flow with our path:**
1. `gin_gdaki.h` at `src/include/nccl_device/gin/gdaki/` does `#include "doca_gpunetio/doca_gpunetio_device.h"`
2. Compiler first checks: `src/include/nccl_device/gin/gdaki/doca_gpunetio/doca_gpunetio_device.h` ❌ (doesn't exist)
3. Compiler then searches include paths:
   - `${CMAKE_BINARY_DIR}/include/doca_gpunetio/doca_gpunetio_device.h` ✅ (found!)

**Why this is simpler than upstream's approach:**
- Upstream needs to add `${CMAKE_BINARY_DIR}/include/nccl_device/gin/gdaki/` to their include directories
- We reuse `${CMAKE_BINARY_DIR}/include` which is already present for configured headers
- Fewer include paths = simpler configuration and faster compilation

**Headers copied** (from `src/transport/net_ib/gdaki/doca-gpunetio/include/`):
- **Top-level**: `doca_gpunetio_device.h` (plus `doca_gpunetio_host.h`, `doca_gpunetio_config.h`)
- **`common/`**: `doca_gpunetio_verbs_def.h`, `doca_gpunetio_verbs_dev.h`
- **`device/`**: 5 `.cuh` files (device code: `doca_gpunetio_dev_verbs_common.cuh`, `_counter.cuh`, `_cq.cuh`, `_onesided.cuh`, `_qp.cuh`)

**Implementation:** The copying logic is **inlined in `src/CMakeLists.txt`** (section "DOCA GPUNetIO Header Copying"). We do NOT invoke `src/transport/net_ib/gdaki/CMakeLists.txt` — it's kept as a reference for upstream compatibility but is not used in our build. All DOCA sources are listed in the flat `NCCL_SRC_FILES` in `src/CMakeLists.txt`.

#### Host include paths for DOCA

Host code (`gin_host_gdaki.cc`) uses `#include "doca_gpunetio_host.h"` — this requires that `${DOCA_HOME}/include` (which is `src/transport/net_ib/gdaki/doca-gpunetio/include`) is on the include path. Ensure `target_include_directories` includes this path for the `nccl` target.

#### CUDA File Extension Rules (.cu vs .cc vs .cpp)

A source file MUST have `.cu` extension (to be compiled as CUDA by clang) if ANY of these are true:
1. It contains `__device__`, `__global__`, or `__host__` function qualifiers
2. It includes `.cuh` headers (which contain device code)
3. It uses CUDA kernel launch syntax (`<<<...>>>`)
4. It uses CUDA device built-ins (`__syncthreads()`, `threadIdx`, etc.)

**Current state of doca-gpunetio `.cpp` files:** All 13 `.cpp` files in `doca-gpunetio/src/` are **pure host code**. They do NOT:
- Include any `.cuh` headers
- Contain any `__device__`/`__global__` qualifiers
- Use kernel launch syntax

`doca_verbs_cuda_wrapper.cpp` uses `dlfcn.h` to dynamically load CUDA symbols via `dlopen()` — it's a runtime wrapper, not actual CUDA code. **These files correctly remain `.cpp`.**

**Where `.cuh` files ARE compiled:**
- `doca_gpunetio_device.h` includes all 5 `.cuh` files
- `gin_gdaki.h` includes `doca_gpunetio_device.h`
- `gin_device_api.h` includes `gin_gdaki.h`
- This header chain is only included by **device code** (`.cu` files generated by `generate.py`)
- These `.cu` files are compiled by the `nccl_colldevice` target, which uses CUDA compilation

**Other `.cuh` files in the codebase:**
- `src/device/symmetric/*.cuh` (6 files) — kernel/primitives headers, included only by generated `.cu` files via `generate.py`

#### Audit checklist for future merges

When upstream adds new source files, check:
```bash
# Any new .cc/.cpp files that include .cuh headers?
grep -rn '\.cuh' src/ --include="*.cc" --include="*.cpp"

# Any new .cc/.cpp files with CUDA device code?
grep -rln '__device__\|__global__' src/ --include="*.cc" --include="*.cpp"

# New .cuh files?
find src/ -name "*.cuh" | sort
```

If any `.cc`/`.cpp` file includes `.cuh` headers or contains device code, it must be renamed to `.cu` and the `NCCL_SRC_FILES` entry updated accordingly.

---

## 5. New Upstream Features Checklist

When merging, look for these types of additions and ensure they're properly integrated:

### New source files
Add to `NCCL_SRC_FILES` in `src/CMakeLists.txt`. Run `eugo_src_diff_helper.sh` to detect.

### New subdirectories
Common pattern: upstream adds `src/gin/`, `src/rma/`, `src/os/`, `src/scheduler/` etc.
- Add a new comment section in `NCCL_SRC_FILES`
- List all `.cc` files from the new directory
- Do NOT add a separate `CMakeLists.txt` for our build (flat list architecture)

### New CUDA device source files
If upstream adds files that contain CUDA device code (`__device__`, `__global__`, `.cuh` includes):
- **If the file extension is `.cc` or `.cpp`**, rename it to `.cu` so clang compiles it as CUDA
- List the renamed `.cu` file in `NCCL_SRC_FILES`
- Note: Files that merely call CUDA runtime API (`cudaMalloc`, etc.) or use `dlfcn.h`/`dlopen` to load CUDA symbols do NOT need `.cu` — they are host code

### New header-only CUDA files (.cuh)
If upstream adds new `.cuh` files:
- Determine which compilation units include them
- If included by `.cc`/`.cpp` files → those source files must be renamed to `.cu`
- If included only by other headers that flow into the device compilation pipeline (`.cu` files generated by `generate.py`) → no renaming needed
- If located in `doca-gpunetio/include/` → ensure the DOCA header copying logic covers them

### New collective operations
Watch for additions to `all_colls` in `generate.py`. These propagate to:
- `generate.py` — function tables, kernel lists
- `symmetric/generate.py` — symmetric kernels
- C++ source files — new enqueue paths, new transport code
- Headers — new structs, enums

### New CMake options from upstream

When upstream introduces new `option(...)`, `set(... STRING CACHE ...)`, or other CMake configuration variables with `if()`/`else()` branches:

1. **Evaluate**: Decide whether the option should be ON or OFF for our clang+CMake build
2. **Reduce to selected path**: Comment out the unused branch and keep only the selected path live
3. **Annotate**: Use `# @NVIDIA_ORIGINAL:` for commented-out upstream code and `# @EUGO_CHANGE:` to explain why
4. **Set associated macros**: If the option controls a `-D` compile definition, set it manually via `NCCL_COMMON_COMPILE_DEFINITIONS`

#### Example: Reducing an option to a single path

```cmake
# @NVIDIA_ORIGINAL: option(EMIT_LLVM_IR "Generate LLVM IR" OFF)
# @NVIDIA_ORIGINAL: if(EMIT_LLVM_IR)
# @EUGO_CHANGE: We always emit LLVM IR, so the option is removed and
# the body is unconditionally included.
add_subdirectory(ir)
add_dependencies(llvm_ir nccl_header)
add_custom_target(nccl_with_ir ALL DEPENDS nccl llvm_ir)
message(STATUS "LLVM IR generation will be included in default build")
# @NVIDIA_ORIGINAL: endif()
```

#### Example: Setting associated compile definitions

```cmake
# @EUGO_CHANGE: Upstream guards this behind an option(); we always enable it.
list(APPEND NCCL_COMMON_COMPILE_DEFINITIONS EMIT_LLVM_IR=1)
```

### New compile definitions
Check upstream's root `CMakeLists.txt` and `Makefile` for new `-D` flags. Port them to our `NCCL_COMMON_COMPILE_DEFINITIONS` in `CMakeLists.txt`.

### New dependencies
Check for new `find_package()` or library links. Add to `NCCL_DEPENDENCIES` if needed.

---

## 6. AI Agent Instructions

> These instructions are for AI coding agents (Copilot, Claude, etc.) resolving merge conflicts in this repository.

### System prompt context

You are resolving merge conflicts in the `eugo-inc/nccl-cmake` fork of `NVIDIA/nccl`. This fork compiles NCCL exclusively with **clang** (no nvcc/gcc) using **CMake** (no Makefiles). The upstream NVIDIA repository uses nvcc + Makefiles historically but has been adding CMake support since v2.29.x.

### Decision framework for each conflict

```
FOR each conflict region:
  1. Identify: Is this a BUILD, CODEGEN, HEADER, or SOURCE conflict?
  2. Check: Does the eugo side contain a @EUGO_CHANGE annotation?
     - YES → This is an intentional fork change. Preserve the intent, 
             but check if upstream now handles it (making our change obsolete).
     - NO  → This is likely a parallel edit. Accept upstream's version.
  3. Apply category-specific rules from Section 4.
  4. Verify: Would the resolved code compile with clang?
```

### Specific patterns to recognize and handle

#### Pattern: `#if __CUDACC__` (upstream) vs `#ifdef __CUDACC__` (ours) vs `#if NCCL_CHECK_CUDACC` (new upstream)
→ **Always use `#if NCCL_CHECK_CUDACC`** (upstream's new approach). Our `#ifdef` fix is now obsolete.

#### Pattern: `.cu.cc` (old upstream) vs `.cu` (ours and new upstream)
→ **Use `.cu`**. Both sides converged on this. Exception: `host_table` — verify if we still need `.cu` or if `.cc` works with our clang setup.

#### Pattern: `CUDA_MAJOR` (upstream) vs `CUDAToolkit_VERSION_MAJOR` (ours)
→ **Keep ours** (`CUDAToolkit_VERSION_MAJOR`/`CUDAToolkit_VERSION_MINOR`). These come from CMake's `find_package(CUDAToolkit)`.

#### Pattern: `const` on device tables (upstream) vs no `const` (ours)
→ **Keep ours** (no `const`). Required for clang CUDA compilation. The `xla_clang.patch` documents this.

#### Pattern: `(void*)symbol` inline cast (upstream) vs separate `void* sym_ptr = (void*)sym` (ours)
→ **Keep ours** (separate pointer variable). Required for clang global initializer restrictions.

#### Pattern: `nccl::utility::declval` (upstream) vs `cuda::std::declval` (ours)
→ **Accept upstream's** `nccl::utility::declval` first. If clang compilation fails, revert to `cuda::std::declval`.

#### Pattern: Makefile-centric CMakeLists.txt (upstream) vs flat-list CMakeLists.txt (ours)
→ **Keep ours** for `src/CMakeLists.txt`. Extract new source files from upstream's version and add to `NCCL_SRC_FILES`.

#### Pattern: `NCCL_DEVICE_INLINE` / `NCCL_HOST_DEVICE_INLINE` definitions
→ **Accept upstream's** conditional definitions (they now have clang-specific paths).

#### Pattern: `.cuh` files and corresponding source files
→ **Check if any `.cc`/`.cpp` source files include `.cuh` headers.** If so, those source files must be renamed to `.cu`. Currently:
- `doca-gpunetio/src/*.cpp` files are all pure host code — they do NOT include `.cuh` and do NOT need renaming
- `.cuh` files are only included through the device header chain (`gin_gdaki.h` → `doca_gpunetio_device.h`) which flows into generated `.cu` files

#### Pattern: DOCA header copy path mismatch (`doca_gpunetio` vs `doca-gpunetio`)
→ **The `gdaki/CMakeLists.txt` must be invoked** to copy headers from `doca-gpunetio/include/` (hyphen) to `${CMAKE_BINARY_DIR}/include/nccl_device/gin/gdaki/doca_gpunetio/` (underscore). Without this, device compilation will fail to find `.cuh` headers.

### What NOT to do

- Do NOT import upstream's `add_subdirectory()` + `PARENT_SCOPE` pattern into our `src/CMakeLists.txt`
- Do NOT accept `#if __CUDACC__` (bare, unfixed) — must be either `#ifdef` or `#if NCCL_CHECK_CUDACC`
- Do NOT add `const` back to `ncclDevFuncTable[]` or `ncclDevKernelList[]` in generate.py
- Do NOT use `__forceinline__` without a clang guard
- Do NOT hardcode `#define NDEBUG` in source files
- Do NOT import `enhcompat.cc` (it overrides cudart symbols with crashing stubs)
- Do NOT rename `doca-gpunetio/src/*.cpp` files to `.cu` — they are pure host code despite being in a CUDA-related directory
- Do NOT forget the DOCA header copying step — without it, device code fails to compile

---

## 7. Quick Reference: File-by-File Resolution

| File | Resolution Strategy |
|---|---|
| `CMakeLists.txt` | Keep ours. Port new feature flags from upstream. |
| `src/CMakeLists.txt` | Keep ours. Add new source files from upstream to `NCCL_SRC_FILES`. |
| `src/device/CMakeLists.txt` | Keep ours. Check for new generate.py arguments. |
| `src/device/generate.py` | Merge carefully: accept new collectives/algos, keep clang patches. |
| `src/device/symmetric/generate.py` | Same as above. |
| `src/include/nccl_device/*.h` | Accept upstream (they have `NCCL_CHECK_CUDACC` now). |
| `src/include/nccl_device/impl/*.h` | Accept upstream. |
| `src/init.cc` | Accept upstream's new code, keep our `CUDAToolkit_VERSION_*` macros. |
| `src/ras/*.cc` | Accept upstream's NDEBUG removal, keep our `CUDAToolkit_VERSION_*` macros. |
| `ext-profiler/example/CMakeLists.txt` | Accept upstream — usually trivial. |
| `src/misc/CMakeLists.txt` | Accept upstream (reference only, not used by our build). |
| `src/plugin/CMakeLists.txt` | Accept upstream (reference only). |
| `src/nccl_device/CMakeLists.txt` | Accept upstream (reference only). |
| `src/scheduler/CMakeLists.txt` | Accept upstream (reference only). |
| `src/transport/CMakeLists.txt` | Accept upstream (reference only). |
| `src/transport/net_ib/gdaki/CMakeLists.txt` | **Keep ours.** Contains DOCA header-copying logic critical for device compilation. Must be invoked (via `add_subdirectory` or inlined). |

---

## 8. Post-Merge Maintenance

### Update `xla_clang.patch`

After merging, regenerate the patch if our clang changes evolved:

```bash
# Generate diff of our clang-specific changes vs upstream
git diff upstream/master -- src/device/generate.py src/device/common.h \
  src/device/symmetric/generate.py > xla_clang.patch.new
# Review and replace if substantially changed
```

### Comparing a "real" .so and executable against ours.
```bash
# from PyPI
# Get symbols from the PyPI wheel's libnccl
pip download nvidia-nccl-cu12 --no-deps -d ./tmp/eugo/nccl-2.29.3.wheel
cd ./tmp/eugo/nccl-2.29.3.wheel && unzip *.whl
nm -gD --defined-only nvidia/nccl/lib/libnccl.so.2 | awk '{print $3}' | sort > pypi_symbols.txt

# Our symbols
nm -gD --defined-only /tmp/eugo/__debug/nccl/lib64/libnccl.so.2.29.3 | awk '{print $3}' | sort > our_symbols.txt

diff ./pypi_symbols.txt ./our_symbols.txt

# With NVCC
cd /tmp/eugo
git clone https://github.com/NVIDIA/nccl.git nccl-upstream
cd nccl-upstream
git checkout <same-commit-you-merged>  # e.g., 25368a7
make -j$(nproc) CUDA_HOME=/usr/local/cuda
# Result: build/lib/libnccl.so and ncclras

# Exported symbols diff
nm -gD --defined-only /<path_to>/libnccl.so | awk '{print $3}' | sort > ref_symbols.txt
nm -gD --defined-only /<path_to>/libnccl.so | awk '{print $3}' | sort > our_symbols.txt
diff ref_symbols.txt our_symbols.txt

# Alternatively:
nm -DC /tmp/eugo/__debug/nccl/lib64/libnccl.so.2.29.3
nm -DC nvidia/nccl/lib/libnccl.so.2

# Size comparison
ls -lh /<path_to>/reference/libnccl.so /<path_to>/your/libnccl.so

# Symbol count
nm -gD --defined-only /<path_to>/libnccl.so | wc -l

# Check for nccl*/pnccl* public API
nm -gD --defined-only /<path_to>/libnccl.so | grep -E '^[0-9a-f]+ T (nccl|pnccl)'
```

Next, run it through an LLM to check for any unexpected missing symbols or discrepancies. Here is an example output:
```text
Exported symbols: 100% identical. The diff pypi_symbols.txt our_symbols.txt returned empty — your library exports the exact same 127 symbols (63 nccl* + 64 pnccl*) as NVIDIA's official PyPI wheel. This is the most important result.

The differences are all in the imported (undefined) symbols, which reflect different toolchains and runtime choices:

1. C++ Runtime: libstdc++ (PyPI) vs libc++ (ours)
PyPI (GCC/nvcc)	Ours (clang)
std::condition_variable::wait(...)@GLIBCXX_3.4.11	std::__1::condition_variable::wait(...)
std::thread::_M_start_thread(...)@GLIBCXX_3.4.22	std::__1::thread::join()
_Unwind_Resume@GCC_3.0	_Unwind_Resume (no version tag)
Expected. This is the fundamental toolchain difference. Both provide identical functionality.

2. CUDA Runtime: dlopen (PyPI) vs direct link (ours)
PyPI	Ours
No cuda* symbols at all	cudaMalloc@libcudart.so.12, cudaFree@libcudart.so.12, etc. (~50 symbols)
Expected. PyPI's build uses enhcompat.cc + cudawrap.cc to dlopen() libcudart at runtime (so it works without CUDA installed). Our build links directly against libcudart.so.12. Both work — ours just requires libcudart to be present at load time.

3. OpenMP: absent (PyPI) vs present (ours)
PyPI	Ours
(none)	__kmpc_fork_call@VERSION, __kmpc_dispatch_init_8@VERSION, __kmpc_dispatch_next_8@VERSION
Expected. We compile with -fopenmp, NVIDIA doesn't. The OpenMP pragmas in NCCL source are compiled as parallel code for us, serialized for them. This means our build actually utilizes OpenMP parallelism where NCCL has #pragma omp annotations.

4. GLIBC version differences (toolchain/OS age)
PyPI	Ours
__fxstat@GLIBC_2.17, __xstat@GLIBC_2.17	stat@GLIBC_2.33
dlopen@GLIBC_2.17	dlopen@GLIBC_2.34
pthread_create@GLIBC_2.17	pthread_create@GLIBC_2.34
sscanf@GLIBC_2.17	__isoc99_sscanf@GLIBC_2.17
Expected. Our container has a newer glibc. The __fxstat→stat change is the well-known glibc 2.33 stat() modernization. The __isoc99_sscanf vs sscanf difference is clang vs gcc format string handling. Note: our binary requires glibc ≥ 2.34, while PyPI's requires only ≥ 2.18. This means our .so won't load on very old distros (CentOS 7, Amazon Linux 2), but that's fine for our use case.

5. Symbols present in PyPI but absent in ours
Many POSIX symbols (sem_*, shm_*, socketpair, kill, mkdir, mkfifo, nftw, rmdir, chmod, fchmod, madvise, mprotect, sysinfo, uname, etc.) are in PyPI but not ours. This is because our ThinLTO + ICF (--icf=all) aggressively eliminates unused code paths. These functions exist in the source but are behind feature flags or code paths that LTO proved unreachable in our configuration.

Summary
Aspect	Status
Exported API (nccl*/pnccl*)	Identical
C++ ABI	Different (libc++ vs libstdc++) — expected
CUDA linking	Direct vs dlopen — both valid
OpenMP	We have it, they don't — bonus parallelism
Min glibc	Ours: 2.34, PyPI: 2.18 — acceptable
Missing POSIX calls	LTO eliminated dead code — safe
Bottom line: your library is API-compatible with the official NVIDIA release. The differences are all expected consequences of clang + libc++ + direct cudart linking vs gcc + libstdc++ + dlopen cudart.
```

### Comparing install tree against upstream

In addition to comparing exported symbols, **always compare the installed file tree** against NVIDIA's official package. This catches missing headers, misplaced files, wrong directory structure, or missing binaries that symbol comparison alone won't detect.

#### Get the reference tree from NVIDIA's official RPM

```bash
# On a Fedora/RHEL system with the CUDA repo configured:
dnf repoquery -l libnccl-devel | grep -v '^\.build-id' | sort > ref_tree.txt

# Or from the PyPI wheel:
pip download nvidia-nccl-cu12 --no-deps -d ./tmp
cd ./tmp && unzip *.whl
find nvidia/nccl/ -type f | sed 's|^nvidia/nccl/|/usr/|' | sort > ref_tree.txt
```

Example reference tree (from `dnf repoquery -l libnccl-devel` for v2.29.x):
```
/usr/bin/ncclras
/usr/include/nccl.h
/usr/include/nccl_device.h
/usr/include/nccl_device/barrier.h
/usr/include/nccl_device/comm.h
/usr/include/nccl_device/coop.h
/usr/include/nccl_device/core.h
/usr/include/nccl_device/gin.h
/usr/include/nccl_device/gin/gdaki/doca_gpunetio/common/doca_gpunetio_verbs_def.h
/usr/include/nccl_device/gin/gdaki/doca_gpunetio/common/doca_gpunetio_verbs_dev.h
/usr/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_common.cuh
/usr/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_counter.cuh
/usr/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_cq.cuh
/usr/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_onesided.cuh
/usr/include/nccl_device/gin/gdaki/doca_gpunetio/device/doca_gpunetio_dev_verbs_qp.cuh
/usr/include/nccl_device/gin/gdaki/doca_gpunetio/doca_gpunetio_device.h
/usr/include/nccl_device/gin/gdaki/gin_gdaki.h
/usr/include/nccl_device/gin/gdaki/gin_gdaki_device_host_common.h
/usr/include/nccl_device/gin/gin_device_api.h
/usr/include/nccl_device/gin/gin_device_common.h
/usr/include/nccl_device/gin/gin_device_host_common.h
/usr/include/nccl_device/gin/proxy/gin_proxy.h
/usr/include/nccl_device/gin/proxy/gin_proxy_device_host_common.h
/usr/include/nccl_device/gin_barrier.h
/usr/include/nccl_device/impl/barrier__funcs.h
/usr/include/nccl_device/impl/barrier__types.h
/usr/include/nccl_device/impl/comm__funcs.h
/usr/include/nccl_device/impl/comm__types.h
/usr/include/nccl_device/impl/core__funcs.h
/usr/include/nccl_device/impl/core__types.h
/usr/include/nccl_device/impl/gin__funcs.h
/usr/include/nccl_device/impl/gin__types.h
/usr/include/nccl_device/impl/gin_barrier__funcs.h
/usr/include/nccl_device/impl/gin_barrier__types.h
/usr/include/nccl_device/impl/ll_a2a__funcs.h
/usr/include/nccl_device/impl/ll_a2a__types.h
/usr/include/nccl_device/impl/lsa_barrier__funcs.h
/usr/include/nccl_device/impl/lsa_barrier__types.h
/usr/include/nccl_device/impl/mem_barrier__funcs.h
/usr/include/nccl_device/impl/mem_barrier__types.h
/usr/include/nccl_device/impl/ptr__funcs.h
/usr/include/nccl_device/impl/ptr__types.h
/usr/include/nccl_device/ll_a2a.h
/usr/include/nccl_device/lsa_barrier.h
/usr/include/nccl_device/mem_barrier.h
/usr/include/nccl_device/net_device.h
/usr/include/nccl_device/ptr.h
/usr/include/nccl_device/utility.h
/usr/lib64/libnccl.so
```

**Note:** The reference tree above is for v2.29.x. Update it after each upstream merge — new headers and binaries appear with new features.

#### Get our install tree

```bash
# After cmake --install or examining the build output:
# Our install prefix is typically /tmp/eugo/__debug/nccl or similar
find <our_install_prefix> -type f | sort > our_tree.txt

# Normalize paths for comparison (our prefix → /usr for diff):
sed -i 's|<our_install_prefix>|/usr|g' our_tree.txt
# Also normalize lib64 → lib or vice versa depending on reference:
sed -i 's|/usr/lib64/|/usr/lib/|g' our_tree.txt  # if needed
```

#### Compare the trees

```bash
diff ref_tree.txt our_tree.txt
```

#### What to check for

| Issue | Impact | Fix |
|---|---|---|
| Missing `ncclras` binary | Deployment missing diagnostic tool | Ensure `src/ras/CMakeLists.txt` is invoked and installs the binary |
| Missing `nccl.h` | Users can't compile against NCCL | Check `configure_file()` for `nccl.h.in` in root CMakeLists.txt |
| Missing `nccl_device/*.h` headers | Device API users can't compile | Check `install(DIRECTORY)` for `src/include/nccl_device` |
| Missing `nccl_device.h` | Device API umbrella header missing | Check install rules |
| Missing DOCA `.cuh` headers | GIN/GDAKI device API broken for users | Check DOCA header copying + install rules |
| Missing `libnccl.so` symlink | `-lnccl` linking fails | Check `install(TARGETS)` creates the symlink |
| Extra files in our tree | Not harmful but indicates build debris | Clean up install rules |
| Wrong directory nesting | Include paths break for downstream | Verify `install(DIRECTORY)` destinations match |

#### Expected differences (acceptable)

- `.build-id/` entries: RPM-specific, not present in our install — **OK**
- `lib/` vs `lib64/`: Architecture-dependent, normalize before diffing — **OK**
- `share/doc/` entries: RPM metadata, not present in our install — **OK**
- `nccl_static.a`: Only in `libnccl-static` RPM; we don't build static by default — **OK**
- `pkgconfig/nccl.pc`: May or may not be installed depending on our CMake config — verify if needed

### Testing NCCL functionality
#### With the nccl-tests suite
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make NCCL_HOME=/tmp/eugo/__debug/nccl CUDA_HOME=/usr/local/cuda
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

#### With python
```bash
# Verify all public API symbols are present
python3 -c "import ctypes; lib = ctypes.CDLL('./lib/libnccl.so.2'); ver = ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(ver)); print(f'NCCL version: {ver.value}')"
# You will see something like:
# >>> NCCL version: 22903
```

### Ncclras
```bash
bash-5.2# ncclras --version
# NCCL RAS client version 2.28.3

# or

./eugo_build/src/ras/ncclras --version
# NCCL RAS client version 2.29.3

# or

./eugo_build/src/ras/ncclras --help
# Usage: ./eugo_build/src/ras/ncclras [OPTION]...
# Query the state of a running NCCL job.

# Options:
#   -f, --format=FMT    Output format: text or json (text by default)
#   -h, --host=HOST     Host name or IP address of the RAS client socket of the
#                       NCCL job to connect to (localhost by default)
#   -m, --monitor[=GROUPS] Monitor mode: continuously watch for peer changes.
#                       Optional GROUPS: lifecycle, trace, all, or
#                       combinations like lifecycle,trace (lifecycle by default)
#   -p, --port=PORT     TCP port of the RAS client socket of the NCCL job
#                       (28028 by default)
#   -t, --timeout=SECS  Maximum time for the local NCCL process to wait for
#                       responses from other NCCL processes
#                       (5 secs by default; 0 disables the timeout)
#   -v, --verbose       Increase the verbosity level of the RAS output
#       --help          Print this help and exit
#       --version       Print the version number and exit
```

### Update `eugo_src_diff_helper.sh`

If upstream added new subdirectories (e.g., `src/gin/`, `src/os/`, `src/rma/`), add them to the directory mappings in the script.

### Update this document

After each merge, update:
- Section 4 if upstream introduced new conflict patterns
- Section 5 if new feature categories emerged
- Section 7 if new files appeared that need resolution guidance

---

## 9. Common Pitfalls

1. **Forgetting new source files**: Upstream adds files that compile fine with their `Makefile` globs but are invisible to our explicit `NCCL_SRC_FILES` list. Always run `eugo_src_diff_helper.sh`.

2. **Python indentation mismatches**: Our generate.py historically used 4-space indent, upstream uses 2-space. Mixed indentation causes Python syntax errors. Pick one and stick with it (prefer upstream's 2-space to minimize future diffs).

3. **`NCCL_USE_CMAKE` env variable**: Our root `CMakeLists.txt` sets `ENV{NCCL_USE_CMAKE}` to `"1"`. Upstream's `generate.py` checks this to skip `rules.mk` generation. If upstream changes the env var name or check logic, our device code generation breaks silently.

4. **Renamed files with content changes**: When upstream renames a file AND modifies it, git may show it as delete + add rather than rename. The conflict appears in the "new" filename. Check `git log --diff-filter=R upstream/master` for renames.

5. **Plugin files and `SKIP_UNITY_BUILD_INCLUSION`**: New plugin source files may need to be added to `SKIP_UNITY_BUILD_INCLUSION` properties to avoid ODR violations in unity builds.

6. **Upstream's subdirectory CMakeLists.txt files**: We keep these in the tree but they're NOT used by our build. Don't spend time perfecting their conflict resolution—just accept upstream's version.

7. **DOCA header copying path**: The DOCA headers live at `doca-gpunetio` (hyphen) but includes use `doca_gpunetio` (underscore). Headers must be copied to resolve this mismatch. Upstream copies to `${CMAKE_BINARY_DIR}/include/nccl_device/gin/gdaki/doca_gpunetio/` and adds that parent directory to include paths. We copy to `${CMAKE_BINARY_DIR}/include/doca_gpunetio/` since we already have `${CMAKE_BINARY_DIR}/include` in our include path. The copying logic is inlined in `src/CMakeLists.txt` section "DOCA GPUNetIO Header Copying". Device compilation will error with `fatal error: 'doca_gpunetio/doca_gpunetio_device.h' file not found` if headers aren't copied or the copy destination doesn't match the include paths.

8. **Assuming `.cpp` → `.cu` rename is needed for CUDA-adjacent files**: Not all files in CUDA directories need CUDA compilation. The `doca-gpunetio/src/*.cpp` files are host-side wrappers (IB verbs, mlx5dv, dlopen). Only rename to `.cu` if the file actually contains `__device__`/`__global__` or includes `.cuh` headers. Grep to verify before renaming.

9. **DOCA directory name mismatch**: The physical directory is `doca-gpunetio` (hyphen) but C++ includes use `doca_gpunetio` (underscore). The header copying resolves this by placing files at the underscore path. If you try to fix includes instead of copying, you'll break upstream compatibility.

10. **`__stwt(uint4*, uint4)` in clang** (RESOLVED 2026-07-06): clang 20 lacked this undocumented overload (`use of undeclared identifier '__stwt'`), so the fork temporarily carried an `@EUGO_CHANGE`-wrapped inline-PTX implementation in `src/include/nccl_device/gin/proxy/gin_proxy.h`. Clang 23 (our LLVM toolchain pin) natively ships the full `__stwt` store-intrinsic family in `__clang_cuda_intrinsics.h` (`__INTRINSIC_STORE_FAMILY(wt)`, identical `st.global.wt.v4.u32` PTX), which turned our copy into a hard `redefinition of '__stwt'` error (plus knock-on `no matching function for call to 'postGfd'` substitution failures) — the workaround was therefore removed. **Do NOT re-add it during upstream merges**: upstream's bare `__stwt(...)` calls compile as-is with clang 23+. Historical analysis: `__deleteme/STWT_INVESTIGATION.md`.

---

## 10. Version History

| Date | Upstream Version | Merge Branch | Notes |
|---|---|---|---|
| 2026-02-08 | v2.29.3-1 | `NVIDIA-upstream-master-02-26` | 29 conflicts. Upstream added CMake build (`add_subdirectory` style), `NCCL_CHECK_CUDACC` clang fix, DOCA GPUNetIO headers. Key issues: DOCA header copy path (`build/include/doca_gpunetio/` vs upstream's nested path), missing `DOCA_VERBS_USE_IBV_WRAPPER` define, `CUmemGenericAllocationHandle` nullptr→0 fix, `ptr__funcs.h` bad merge artifacts (orphaned templates), `__stwt(uint4*, uint4)` missing in clang (added inline PTX implementation). Comprehensive testing validated all clang compatibility fixes. |

