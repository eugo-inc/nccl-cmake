---
name: eugo-build-and-test
description: Use when building or smoke-testing the eugo nccl-cmake fork (CMake+clang NCCL) - the local eugo-container build, the protomolecule native/cuda_nccl consumer flow (fetch-by-commit tarball, EUGO_CMAKE_COMMON_OPTIONS configure, ninja install), and what a healthy build looks like (libnccl.so.2 + ncclras + symbol parity vs the PyPI wheel). Activates on "build nccl", "compile nccl", "nccl smoke test", "test libnccl", "ncclras", "cuda_nccl package build", "nccl symbols", "ncclGetVersion".
---

# Build and test the eugo nccl-cmake fork

NVIDIA NCCL rebuilt with CMake (upstream is Makefile-only) and compiled entirely
with clang (host + device; no nvcc, no gcc). Two build surfaces exist; keep them
distinct:

1. Local build in the eugo container - iterate on this repo.
2. protomolecule `dependencies/native/cuda_nccl` - how eugo actually ships it.

The deep in-repo playbook is `.github/copilot-instructions.md` (merge-focused, but
sections 3.4 and 8 own build validation and testing) - defer to it for detail.

## Local build (eugo container)

```bash
cmake -B eugo_build -S . ${EUGO_CMAKE_COMMON_OPTIONS} \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=clang++
ninja -C eugo_build
```

- `CUDAARCHS` (env) or `-DCMAKE_CUDA_ARCHITECTURES=...` is REQUIRED - the fork
  deliberately removed upstream's arch auto-detection.
- A working python3 is a buildtime dep: `src/device/generate.py` and
  `src/device/symmetric/generate.py` generate device sources at build time.
  The root CMakeLists sets `ENV{NCCL_USE_CMAKE}=1` so generate.py skips rules.mk.
- Version comes from `makefiles/version.mk` (parsed at configure time) - never
  hardcode it.

## How protomolecule builds it (`dependencies/native/cuda_nccl`)

- `meta.json` pins this fork BY COMMIT (`version.kind = git_commit`, repo
  `eugo-inc/nccl-cmake`, branch master). Nothing you merge here ships until that
  pin is bumped (see the eugo-upstream-merge skill).
- `setup` fetches `https://github.com/eugo-inc/nccl-cmake/archive/<commit>.tar.gz`
  (repo name is literally `nccl-cmake`, NOT derived from the package name), then
  configures in `${EUGO_OUT_OF_SOURCE_BUILD_DIRECTORY_NAME}` with
  `${EUGO_CMAKE_COMMON_OPTIONS}` + `-DCMAKE_INSTALL_PREFIX=${EUGO_INSTALL_PREFIX_PATH}`.
  The harness injects Ninja generator, Release, C/CXX/CUDA standards,
  CMAKE_PREFIX_PATH, `CMAKE_INSTALL_LIBDIR=lib`, and PIC via that variable.
- Sibling cmake packages end setup with `ninja -vvv && ninja install -vvv`; as of
  2026-07 cuda_nccl's setup stops after configure - if a harness build installs
  nothing, add those two lines protomolecule-side (that file, not this repo).
- Dependency posture (meta.json is ground truth): libcudart linked directly;
  libnvidia-ml loaded via dlopen with the system `nvml.h` header
  (`EUGO_NCCL_NVML_DIRECT_HEADER=1`) so non-GPU hosts do not crash; cuda_cccl
  headers; nvtx stays VENDORED (upstream's modified copy - do not switch to
  system cuda_nvtx yet); rdma_core/openmpi deliberately not linked.

## What healthy looks like

- Artifacts: `eugo_build/src/libnccl.so.2.<minor>.<patch>` (+ `.so.2`, `.so`
  symlinks), `eugo_build/src/ras/ncclras`. Install adds `nccl.h`,
  `nccl_device.h`, the `nccl_device/` header dir, and the `NCCLConfig` CMake
  export under `lib/cmake/nccl` (no .pc file - pkg-config is disabled on purpose).
- Only `nccl` and `ncclras` install; the five ext-*/example plugins stay in the
  build tree by design.

Quick smoke (no GPU needed):

```bash
python3 -c "import ctypes; lib = ctypes.CDLL('./eugo_build/src/libnccl.so.2'); \
v = ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v)); print(v.value)"  # e.g. 22903
./eugo_build/src/ras/ncclras --version
```

Deeper verification (copilot-instructions.md section 8):

- Symbol parity vs the official PyPI wheel (`nvidia-nccl-cu12`): exported
  `nccl*`/`pnccl*` sets must be IDENTICAL. Expected diffs are all in imports:
  libc++ vs libstdc++, direct libcudart vs dlopen, OpenMP present, newer glibc
  floor, LTO-eliminated dead POSIX calls.
- Functional (GPU host): NVIDIA's `nccl-tests` built with
  `NCCL_HOME=<install prefix>`, e.g. `./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1`.
- Undefined-symbol regressions in consumers (`ncclGetErrorString`,
  `ncclRasCommInit`) mean a stale system libnccl vs a newer pytorch - bump the
  cuda_nccl pin and rebuild, not pytorch.

## Related

- eugo-rebuild (cheapest correct rebuild), eugo-cmake-review (pre-commit checks),
  eugo-upstream-merge + eugo-upstream-merge-reference (sync + adoption).
