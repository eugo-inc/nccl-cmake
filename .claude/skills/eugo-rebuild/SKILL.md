---
name: eugo-rebuild
description: Decide what level of rebuild a change to the eugo nccl-cmake fork actually requires - docs-only vs configure check vs full ninja build - and when the protomolecule native/cuda_nccl package pin must be bumped. Activates on "rebuild nccl", "do I need to rebuild nccl", "nccl configure check", "bump cuda_nccl pin", "when does nccl ship".
---

# eugo-rebuild (nccl-cmake)

Pick the cheapest step that still proves the change. NCCL is a single shared
library plus one executable; full builds are minutes, not hours - but the
configure check is seconds, so still run it first for build-file changes.

## Decision table (first matching row wins)

| Files changed | Action |
|---|---|
| Docs, comments, `.github/`, `.claude/`, `__deleteme/`, examples text | Nothing. |
| Root `CMakeLists.txt`, `src/**/CMakeLists.txt`, `cmake/*` | Configure check, then full build. |
| `src/device/generate.py`, `src/device/symmetric/generate.py` | Full build (codegen reruns; also check `xla_clang.patch` still reflects the clang delta). |
| `src/**/*.cc`, `*.cu`, `*.h`, `*.cuh` | Full build (incremental `ninja -C eugo_build` is fine). |
| `makefiles/version.mk` | Full build; version is parsed at configure time and lands in `nccl.h` + the .so name. |
| `ext-net/`, `ext-tuner/`, `ext-profiler/`, `ext-mixed/`, `ext-env/` | Full build only if you care about the example plugins; they are built but never installed. |

## Configure check (cheap gate)

```bash
cmake -B eugo_build -S . ${EUGO_CMAKE_COMMON_OPTIONS} \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=clang++
```

Catches missing `find_package` deps (CUDAToolkit, cccl), broken generate.py
invocation, and syntax errors before any compilation. `CUDAARCHS` must be set.

## Full build + smoke

```bash
ninja -C eugo_build
```

Then the ctypes `ncclGetVersion` + `ncclras --version` smoke from
eugo-build-and-test. After adding/removing source files, also run
`./eugo_src_diff_helper.sh` to confirm the flat `NCCL_SRC_FILES` list matches
upstream's tree.

## When does anything actually ship?

Never on push. protomolecule's `dependencies/native/cuda_nccl/meta.json` pins a
COMMIT of this repo; a change only reaches the platform when that pin is bumped
and the cuda_nccl package rebuilds. Consumers (pytorch links the system libnccl
via `USE_SYSTEM_NCCL=ON`) then need a rebuild only if they want new symbols -
undefined `ncclRasCommInit`/`ncclGetErrorString` at pytorch link time is the
classic stale-pin symptom.

## Related

- eugo-build-and-test (commands, healthy-state), eugo-upstream-merge (pin-bump
  adoption flow).
