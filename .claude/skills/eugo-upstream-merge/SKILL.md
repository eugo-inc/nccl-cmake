---
name: eugo-upstream-merge
description: Merge upstream NVIDIA/nccl into this eugo fork (eugo-inc/nccl-cmake, branch master) — the CMake+clang NCCL fork — and adopt the result in protomolecule's native/cuda_nccl package. Activates on "merge upstream nccl", "nccl sync", "nccl cmake", "bump nccl", "cuda_nccl meta.json", "ncclRasCommInit".
---

# Eugo nccl-cmake fork: upstream merge + protomolecule adoption

## What this fork is

NVIDIA NCCL with two eugo additions: a **CMake build** (upstream is
Makefile-only) and **clang compatibility**. It is NOT a pytorch submodule —
it is built as the system package `native/cuda_nccl` by protomolecule, which
pins it **by commit** in `dependencies/native/cuda_nccl/meta.json`
(`version.kind = git_commit`, `branch = master`). PyTorch links the built
system libnccl via `USE_SYSTEM_NCCL=ON`.

Consequence: merging upstream here changes NOTHING in any build until
protomolecule's `meta.json` `version.commit` is bumped. That decouples
merge-time from adoption-time — merge early, adopt deliberately.

## Merge recipe

1. Branch: `NVIDIA-upstream-master-MM-YY` off `master` (convention from
   PR #9, "NVIDIA upstream master 02 26").
2. `git remote add upstream https://github.com/NVIDIA/nccl.git`
   then `git fetch upstream && git merge upstream/master`.
3. The eugo value-add lives in the CMake layer (`CMakeLists.txt` + cmake/
   helpers) which upstream does not have — conflicts there are rare; the real
   work is teaching the CMake build about NEW upstream sources/options each
   sync (compare upstream's `makefiles/` + `src/Makefile` deltas against our
   target lists).
4. PR to `master`, **merge-commit method only**.
5. Adopt: bump `protomolecule/dependencies/native/cuda_nccl/meta.json`
   `version.commit` to the merge SHA and rebuild the package.

## Known history / gotchas

- **`ncclGetErrorString` / `_Z15ncclRasCommInit...` undefined symbols**
  (documented in pytorch's `eugo_wrapper.cmake` history notes): symptoms of a
  stale system libnccl vs a newer pytorch expecting newer NCCL symbols (RAS
  subsystem). Fix = bump this fork + rebuild cuda_nccl, and ensure pytorch's
  `Dependencies.cmake` links system nccl properly.
- **`-maxrregcount`**: CUDA 13's `enable_smem_spilling` work is expected to
  make register-count pinning obsolete (tracked in protomolecule
  `dependencies/python/wave_4/torch/__cuda_llvm_updates.txt`) — when adopting
  a CUDA-13-era NCCL, revisit any maxrregcount usage in the CMake flags.
- Build sanity inside the eugo container:
  `cmake -B eugo_build -S . ${EUGO_CMAKE_COMMON_OPTIONS} -DBUILD_SHARED_LIBS=ON && ninja -C eugo_build`
  (the historical full flag line is preserved in pytorch's
  `eugo_wrapper.cmake` cleanup notes).

## History anchors

- `master` @ `db77e950` = merge of NVIDIA upstream master 2026-02 (PR #9,
  head `7603960b`); protomolecule still pins pre-merge `688abf3c` until its
  meta.json is bumped.
