---
name: eugo-upstream-merge
description: Merge upstream NVIDIA/nccl into this eugo fork (eugo-inc/nccl-cmake, branch eugo-main) — the CMake+clang NCCL fork — and adopt the result in protomolecule's native/cuda_nccl package. Covers branch prep, conflict triage/order, the per-region decision framework, and adoption. Activates on "merge upstream nccl", "nccl sync", "nccl cmake", "bump nccl", "cuda_nccl meta.json", "ncclRasCommInit".
---

# Eugo nccl-cmake fork: upstream merge + protomolecule adoption

## What this fork is

NVIDIA NCCL with two eugo additions: a **CMake build** (upstream is
Makefile-only) and **clang compatibility**. It is NOT a pytorch submodule —
it is built as the system package `native/cuda_nccl` by protomolecule, which
pins it **by commit** in `dependencies/native/cuda_nccl/meta.json`
(`version.kind = git_commit`, branch = the fork default `eugo-main`). PyTorch
links the built system libnccl via `USE_SYSTEM_NCCL=ON`.

Remotes and branches:

- `origin` = https://github.com/eugo-inc/nccl-cmake.git (our fork; default
  branch `eugo-main` — our stable clang+CMake branch)
- `upstream` = https://github.com/NVIDIA/nccl.git (their `master` is the
  merge source; upstream-branch names stay `master`)
- Historical branches: `NVIDIA-master-cmake` (CMake integration work),
  `NVIDIA-master` (pristine mirror of upstream master)

Consequence: merging upstream here changes NOTHING in any build until
protomolecule's `meta.json` `version.commit` is bumped. That decouples
merge-time from adoption-time — merge early, adopt deliberately.

## Merge recipe

1. Branch: `NVIDIA-upstream-master-MM-YY` off `eugo-main` (convention from
   PR #9, "NVIDIA upstream master 02 26").
2. `git remote add upstream https://github.com/NVIDIA/nccl.git`
   then `git fetch upstream && git merge upstream/master`.
3. The eugo value-add lives in the CMake layer (`CMakeLists.txt` + cmake/
   helpers) which upstream does not have — conflicts there are rare; the real
   work is teaching the CMake build about NEW upstream sources/options each
   sync (compare upstream's `makefiles/` + `src/Makefile` deltas against our
   target lists).
4. PR to `eugo-main`, **merge-commit method only**.
5. Adopt: bump `protomolecule/dependencies/native/cuda_nccl/meta.json`
   `version.commit` to the merge SHA and rebuild the package.

## Conflict triage

Categorize conflicts before touching any file:

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

Resolve in dependency order (dependencies flow downward):

1. **HEADER** — nccl_device headers (most mechanical)
2. **CODEGEN** — generate.py files (requires understanding both codegen
   pipelines)
3. **BUILD** — CMakeLists.txt files (depends on knowing the new source files
   from steps 1-2)
4. **SOURCE** — C++ source files (mostly local macro name changes)
5. **OTHER** — documentation, config, etc.

## Decision framework per conflict region

1. Identify the category (BUILD / CODEGEN / HEADER / SOURCE).
2. Does the eugo side contain an `@EUGO_CHANGE` annotation?
   - YES -> intentional fork change. Preserve the intent, but check whether
     upstream now handles it (making our change obsolete).
   - NO -> likely a parallel edit. Accept upstream's version.
3. Apply the category-specific rules in eugo-upstream-merge-reference.
4. Verify: would the resolved code compile with clang?

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

- `eugo-main` (then named `master`) @ `db77e950` = merge of NVIDIA upstream
  master 2026-02 (PR #9, head `7603960b`); protomolecule still pins pre-merge
  `688abf3c` until its meta.json is bumped. Per-sync conflict details live in
  eugo-upstream-merge-reference (Sync history).

## Related

- eugo-upstream-merge-reference (per-file conflict catalog, category rules,
  post-merge gate, pitfalls), eugo-build-and-test (build + symbol/install-tree
  parity), eugo-cmake-review (annotation + invariant checks on the resolved
  tree).
