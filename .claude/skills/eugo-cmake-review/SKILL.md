---
name: eugo-cmake-review
description: Pre-commit review checklist for CMake changes in the eugo nccl-cmake fork - the @EUGO_CHANGE / @NVIDIA_ORIGINAL annotation style, the flat NCCL_SRC_FILES rule, clang-only flag hygiene (no nvcc-isms), and the invariants that must not regress. Activates on "review nccl cmake", "check nccl CMakeLists", "nccl annotation", "NVIDIA_ORIGINAL", "NCCL_SRC_FILES", "pre-commit nccl audit".
---

# eugo CMake review (nccl-cmake, pre-commit)

Backbone: the fork's non-negotiable rules in CLAUDE.md (clang-only, CMake-only,
flat source list, annotations, xla_clang.patch delta, option reduction). This
skill is the diff-time checklist.

## Annotation style (THIS repo - differs from the pytorch fork)

- Section wrappers: `# === @begin: <topic> ===` ... `# === @end: <topic> ===`.
- Our divergences: `# @EUGO_CHANGE: <why>` (reason required, WHY not WHAT).
- Preserved-but-disabled upstream code: comment it out under
  `# @NVIDIA_ORIGINAL:` (optionally `@begin:`/`@end:` for blocks) so the intent
  survives the next merge. Never silently delete upstream logic.
- There is NO annotation snapshot or sync_audit script here (unlike the pytorch
  fork) - balance of markers is checked by eye; grep both forms in your diff.

## Checklist for any CMake diff

1. New/changed logic annotated per the style above; disabled upstream branches
   kept as `@NVIDIA_ORIGINAL` comments.
2. Flat source list rule (R3): all libnccl sources live in `NCCL_SRC_FILES` in
   `src/CMakeLists.txt`, grouped by subdirectory comments. Do NOT reintroduce
   per-subdir CMakeLists + PARENT_SCOPE. The in-tree `src/*/CMakeLists.txt`
   (misc, plugin, transport, scheduler, nccl_device, ...) are upstream reference
   copies, NOT part of our build - two exceptions ARE live:
   `src/device/CMakeLists.txt` (codegen + nccl_colldevice STATIC) and
   `src/transport/net_ib/gdaki/CMakeLists.txt` (DOCA header copying).
   After adding/removing sources, run `./eugo_src_diff_helper.sh`.
3. Clang-only flag hygiene (R1): no nvcc-isms - `-Xptxas`, `-Xfatbin`,
   `--expt-*`, `-gencode`, `-ccbin`, `-maxrregcount` must not (re)appear.
   Warning control stays in `EUGO_COMMON_WARNING_FLAGS`; do not resurrect
   upstream's global `-Wall ... -g` block.
4. New upstream `option()` (R6): do not carry the option. Pick the path we ship,
   comment the other branch as `@NVIDIA_ORIGINAL`, and put any associated `-D`
   macro in `NCCL_COMMON_COMPILE_DEFINITIONS` (see the EMIT_LLVM_IR and
   BUILD_NCCL4PY precedents in root CMakeLists.txt).
5. Install surface: ONLY `nccl` and `ncclras` install (plus headers and the
   `NCCLConfig` export). Example plugins from `ext-*` must never gain an
   `install()` - a stray `libnccl-net.so` in the prefix risks accidental
   dlopen by NCCL's plugin chain. Destinations use `GNUInstallDirs` variables
   (the harness pins the canonical libdir, currently `lib64` per `eugo.std` `CANONICAL_INSTALL_LIBDIR`); never hardcode a libdir - use the GNUInstallDirs variable.

## Invariants that must not regress (grep these)

| Invariant | Where |
|---|---|
| `set(CMAKE_CUDA_FLAGS "")` BEFORE `project()`, repopulated from `$ENV{CUDAFLAGS}` after | root CMakeLists.txt (CMake CUDA-detection bug) |
| `set(ENV{NCCL_USE_CMAKE} "1")` | root CMakeLists.txt (generate.py skips rules.mk on it) |
| `EUGO_NCCL_NVML_DIRECT_HEADER=1` in `NCCL_COMMON_COMPILE_DEFINITIONS` | system nvml.h + dlopen'd libnvidia-ml (non-GPU hosts must not crash) |
| `DOCA_VERBS_USE_IBV_WRAPPER` (+ CUDA/NET wrapper defines) | src/CMakeLists.txt; upstream is missing the IBV one |
| No `NCCL_MAJOR`/`NCCL_VERSION_CODE` compile definitions | versions come from generated nccl.h only (ODR) |
| `CMAKE_CUDA_SEPARABLE_COMPILATION ON`, hidden CXX/CUDA visibility | root CMakeLists.txt |
| Version parsed from `makefiles/version.mk` | root CMakeLists.txt |
| No `CMAKE_UNITY_BUILD` enablement | known-broken for this package (symbol name collisions) |
| xla_clang.patch delta intact: no `.cu.cc` files, no `const ncclDevFuncTable`/`const ncclDevKernelList` in generate.py | R5; `find src/device -name "*.cu.cc"` and grep generate.py |

## Before commit

```bash
grep -rn '<<<<<<<\|>>>>>>>' CMakeLists.txt src/ cmake/   # no conflict markers
cmake -B eugo_build -S . -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=clang++  # configure passes
```

Update the eugo-upstream-merge-reference skill (per-file table, pitfalls) in
the same commit when the diff changes merge-relevant behavior.

## Related

- eugo-rebuild, eugo-build-and-test, eugo-upstream-merge,
  eugo-upstream-merge-reference.
