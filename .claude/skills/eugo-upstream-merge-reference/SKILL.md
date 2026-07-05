---
name: eugo-upstream-merge-reference
description: Deep-reference companion to eugo-upstream-merge for the nccl-cmake fork - an index into .github/copilot-instructions.md's per-file conflict catalog, conflict categories (BUILD/CODEGEN/HEADER/SOURCE/DOCA), post-merge validation, and the ten known pitfalls (__stwt, DOCA hyphen vs underscore, NCCL_USE_CMAKE, host_table.cu). Activates on "nccl conflict catalog", "per-file nccl resolution", "generate.py conflict", "DOCA headers", "__stwt", "xla_clang.patch", "nccl post-merge validation".
---

# eugo-upstream-merge-reference (nccl-cmake)

The merge RECIPE lives in the eugo-upstream-merge skill. The deep per-file
material lives in-repo in `.github/copilot-instructions.md` - this skill is the
index so you open the right section instead of rediscovering it.

## Map of copilot-instructions.md

| Need | Section |
|---|---|
| Fork invariants R1-R6 (clang-only, CMake-only, flat sources, annotations, xla_clang.patch, option reduction) | 1 |
| Merge workflow + branch naming | 3 (3.1-3.3) |
| Post-merge validation commands | 3.4 |
| Conflict rules by category | 4 (4.1 BUILD, 4.2 CODEGEN, 4.3 HEADER, 4.4 SOURCE, 4.5 DOCA/extensions) |
| New-upstream-feature checklist (new files, dirs, collectives, options, deps) | 5 |
| File-by-file resolution table | 7 |
| xla_clang.patch regen, symbol + install-tree comparison vs PyPI/upstream, nccl-tests, ncclras | 8 |
| Ten known pitfalls | 9 |
| Per-sync history (conflict counts, what bit us) | 10 |

## Per-file quick table (distilled from section 7)

- KEEP OURS, port upstream deltas in: root `CMakeLists.txt` (new feature flags),
  `src/CMakeLists.txt` (new sources into `NCCL_SRC_FILES`),
  `src/device/CMakeLists.txt` (new generate.py args),
  `src/transport/net_ib/gdaki/CMakeLists.txt` (DOCA header-copy logic - load-bearing).
- MERGE CAREFULLY: `src/device/generate.py` + `src/device/symmetric/generate.py`
  (accept new collectives/algos, keep clang patches: no `const` tables, `.cu`
  extensions, `host_table.cu` not `.cc`).
- ACCEPT UPSTREAM: `src/include/nccl_device/**` (they carry `NCCL_CHECK_CUDACC`
  now), `src/init.cc` and `src/ras/*.cc` (keep our `CUDAToolkit_VERSION_*`
  macros), all reference-only `src/*/CMakeLists.txt`, `ext-*` examples.

## Pitfalls that recur (section 9, condensed)

1. New upstream sources invisible to our explicit list - run
   `./eugo_src_diff_helper.sh` every merge.
2. generate.py indentation mixing (prefer upstream 2-space).
3. `NCCL_USE_CMAKE` env contract between root CMakeLists and generate.py -
   breaks codegen SILENTLY if upstream renames the check.
4. Renames shown as delete+add - check `git log --diff-filter=R upstream/master`.
5. DOCA: physical dir `doca-gpunetio` (hyphen) vs include path `doca_gpunetio`
   (underscore); we copy headers to `${CMAKE_BINARY_DIR}/include/doca_gpunetio/`.
   Missing copy = `fatal error: 'doca_gpunetio/doca_gpunetio_device.h' file not found`.
6. Not every `.cpp` near CUDA code needs `.cu` - only rename if it really has
   `__device__`/`__global__` or includes `.cuh`.
7. `__stwt(uint4*, uint4)` missing in clang 20 - guarded inline-PTX workaround in
   `src/include/nccl_device/gin/proxy/gin_proxy.h`; drop via
   `NCCL_EUGO_HAS_STWT_UINT4` once clang ships it (see
   `__deleteme/STWT_INVESTIGATION.md`).

## Post-merge gate (from 3.4 + 8)

```bash
./eugo_src_diff_helper.sh
cmake -B eugo_build -S . -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=clang++ && ninja -C eugo_build
grep -rn '<<<<<<<\|>>>>>>>' src/ CMakeLists.txt
find src/device -name "*.cu.cc"                          # must be empty
grep -n 'const ncclDevFuncTable\|const ncclDevKernelList' src/device/generate.py  # must be empty
```

Then symbol parity vs the PyPI wheel and the smoke tests - see
eugo-build-and-test. Finally: adoption is a separate step (bump
`protomolecule/dependencies/native/cuda_nccl/meta.json` commit pin) - covered by
eugo-upstream-merge.

## Related

- eugo-upstream-merge (recipe + adoption), eugo-cmake-review (annotation and
  invariant checks on the resolved tree), eugo-build-and-test.
