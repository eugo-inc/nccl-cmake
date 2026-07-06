# CLAUDE.md

Eugo fork of NVIDIA/nccl (`eugo-inc/nccl-cmake`, branch `eugo-main`; upstream
branch names stay `master`). The fork exists to build NCCL with clang ONLY
(host + device - no nvcc, no gcc) and CMake ONLY (upstream `Makefile` logic
is informational - NEVER import it verbatim). Ships solely as protomolecule's
`native/cuda_nccl` system package, pinned BY COMMIT in its `meta.json`;
pytorch links it via `USE_SYSTEM_NCCL=ON` - nothing merged here ships until
that pin is bumped. Playbooks live in `.claude/skills/` - route by event:

- Building or smoke-testing -> `eugo-build-and-test`: local container build,
  the cuda_nccl consumer flow, symbol parity vs the PyPI wheel, install-tree
  comparison, nccl-tests.
- Unsure how much validation a diff needs -> `eugo-rebuild` (files -> action).
- About to merge upstream -> `eugo-upstream-merge` (branch prep, conflict
  triage, resolution order, decision framework, adoption). NEVER improvise a
  conflict resolution -> `eugo-upstream-merge-reference` names the rule for
  every file and category (BUILD/CODEGEN/HEADER/SOURCE/DOCA).
- Merge conflicts all resolved -> run the post-merge gate in
  `eugo-upstream-merge-reference` (source-diff helper, build, orphaned
  conflict-marker grep, xla_clang.patch intactness) plus the symbol and
  install-tree parity checks in `eugo-build-and-test`; report each pass/fail.
- Touched a CMakeLists -> `eugo-cmake-review` before committing.

Fork invariants (every change, not just merges):

- All libnccl sources live in the flat `NCCL_SRC_FILES` list in
  `src/CMakeLists.txt` - NEVER reintroduce per-subdirectory CMakeLists +
  `PARENT_SCOPE` for the main library.
- NEVER leave a divergence unannotated -> `# @EUGO_CHANGE: <why>`; NEVER
  silently delete upstream code -> comment it out under
  `# @NVIDIA_ORIGINAL:` so the intent survives the next merge.
- NEVER carry an upstream `option()` both ways -> pick the path we ship,
  comment the unused branch as `@NVIDIA_ORIGINAL`, and set any associated
  `-D` macro via `NCCL_COMMON_COMPILE_DEFINITIONS`.
- The `xla_clang.patch` clang delta must survive every change; observable:
  `find src/device -name "*.cu.cc"` and
  `grep -n 'const ncclDevFuncTable\|const ncclDevKernelList' src/device/generate.py`
  must both return nothing.

## Working with the user

NEVER delete or rewrite the user's "Prompt" lines (lines they wrote as
prompts/questions) -> reproduce them verbatim and put your response alongside.
