---
name: eugo-build-and-test
description: Use when building or smoke-testing the eugo nccl-cmake fork (CMake+clang NCCL) - the local eugo-container build, the protomolecule native/cuda_nccl consumer flow (fetch-by-commit tarball, EUGO_CMAKE_COMMON_OPTIONS configure, ninja install), and what a healthy build looks like (libnccl.so.2 + ncclras, symbol parity vs the PyPI wheel / an nvcc reference build, install-tree parity vs the official package layout, nccl-tests). Activates on "build nccl", "compile nccl", "nccl smoke test", "test libnccl", "ncclras", "cuda_nccl package build", "nccl symbols", "install tree", "ncclGetVersion".
---

# Build and test the eugo nccl-cmake fork

NVIDIA NCCL rebuilt with CMake (upstream is Makefile-only) and compiled entirely
with clang (host + device; no nvcc, no gcc). Two build surfaces exist; keep them
distinct:

1. Local build in the eugo container - iterate on this repo.
2. protomolecule `dependencies/native/cuda_nccl` - how eugo actually ships it.

Post-merge validation ORDER (what to run when) lives in
eugo-upstream-merge-reference; the verification commands themselves are below.

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
  `eugo-inc/nccl-cmake`, branch eugo-main). Nothing you merge here ships until that
  pin is bumped (see the eugo-upstream-merge skill).
- `setup` fetches `https://github.com/eugo-inc/nccl-cmake/archive/<commit>.tar.gz`
  (repo name is literally `nccl-cmake`, NOT derived from the package name), then
  configures in `${EUGO_OUT_OF_SOURCE_BUILD_DIRECTORY_NAME}` with
  `${EUGO_CMAKE_COMMON_OPTIONS}` + `-DCMAKE_INSTALL_PREFIX=${EUGO_INSTALL_PREFIX_PATH}`.
  The harness injects Ninja generator, Release, C/CXX/CUDA standards,
  CMAKE_PREFIX_PATH, the canonical libdir (`CMAKE_INSTALL_LIBDIR=lib64` per `eugo.std` `CANONICAL_INSTALL_LIBDIR`), and PIC via that variable.
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
  export under `lib64/cmake/nccl` (no .pc file - pkg-config is disabled on purpose).
- Only `nccl` and `ncclras` install; the five ext-*/example plugins stay in the
  build tree by design.

Quick smoke (no GPU needed):

```bash
python3 -c "import ctypes; lib = ctypes.CDLL('./eugo_build/src/libnccl.so.2'); \
v = ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v)); print(v.value)"  # e.g. 22903
./eugo_build/src/ras/ncclras --version
```

## Symbol parity vs official builds

Exported `nccl*`/`pnccl*` sets must be IDENTICAL to NVIDIA's (v2.29.3: 127
symbols = 63 `nccl*` + 64 `pnccl*`).

```bash
# Reference 1: the official PyPI wheel
pip download nvidia-nccl-cu12 --no-deps -d ./tmp/nccl-wheel
cd ./tmp/nccl-wheel && unzip *.whl
nm -gD --defined-only nvidia/nccl/lib/libnccl.so.2 | awk '{print $3}' | sort > pypi_symbols.txt

# Reference 2 (optional): upstream built with nvcc at the same commit
git clone https://github.com/NVIDIA/nccl.git nccl-upstream && cd nccl-upstream
git checkout <same-commit-you-merged>
make -j$(nproc) CUDA_HOME=/usr/local/cuda    # -> build/lib/libnccl.so + ncclras

# Ours, then diff
nm -gD --defined-only <prefix>/lib64/libnccl.so.2.X.Y | awk '{print $3}' | sort > our_symbols.txt
diff pypi_symbols.txt our_symbols.txt

# Extra views
nm -DC <lib>                                       # demangled, incl. undefined
ls -lh <ref_lib> <our_lib>                         # size comparison
nm -gD --defined-only <lib> | wc -l                # symbol count
nm -gD --defined-only <lib> | grep -E '^[0-9a-f]+ T (nccl|pnccl)'  # public API
```

Run the diff through an LLM to flag unexpected discrepancies. Expected
(acceptable) differences are all on the IMPORT side:

- C++ runtime: their `std::...@GLIBCXX_*` (libstdc++) vs our `std::__1::...`
  (libc++); `_Unwind_Resume` loses its `@GCC_3.0` version tag.
- CUDA runtime: PyPI has NO `cuda*` imports (they dlopen libcudart via
  `enhcompat.cc` + `cudawrap.cc`); we link libcudart directly, so ~50
  `cuda*@libcudart.so.12` imports are correct for us.
- OpenMP: we import `__kmpc_*` (built with `-fopenmp`, NCCL's `#pragma omp`
  actually parallelizes); NVIDIA's build has none.
- glibc floor: ours needs >= 2.34 (`stat@GLIBC_2.33` modernization replaces
  `__fxstat`/`__xstat`, `__isoc99_sscanf` vs `sscanf`); PyPI's needs only
  >= 2.18. Ours will not load on CentOS 7-era distros - accepted.
- POSIX symbols present in PyPI but absent in ours (`sem_*`, `shm_*`,
  `socketpair`, `nftw`, ...): ThinLTO + `--icf=all` eliminated code paths
  proved unreachable in our configuration - safe.

## Install-tree parity vs the official package layout

Always compare the installed FILE TREE too - catches missing headers,
misplaced files, or missing binaries that symbol diffs cannot see.

```bash
# Reference tree, either from NVIDIA's RPM (needs the CUDA dnf repo):
dnf repoquery -l libnccl-devel | grep -v '^\.build-id' | sort > ref_tree.txt
# ...or from the PyPI wheel:
find nvidia/nccl/ -type f | sed 's|^nvidia/nccl/|/usr/|' | sort > ref_tree.txt

# Ours, normalized to /usr for diffing:
find <our_install_prefix> -type f | sort > our_tree.txt
sed -i 's|<our_install_prefix>|/usr|g' our_tree.txt
sed -i 's|/usr/lib64/|/usr/lib/|g' our_tree.txt   # normalize libdir if needed

diff ref_tree.txt our_tree.txt
```

Reference tree for v2.29.x (regenerate after each merge - new headers and
binaries appear with new features): `/usr/bin/ncclras`; `/usr/include/nccl.h`
and `nccl_device.h`; `/usr/lib64/libnccl.so`; and under
`/usr/include/nccl_device/`: `barrier.h`, `comm.h`, `coop.h`, `core.h`,
`gin.h`, `gin_barrier.h`, `ll_a2a.h`, `lsa_barrier.h`, `mem_barrier.h`,
`net_device.h`, `ptr.h`, `utility.h`; `impl/` with `__funcs.h` + `__types.h`
pairs for barrier, comm, core, gin, gin_barrier, ll_a2a, lsa_barrier,
mem_barrier, ptr; `gin/` with `gin_device_api.h`, `gin_device_common.h`,
`gin_device_host_common.h`, `proxy/gin_proxy.h`,
`proxy/gin_proxy_device_host_common.h`, and `gdaki/` (`gin_gdaki.h`,
`gin_gdaki_device_host_common.h`, plus `doca_gpunetio/` with
`doca_gpunetio_device.h`, `common/doca_gpunetio_verbs_{def,dev}.h`, and
`device/doca_gpunetio_dev_verbs_{common,counter,cq,onesided,qp}.cuh`).

What a tree diff can reveal:

| Issue | Fix |
|---|---|
| Missing `ncclras` | Ensure `src/ras/CMakeLists.txt` is invoked and installs it |
| Missing `nccl.h` | Check `configure_file()` for `nccl.h.in` in root CMakeLists |
| Missing `nccl_device/*.h` | Check `install(DIRECTORY)` for `src/include/nccl_device` |
| Missing `nccl_device.h` umbrella | Check install rules |
| Missing DOCA `.cuh` headers | Check DOCA header copying + install rules |
| Missing `libnccl.so` symlink | Check `install(TARGETS)` creates it (`-lnccl` fails otherwise) |
| Extra files | Build debris - clean up install rules |
| Wrong nesting | Verify `install(DIRECTORY)` destinations |

Acceptable differences: `.build-id/` entries and `share/doc/` (RPM-specific);
`lib/` vs `lib64/` (normalize first); `nccl_static.a` (only in the
libnccl-static RPM - we don't build static); `pkgconfig/nccl.pc` (we disable
pkg-config on purpose).

## Functional tests

- GPU host - NVIDIA's nccl-tests suite:

  ```bash
  git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests
  make NCCL_HOME=<install prefix> CUDA_HOME=/usr/local/cuda
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
  ```

- `ncclras --help` documents the RAS client options (`--format`, `--host`,
  `--monitor[=GROUPS]`, `--port` (default 28028), `--timeout`, `--verbose`).
- Undefined-symbol regressions in consumers (`ncclGetErrorString`,
  `ncclRasCommInit`) mean a stale system libnccl vs a newer pytorch - bump the
  cuda_nccl pin and rebuild, not pytorch.

## Related

- eugo-rebuild (cheapest correct rebuild), eugo-cmake-review (pre-commit checks),
  eugo-upstream-merge + eugo-upstream-merge-reference (sync + adoption).
