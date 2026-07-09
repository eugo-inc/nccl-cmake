cmake_minimum_required(VERSION 4.0)


# === @begin: Reading version dynamically (from the file) ===
# @NVIDIA_ORIGINAL: @begin:
# @EUGO_CHANGE: We preserved this as this allows reading version dynamically and not update that manually on every upstream sync.

# Version information
# Read makefiles/version.mk file
file(READ ${CMAKE_SOURCE_DIR}/makefiles/version.mk VERSION_CONTENT)
string(REGEX REPLACE ".*NCCL_MAJOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MAJOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_MINOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MINOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_PATCH[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_PATCH "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_SUFFIX[ ]*:=[ ]*([a-zA-Z0-9]*).*" "\\1" NCCL_SUFFIX "${VERSION_CONTENT}")
string(REGEX REPLACE ".*PKG_REVISION[ ]*:=[ ]*([0-9]+).*" "\\1" PKG_REVISION "${VERSION_CONTENT}")
math(EXPR NCCL_VERSION_CODE "(${NCCL_MAJOR} * 10000) + (${NCCL_MINOR} * 100) + ${NCCL_PATCH}")

# @EUGO_CHANGE:
# 1. `NCCL_USE_CMAKE` - is only used in `@/src/device/generate.py` and `@/src/device/symmetric/generate.py`, so we shouldn't pass it as a compile definition but as an environment variable as done a few lines below!
#    1. ~~@EUGO_CHANGE: `NCCL_USE_CMAKE` -> `NCCL_USE_CMAKE=1` - this makes check work better as w/o value `-D` makes value to be defined as an empty string (yep, in C pre-processor it's distinct from undefined).~~
# 2. `NCCL_MAJOR`, `NCCL_MINOR`, `NCCL_PATCH`, and `NCCL_VERSION_CODE` are commented out here as they're set via `nccl.h` generated header and we shouldn't provide their definition twice to avoid build- and runtime ODR violation.
#
# Make version information available to C++ source files
# add_compile_definitions(
    # NCCL_USE_CMAKE=1

    # NCCL_MAJOR=${NCCL_MAJOR}
    # NCCL_MINOR=${NCCL_MINOR}
    # NCCL_PATCH=${NCCL_PATCH}
    # NCCL_VERSION_CODE=${NCCL_VERSION_CODE}
# )

# @EUGO_CHANGE:
# This is used in `@/src/device/generate.py` and ``@/src/device/symmetric/generate.py` so we preserve it.
set(ENV{NCCL_USE_CMAKE} "1")
# @NVIDIA_ORIGINAL: @end


# This is our change as their reference implementation sets this guy separately every time it's needed making it more fragile if somebody will set it differently in different use places.
set(NCCL_VERSION "${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}")
# === @end: Reading version dynamically (from the file) ===


# === @begin: General Configuration ===
# Unset CMAKE_CUDA_FLAGS to fix CMake bugs in detecting the CUDA toolchain
set(CMAKE_CUDA_FLAGS "") # @EUGO_CHANGE: Added to fix CMake bugs in detecting the CUDA toolchain

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#set(CMAKE_C_VISIBILITY_PRESET default) # Intentionally commented out
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

# `C` language seems to not be used but we set that for the match w/ the original version (where it's used for `@/src/enhcompat.cc` which has a few functions w/ C linkage).
project(nccl
    LANGUAGES CUDA CXX C
    VERSION ${NCCL_VERSION}
)

set(CMAKE_CUDA_FLAGS
  " $ENV{CUDAFLAGS}"
)

# @EUGO_CHANGE: we only support Linux so need to detect OS
# Detect OS
set(NCCL_OS_LINUX ON)
add_definitions(-DNCCL_OS_LINUX)

# @EUGO_CHANGE:
# Disabled as we prefer to control warnings by using our global / shared flags.
# set(CMAKE_CXX_FLAGS
#     "${CMAKE_CXX_FLAGS} \
#     -Wall -Wvla \
#     -Wno-unused-function \
#     -Wno-sign-compare"
# )
# === @end: General Configuration ===


# === @begin: Warnings Configuration ===
# @NVIDIA_ORIGINAL: @begin: (upstream compiler flags - nvcc-specific, not compatible with clang)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -Wall -Wno-unused-function -Wno-sign-compare -Wvla -g")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all -fPIC")
# @NVIDIA_ORIGINAL: @end
set(
    EUGO_COMMON_WARNING_FLAGS
    "-Wno-deprecated-this-capture \
    -Wno-null-conversion \
    -Wno-deprecated-enum-enum-conversion \
    -Wno-deprecated-volatile"
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EUGO_COMMON_WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EUGO_COMMON_WARNING_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${EUGO_COMMON_WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")
# === @end: Warnings Configuration ===


# === @begin: Build Options ===

# @Important:
# In original code, most of these flags are CMake options. We reduce them to -D for ones which we use and omit others to reduce the complexity of our build as we support only 1 platform and only 1 compiler, making our life easier. Much easier.

# @NVIDIA_ORIGINAL: @begin: (upstream uses option() calls)
# option(BUILD_DEBIAN_PACKAGE "Build Debian package" OFF)
# option(BUILD_REDHAT_PACKAGE "Build Redhat package" OFF)
# option(BUILD_TXZ_PACKAGE "Build TXZ package" OFF)
# option(BUILD_CONDA_PACKAGE "Build Conda package" OFF)
# option(BUILD_SRCTXZ_PACKAGE "Build source tarball package" OFF)
# option(BUILD_DOC_PACKAGE "Build documentation package" OFF)
# option(BUILD_PACKAGES "Build all packages" OFF)
# option(VERBOSE "Enable verbose output" OFF)
# option(KEEP "Keep intermediate files" OFF)
# option(DEBUG "Enable debug build" OFF)
# option(ASAN "Enable Address Sanitizer" OFF)
# option(UBSAN "Enable Undefined Behavior Sanitizer" OFF)
# option(TRACE "Enable tracing" OFF)
# option(WERROR "Treat warnings as errors" OFF)
# option(PROFAPI "Enable profiling API" ON)
# option(NVTX "Enable NVTX" ON)
# option(RDMA_CORE "Enable RDMA core" OFF)
# option(NET_PROFILER "Enable network profiler" OFF)
# option(MLX5DV "Enable MLX5DV" OFF)
# option(MAX_EXT_NET_PLUGINS "Maximum external network plugins" 0)
# option(EMIT_LLVM_IR "Generate LLVM IR for device APIs" OFF)
# @NVIDIA_ORIGINAL: @end
#
# @EUGO_CHANGE: We use a flat compile definitions list for the subset of options
# that map to C/C++ macros. Options without corresponding macros (packaging, sanitizers,
# verbosity) are in the upstream option() block above for reference only.
set(NCCL_COMMON_COMPILE_DEFINITIONS
    # BUILD_DEBIAN_PACKAGE=OFF # Build Debian package
    # BUILD_REDHAT_PACKAGE=OFF # Build Redhat package
    # BUILD_TXZ_PACKAGE=OFF # Build TXZ package
    # BUILD_CONDA_PACKAGE=OFF # Build Conda package
    # BUILD_SRCTXZ_PACKAGE=OFF # Build source tarball package
    # BUILD_DOC_PACKAGE=OFF # Build documentation package
    # BUILD_PACKAGES=OFF # Build all packages
    # VERBOSE=1 # "Enable verbose output" [there is no associated C/C++ macro], disabled as this only applies to Make/CMake verbosity in original implementation
    # KEEP=1 # "Keep intermediate files" [there is no associated C/C++ macro], disabled as this only applies to original implementation and its Make/CMake execution
    # ASAN=1 # "Enable Address Sanitizer" [there is no associated C/C++ macro], not yet supported in Eugo environment
    # UBSAN=1 # "Enable Undefined Behavior Sanitizer" [there is no associated C/C++ macro], not yet supported in Eugo environment

    # WERROR=1 # "Treat warnings as errors" [there is no associated C/C++ macro]
    # We're not reimplementing this as it repeats the built-in CMake flags:
    # 1. `CMAKE_COMPILE_WARNING_AS_ERROR`
    #    1. https://cmake.org/cmake/help/latest/prop_tgt/COMPILE_WARNING_AS_ERROR.html
    #    2. https://cmake.org/cmake/help/latest/variable/CMAKE_COMPILE_WARNING_AS_ERROR.html
    # 2. `CMAKE_LINK_WARNING_AS_ERROR`
    #    1. https://cmake.org/cmake/help/latest/prop_tgt/LINK_WARNING_AS_ERROR.html
    #    2. https://cmake.org/cmake/help/latest/variable/CMAKE_LINK_WARNING_AS_ERROR.html

    PROFAPI=1 # "Enable profiling API", @Important: this is not about "NCCL's API for profiling" but about "profiling NCCL's own API".
    # If enabled, this causes every public NCCL function to be emitted as a weak symbol instead of a strong one w/ an extra strong alias symbol prefixed w/ `p`.
    # E.g., instead of having just `strong` `ncclCommInitAll`, w/ `PROFAPI` enabled we'll have both `weak` `ncclCommInitAll` symbol and `strong` alias-symbol `pncclCommInitAll` pointing to `ncclCommncclCommInitAllInit`.
    # This, in theory, should allow redirecting calls to the `NCCL_API`-marked functions to the ad-hoc stubs for profiling purposes w/o losing the access to original symbol.
    # Yet, I'm not aware of the public code relying on that + I'm not sure if this is correct - usually, you define `alias` symbol to be `weak` and not its target, but who knows what they've meant here ... It's not the first confusing design choice we've seen so far.
    # This setting is enabled in reference `NCCL` builds so we keep it too.
    #
    # References:
    # 1. @/src/include/core.h#L17-32
    # 2. https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-alias-function-attribute

    # NVTX=1 # "Enable NVTX" [there is no associated C/C++ macro]. In reference implementation. this is only used in negated logic - if this option is disabled, it sets `NVTX_DISABLE` macro, which causes vendored `native/cuda_nvtx` headers to contain no compilable symbols (some NVTX macros are used by NCCL regardless of this setting).

    # NCCL_ENABLE_NET_PROFILING=1 # "Enable network profiler"  [option: `NET_PROFILER`], disabled by default + causes building an extra plugin which isn't easy to build and maintain + all 3 shipped plugins are considered tests/examples of NCCL usage and not something NCCL really needs to operate properly.
    # NCCL_NET_MAX_PLUGINS=0 # "Maximum external network plugins" [option: `MAX_EXT_NET_PLUGINS=<int>`], this is set to `0` by default, meaning no extra plugins can be loaded, which makes sense as we're not building any of them.

    # EMIT_LLVM_IR=1 # Generate LLVM IR for device APIs" [option: `EMIT_LLVM_IR`], this is used in `@/bindings/ir/` subdirectory so it can be used by code generation packages to enable calling NCCL APIs w/o the roundtrip to the host. However, as of now we don't have such packages so we don't enable it yet.

    EUGO_NCCL_NVML_DIRECT_HEADER=1 # Eugo-specific: allows using `nvml.h` from the system but still load `libnvidia-ml` dynamically via `dlopen`. This is safe in Eugo Environment as we always have one and only one version of this library in a given version and it's better to have all APIs set for efficiency rather than not.
)

# === @begin:Options conditionals ===
# @NVIDIA_ORIGINAL: @begin: (upstream package building subdirectories)
# @EUGO_CHANGE: Not needed in Eugo environment, but preserved for reference.
# NOTE: all of these are commented out because we don't build them
# if(MAX_EXT_NET_PLUGINS GREATER 0)
#     add_definitions(-DNCCL_NET_MAX_PLUGINS=${MAX_EXT_NET_PLUGINS})
# endif()
# Add package building subdirectories (commented out as not needed in Eugo environment, but preserved for the reference implementation)
# if(BUILD_DEBIAN_PACKAGE OR BUILD_PACKAGES)
    # add_subdirectory(pkg/debian)
# endif()

# if(BUILD_REDHAT_PACKAGE OR BUILD_PACKAGES)
    # add_subdirectory(pkg/redhat)
# endif()

# if(BUILD_TXZ_PACKAGE OR BUILD_PACKAGES)
    # add_subdirectory(pkg/txz)
# endif()

# if(BUILD_CONDA_PACKAGE OR BUILD_PACKAGES)
    # add_subdirectory(pkg/conda)
# endif()

# if(BUILD_SRCTXZ_PACKAGE OR BUILD_PACKAGES)
    # add_subdirectory(pkg/srctxz)
# endif()


# === @@begin: EMIT_LLVM_IR ===
# if(EMIT_LLVM_IR)
#     add_subdirectory(ir)
#     add_dependencies(llvm_ir nccl_header)
#     add_custom_target(nccl_with_ir ALL DEPENDS nccl llvm_ir)
#     message(STATUS "LLVM IR generation will be included in default build")
# endif()

# if(EMIT_LLVM_IR)
#     add_definitions(-DEMIT_LLVM_IR=1)
# endif()
# === @@end: EMIT_LLVM_IR ===
# === @end: Option Conditionals ===
# @NVIDIA_ORIGINAL: @end


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND NCCL_COMMON_COMPILE_DEFINITIONS
        # DEBUG=1 # "Enable debug build" [there is no associated C/C++ macro], this implies the following individual macros:
        DEBUG_EVENTS=1
        DEBUG_PROXY=1
        NVTX_DEBUG_PRINT=1 # This macro isn't used in `native/cuda_nccl`, but in its vendored `native/cuda_nvtx` headers.

        # TRACE=1 # "Enable tracing" [there is no associated C/C++ macro], this implies the following individual macros:
        ENABLE_TRACE=1
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0")
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -G -g")
    )
endif()
# === @end: Build Options ===


# === @begin: dependencies/ ===
# === @@begin: CUDAToolkit ===
find_package(CUDAToolkit REQUIRED)
message(STATUS "[EUGO]: Find CUDAToolkit: ${CUDAToolkit_CONFIG}")

# @NVIDIA_ORIGINAL: @begin: (upstream CUDA version detection)
# @EUGO_CHANGE: Upstream extracts CUDA_MAJOR/CUDA_MINOR from CUDAToolkit_VERSION string
# and uses them as global compile definitions. We instead use CMake's
# CUDAToolkit_VERSION_MAJOR/CUDAToolkit_VERSION_MINOR (set automatically by find_package)
# as PRIVATE target_compile_definitions in src/CMakeLists.txt.
#
# string(REGEX MATCH "([0-9]+\\.[0-9]+)" CUDA_VERSION "${CUDAToolkit_VERSION}")
# string(REGEX MATCH "([0-9]+)" CUDA_MAJOR "${CUDA_VERSION}")
# string(REGEX REPLACE ".*\\.([0-9]+)$" "\\1" CUDA_MINOR "${CUDA_VERSION}")
#
# add_compile_definitions(
#     CUDA_MAJOR=${CUDA_MAJOR}
#     CUDA_MINOR=${CUDA_MINOR}
# )
# @NVIDIA_ORIGINAL: @end

# @NVIDIA_ORIGINAL: @begin: (upstream CUDA architecture auto-detection)
# @EUGO_CHANGE: We set CMAKE_CUDA_ARCHITECTURES externally via toolchain file or
# command-line (-DCMAKE_CUDA_ARCHITECTURES=...) so this auto-detection is not needed.
#
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
#     message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting defaults based on CUDA version")
#     if(${CUDA_MAJOR} LESS 9)
#         set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61")
#     elseif(${CUDA_MAJOR} EQUAL 9)
#         set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70")
#     elseif(${CUDA_MAJOR} EQUAL 10)
#         set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70")
#     elseif(${CUDA_MAJOR} EQUAL 11)
#         if(${CUDA_MINOR} LESS 8)
#             set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70;80")
#         else()
#             set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70;80;90")
#         endif()
#     elseif(${CUDA_MAJOR} EQUAL 12)
#         if(${CUDA_MINOR} LESS 8)
#             set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;80;90")
#         else()
#             set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;80;90;100;120")
#         endif()
#     elseif(${CUDA_MAJOR} EQUAL 13)
#         set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;80;90;100;110;120")
#     else()
#         set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;80;90;100;110;120")
#     endif()
# endif()
# message(STATUS "Using CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
# @NVIDIA_ORIGINAL: @end
# === @@end: CUDAToolkit ===


# === @@begin: CCCL ===
# This is also part of `CUDAToolkit` (`native/cuda`) but it's not found by `FindCUDAToolkit` as it's open-source and header-only library + this may even be installed separately.
# Hence, we search it separately and use only the portion we need.
find_package(cccl CONFIG REQUIRED)
message(STATUS "[EUGO]: CCCL found: ${cccl_CONFIG}")
# === @@end: CCCL ===


# === @@begin: glibc-merged libs ===
# In reference implementation, all `native/cuda_nccl` are linked against `librt` (realtime, not runtime!), `libpthread`, and `libdl`.
# However, in modern `glibc` versions they're all merged into main `libc.so` file and dedicated binaries are simply compatability stubs.
# Hence, we don't have to lookup and link them explicitely.
# === @@end: glibc-merged libs ===


# === @@begin: Discarded dependencies ===
# option(RDMA_CORE "Enable RDMA core" OFF) # Sets `NCCL_BUILD_RDMA_CORE=1` macro and links `libibverbs` (`native/rdma_core`)
# option(MLX5DV "Enable MLX5DV" OFF) # Sets `NCCL_BUILD_MLX5DV=1` macro and links `libmlx5` (`native/rdma_core`)
# === @@end: Discarded dependencies ===
# === @@begin: Sanitizer Options ===
# @NVIDIA_ORIGINAL: @begin: (upstream sanitizer support)
# @EUGO_CHANGE: Not yet supported in Eugo environment.
#
# if(ASAN)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -static-libasan")
# endif()
#
# if(UBSAN)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined -static-libubsan")
# endif()
# @NVIDIA_ORIGINAL: @end
# === @@end: Sanitizer Options ===


# === @@begin: EXTRA_LIBS and Library Dependencies ===
# @NVIDIA_ORIGINAL: @begin: (upstream library dependencies)
# @EUGO_CHANGE: In modern glibc, librt/libpthread/libdl are merged into libc.so.
# We don't need to find or link them separately. RDMA_CORE and MLX5DV are not
# used in our environment.
#
# set(EXTRA_LIBS)
#
# if(RDMA_CORE)
#     add_definitions(-DNCCL_BUILD_RDMA_CORE=1)
#     find_library(VERBS_LIBRARY NAMES verbs)
#     if(VERBS_LIBRARY)
#         list(APPEND EXTRA_LIBS ${VERBS_LIBRARY})
#     endif()
# endif()
#
# if(MLX5DV)
#     add_definitions(-DNCCL_BUILD_MLX5DV=1)
#     find_library(MLX5_LIBRARY NAMES mlx5)
#     if(MLX5_LIBRARY)
#         list(APPEND EXTRA_LIBS ${MLX5_LIBRARY})
#     endif()
# endif()
#
# if(NET_PROFILER)
#     add_definitions(-DNCCL_ENABLE_NET_PROFILING=1)
# endif()
#
# find_library(RT_LIBRARY NAMES rt)
# if(RT_LIBRARY)
#     list(APPEND EXTRA_LIBS ${RT_LIBRARY})
# endif()
# @NVIDIA_ORIGINAL: @end
# === @@end: EXTRA_LIBS and Library Dependencies ===


# === @@begin: Finalized dependencies list ===
# @EUGO_CHANGE:
# In original implementation, they link their binaries against all `CUDAToolkit_LIBRARIES` which is overkill. Based on our research, they only use `cudart` functions here to manage the GPU on the host side and directly access intrinsics (like `fp8`) on the device one.
# They also rely on `CCCL` (in particular, `libcu++`/`libcudacxx`) but as it's the header only library, they just add that into include dirs list and don't explicitely search for it.
#
# In our NCCL implementation, we limit dependencies list to that is exactly used to avoid too tight dependency graph and reduce the number of unneeded libraries loaded at runtime.
#
# In theory, the best solution here is to check every built component (as we have 1 library, 1 (buildtime-only) object library, and 1 executable in this package) and determine on which CUDA parts they depend but it may be overly complicated due to how messy the headers chain in NCCL.
# As after the build we're getting only 2 binaries - `libnccl` (library) and `ncclras` (executable) - and the latter links against former, it's safe and sound to define just a single list of CUDA dependencies for all of them.
#
# @Important:
# Based on our research, NCCL doesn't directly depend on `libcuda` (`CUDA::driver_library`) so we don't add it here (or `cudadevrt`).
# NCCL dynamically loads the CUDA driver API — it never links against it directly. This is a common pattern for libraries that want to support multiple versions of CUDA without requiring users to have a specific version of the CUDA driver installed at build time.
# For example, `DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070); // expands to: extern PFN_cuMemGetHandleForAddressRange_v11070 pfn_cuMemGetHandleForAddressRange` is just a pointer declaration for a function that may be loaded at runtime if available, but it's not linked against `libcuda` at build time.

# @ EUGO_CHANGE: @begin: Makes the system-wide `nvml.h` visible to all targets. Needed for `EUGO_NCCL_NVML_DIRECT_HEADER=1` support.
include_directories("${CUDAToolkit_INCLUDE_DIRS}")
# @ EUGO_CHANGE: @end

set(NCCL_DEPENDENCIES CCCL::libcudacxx CUDA::cudart)
# === @@end: Finalized dependencies list ===
# === @end: dependencies/ ===

# === @begin: src/ ===
add_subdirectory(src)
# === @end: src/ ===

# === @begin: @NVIDIA_ORIGINAL: ===
# @Important:
# We don't build these plugins as they're primarly used for examples and testing:
# 1. Moreover, official `nvidia_nccl_cu13` PyPi wheel doesn't ship them either.
# 2. `ext-net` was used as a foundation by AWS `nccl-ofi-plugin`.
# 3. And the default install target doesn't install them either.
#
# # === @begin: ext-mixed/ ===
# add_subdirectory(ext-mixed/example)
# # === @end: ext-mixed/ ===
#
# # === @begin: ext-net/ ===
# add_subdirectory(ext-net)
# # === @end: ext-net/ ===
#
# # === @begin: ext-profiler/ ===
# add_subdirectory(ext-profiler/example)
# # === @end: ext-profiler/ ===
#
# # === @begin: ext-tuner/ ===
# add_subdirectory(ext-tuner/example)
# # === @end: ext-tuner/ ===
#
# # === @begin: ext-env/ ===
# add_subdirectory(ext-env/example)
# # === @end: ext-env/ ===
# === @end: @NVIDIA_ORIGINAL ===
#
# === @end: Subdirectories ===


# === @begin: DOCA Definitions / @NVIDIA_ORIGINAL: @begin: ===
# add_definitions(-DDOCA_VERBS_USE_CUDA_WRAPPER)
# add_definitions(-DDOCA_VERBS_USE_NET_WRAPPER)
# add_definitions(-DNCCL_GIN_PROXY_ENABLE=1)
# @NVIDIA_ORIGINAL: @end
# @EUGO_CHANGE: DOCA defines moved to target_compile_definitions() in src/CMakeLists.txt
# to ensure they propagate to clang-scan-deps. See src/CMakeLists.txt for the actual defines:
#   DOCA_VERBS_USE_CUDA_WRAPPER, DOCA_VERBS_USE_NET_WRAPPER, DOCA_VERBS_USE_IBV_WRAPPER, NCCL_GIN_PROXY_ENABLE
# Note: upstream is MISSING DOCA_VERBS_USE_IBV_WRAPPER which is required for ibv wrapper to work.
# === @end: DOCA Definitions ===


# === @begin: Install and Configuration Targets ===
include(GNUInstallDirs)

# Install the headers into the include directory
install(FILES ${NCCL_PUBLIC_HEADERS} TYPE INCLUDE)
# @EUGO_CHANGE: @begin: install the nccl_device/ header tree — src/CMakeLists.txt exports
# NCCL_DEVICE_HEADERS_DIR PARENT_SCOPE with a comment promising an `install(DIRECTORY ...)` here,
# but the rule was never written: the installed nccl_device.h:8 does `#include "nccl_device/coop.h"`
# and every consumer of the NCCL 2.28+ symmem device API (e.g. torch's NCCLSymmetricMemory) hit
# "fatal error: 'nccl_device/coop.h' file not found". Upstream's Makefile ships the full tree
# (src/Makefile:11). No trailing slash → lands as include/nccl_device/...; FILES_MATCHING skips
# the tree's README.md. (DOCA gdaki/proxy sub-headers are include-guarded off unless
# NCCL_GIN_{GDAKI,PROXY}_ENABLE is defined — no DOCA staging needed for the default surface.)
install(DIRECTORY ${NCCL_DEVICE_HEADERS_DIR} TYPE INCLUDE FILES_MATCHING PATTERN "*.h")
# @EUGO_CHANGE: @end

# @EUGO_CHANGE: @EUGO_NOTE: Only `nccl` (libnccl.so) and `ncclras` are installed as shippable artifacts.
# The five targets from the ext-*/example add_subdirectory() calls above are intentionally
# NOT listed here:
#   - libnccl-mixed-example.so     → ${CMAKE_BINARY_DIR}/test/unit/plugins/
#   - libnccl-net-example.so       → ${CMAKE_BINARY_DIR}/test/unit/plugins/
#   - libnccl-profiler-example.so  → ${CMAKE_BINARY_DIR}/lib/ (POST_BUILD copy → test/unit/plugins/)
#   - libnccl-tuner-example.so     → ${CMAKE_BINARY_DIR}/test/unit/plugins/
#   - libnccl-env-example.so       → ${CMAKE_BINARY_DIR}/test/unit/plugins/
# These are NVIDIA's reference/example plugins — scaffolding for external developers writing
# their own plugins — not production artifacts. Our production deployments load different
# plugins at runtime (e.g., aws-ofi-nccl for EFA via NCCL_NET_PLUGIN), built and installed as
# separate packages outside this tree. Shipping the examples would pollute the install prefix
# and risk accidental dlopen() by NCCL's plugin chain looking for libnccl-net.so /
# libnccl-tuner.so / libnccl-profiler.so / libnccl-env.so.
#
# If a test harness needs them outside the build tree, reference ${CMAKE_BINARY_DIR} directly
# rather than installing here.
install(
    TARGETS nccl ncclras # `nccl_colldevice` isn't installed separately but used as a private implementation detail for `nccl`
    EXPORT NCCLConfig
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# @TODO+: Implement pkg-config guy and adjust our cmake stuff in a compatible way!
install(
    EXPORT NCCLConfig
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nccl
    NAMESPACE NCCL::
)

# @EUGO_CHANGE: @begin:
# Reference `NCCL` implementation doesn't have `nccl.pc.in` compatible w/ CMake - version and prefix are always left in placholder state, rendering it useless.
# What's more, CMake itself doesn't fill interface dependencies and similar stuff, breaking downstream consumers further as not propagating the required dependencies.
# It's hard for me to envision who would use `pkg-config` to discover `nccl` installation modern days as CUDA support is nearly absent in Make and other `pkg-config` fans, so we simply disable it.
#
# Configure pkg-config file
# configure_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/nccl.pc.in
#     ${CMAKE_BINARY_DIR}/lib/pkgconfig/nccl.pc
#     @ONLY
# )
# @EUGO_CHANGE: @end

