# === @begin: `ncclras` executable ===
# Create an executable target named "ncclras"
add_executable(
  ncclras

  # Source files
  client.cc

  # NOTE: Below is intentionally commented out. They are only used with `libnccl` and are not needed for `ncclras` executable!
  # client_support.cc
  # collectives.cc
  # peers.cc
  # ras.cc
  # rasnet.cc
)


# `PRIVATE` as you couldn't have `PUBLIC` or `INTERFACE` `compile_definitions` in executables
target_compile_definitions(ncclras PRIVATE ${NCCL_COMMON_COMPILE_DEFINITIONS})


# Header files - refactored to use `target_include_directories()`
# `ras_internal.h` is a private header, so it's not added to `NCCL_PUBLIC_HEADERS` list! (as `libnccl` compiles even without `ncclras`)
target_include_directories(ncclras PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# We don't need to explicitely link against CUDA libraries as `ncclras` doesn't call their APIs at all and `libnccl` who uses them (including its interfaces) properly declares its dependency on CUDA libraries in its own `target_link_libraries`.
# `PRIVATE` as you couldn't have `PUBLIC` or `INTERFACE` `link_libraries` in executables
#
# @Important:
# We keep this as a reference to original implementation but in reality we only use `libnccl` headers.
# `ncclras` communicates w/ running NCCL process via sockets and uses a few header-only definitions from it only. None of symbols requiring linkage are used.
#
# Another reason to keep it as is to avoid manually resetting all required header paths and `-D` macros.
target_link_libraries(ncclras PRIVATE nccl)
# === @end: `ncclras` executable ===