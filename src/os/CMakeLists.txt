# if(NCCL_OS_LINUX) # @EUGO_CHANGE: We moved all sources to `src/CMakeLists.txt` (flat NCCL_SRC_FILES list). --- IGNORE ---
#     set(OS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/linux.cc PARENT_SCOPE)
# elseif(NCCL_OS_WINDOWS)
#     set(OS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/windows.cc PARENT_SCOPE)
# endif()
