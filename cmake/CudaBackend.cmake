# CUDA / OptiX backend (-DSPECTRAL_ENABLE_CUDA=ON).
#   OPTIX_ROOT: OptiX SDK directory (containing include/optix.h), or set the OPTIX_ROOT env var.
#   CMAKE_CUDA_ARCHITECTURES: PTX target (default 86; the driver JIT-compiles PTX for newer GPUs
#   such as Blackwell sm_120 -- set 120 explicitly with CUDA >= 12.8 to target it directly).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
enable_language(CUDA)
find_package(CUDAToolkit 12.0 REQUIRED)

set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "OptiX SDK root (contains include/optix.h)")
find_path(OPTIX_INCLUDE_DIR optix.h HINTS "${OPTIX_ROOT}/include" PATHS
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include" /opt/optix/include)
if(NOT OPTIX_INCLUDE_DIR)
  message(FATAL_ERROR "optix.h not found: set -DOPTIX_ROOT=/path/to/OptiX-SDK")
endif()
message(STATUS "OptiX headers: ${OPTIX_INCLUDE_DIR}")

# Device programs -> PTX, embedded as a byte array into libspectral.
add_library(spectral_optix_ptx OBJECT src/backend/cuda/programs/device_programs.cu)
set_target_properties(spectral_optix_ptx PROPERTIES CUDA_PTX_COMPILATION ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(spectral_optix_ptx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${OPTIX_INCLUDE_DIR})
target_compile_definitions(spectral_optix_ptx PRIVATE SPECTRAL_MAX_BANDS=${SPECTRAL_MAX_BANDS})
target_compile_options(spectral_optix_ptx PRIVATE --use_fast_math -lineinfo --expt-relaxed-constexpr)

set(SPECTRAL_PTX_CPP ${CMAKE_CURRENT_BINARY_DIR}/spectral_optix_ptx_embedded.cpp)
add_custom_command(OUTPUT ${SPECTRAL_PTX_CPP}
  COMMAND ${CMAKE_COMMAND} "-DINPUT=$<TARGET_OBJECTS:spectral_optix_ptx>" -DOUTPUT=${SPECTRAL_PTX_CPP}
          -DSYMBOL=spectral_optix_ptx -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
  DEPENDS spectral_optix_ptx $<TARGET_OBJECTS:spectral_optix_ptx> ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
  COMMAND_EXPAND_LISTS
  VERBATIM)

target_sources(spectral PRIVATE src/backend/cuda/cuda_backend.cpp ${SPECTRAL_PTX_CPP})
target_include_directories(spectral PRIVATE ${OPTIX_INCLUDE_DIR})
target_link_libraries(spectral PRIVATE CUDA::cudart CUDA::cuda_driver)
target_compile_definitions(spectral PRIVATE SPECTRAL_HAS_CUDA=1)
