# CUDA / OptiX backend (-DSPECTRAL_ENABLE_CUDA=ON). Works without extra options on a machine with the CUDA
# toolkit and an NVIDIA driver:
#   OPTIX_ROOT:              OptiX SDK directory (contains include/optix.h), or the OPTIX_ROOT env var. When not
#                            found, the header-only NVIDIA/optix-dev repository is fetched (SPECTRAL_FETCH_OPTIX).
#                            The OptiX runtime itself ships with the driver.
#   SPECTRAL_OPTIX_TAG:      optix-dev tag to fetch (v9.0.0 needs driver R570+, v9.1.0 a newer driver).
#   CMAKE_CUDA_ARCHITECTURES: default = compute capability of GPU 0 from nvidia-smi (Blackwell RTX PRO 6000 -> 120),
#                            capped at what the installed nvcc supports; 86 when no GPU is visible (PTX is
#                            JIT-compiled by the driver for newer GPUs).
option(SPECTRAL_FETCH_OPTIX "Download OptiX headers (github.com/NVIDIA/optix-dev) when optix.h is not found" ON)
set(SPECTRAL_OPTIX_TAG "v9.0.0" CACHE STRING "NVIDIA/optix-dev tag fetched when OPTIX_ROOT is not set")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(_spectral_arch 86)
  find_program(_spectral_nvidia_smi nvidia-smi)
  if(_spectral_nvidia_smi)
    execute_process(COMMAND ${_spectral_nvidia_smi} --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE _cc RESULT_VARIABLE _cc_res ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_cc_res EQUAL 0 AND _cc MATCHES "^([0-9]+)\\.([0-9]+)")
      set(_spectral_arch "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    endif()
  endif()
  # nvcc version (before enable_language, whose compiler check already uses the architecture).
  set(_nvcc "${CMAKE_CUDA_COMPILER}")
  if(NOT _nvcc AND DEFINED ENV{CUDACXX})
    set(_nvcc "$ENV{CUDACXX}")
  endif()
  if(NOT _nvcc)
    find_program(_nvcc nvcc HINTS /usr/local/cuda/bin)
  endif()
  if(_nvcc)
    execute_process(COMMAND ${_nvcc} --version OUTPUT_VARIABLE _nvcc_out ERROR_QUIET)
    if(_nvcc_out MATCHES "release ([0-9]+\\.[0-9]+)")
      set(_nvcc_ver ${CMAKE_MATCH_1})
      if(_spectral_arch GREATER_EQUAL 100 AND _nvcc_ver VERSION_LESS 12.8)
        message(WARNING "nvcc ${_nvcc_ver} cannot target sm_${_spectral_arch} (needs CUDA 12.8+); using PTX for sm_90 "
                        "(JIT-compiled by the driver). Install CUDA 12.8+ for native Blackwell code.")
        set(_spectral_arch 90)
      endif()
    endif()
    if(NOT CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER "${_nvcc}")
    endif()
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${_spectral_arch})
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
enable_language(CUDA)
find_package(CUDAToolkit 12.0 REQUIRED)

set(OPTIX_ROOT "$ENV{OPTIX_ROOT}" CACHE PATH "OptiX SDK root (contains include/optix.h)")
find_path(OPTIX_INCLUDE_DIR optix.h HINTS "${OPTIX_ROOT}/include" PATHS
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include" /opt/optix/include)
if(NOT OPTIX_INCLUDE_DIR AND SPECTRAL_FETCH_OPTIX)
  include(FetchContent)
  message(STATUS "optix.h not found: fetching NVIDIA/optix-dev ${SPECTRAL_OPTIX_TAG}")
  # SOURCE_SUBDIR has no CMakeLists.txt, so MakeAvailable only downloads (headers only).
  FetchContent_Declare(optix_dev GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
                       GIT_TAG ${SPECTRAL_OPTIX_TAG} GIT_SHALLOW TRUE SOURCE_SUBDIR include)
  FetchContent_MakeAvailable(optix_dev)
  set(OPTIX_INCLUDE_DIR "${optix_dev_SOURCE_DIR}/include" CACHE PATH "OptiX include directory" FORCE)
endif()
if(NOT OPTIX_INCLUDE_DIR OR NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
  message(FATAL_ERROR "optix.h not found: set -DOPTIX_ROOT=/path/to/OptiX-SDK (or allow -DSPECTRAL_FETCH_OPTIX=ON)")
endif()
message(STATUS "OptiX headers: ${OPTIX_INCLUDE_DIR}")

# Device programs -> PTX, embedded as a byte array into libspectral.
add_library(spectral_optix_ptx OBJECT src/backend/cuda/programs/device_programs.cu)
set_target_properties(spectral_optix_ptx PROPERTIES CUDA_PTX_COMPILATION ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(spectral_optix_ptx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${OPTIX_INCLUDE_DIR})
target_compile_definitions(spectral_optix_ptx PRIVATE SPECTRAL_MAX_BANDS=${SPECTRAL_MAX_BANDS})
# -rdc=true keeps `params` a .visible symbol that OptiX can bind (as in the OptiX SDK).
target_compile_options(spectral_optix_ptx PRIVATE --use_fast_math -lineinfo --expt-relaxed-constexpr -rdc=true)

set(SPECTRAL_PTX_CPP ${CMAKE_CURRENT_BINARY_DIR}/spectral_optix_ptx_embedded.cpp)
add_custom_command(OUTPUT ${SPECTRAL_PTX_CPP}
  COMMAND ${CMAKE_COMMAND} "-DINPUT=$<TARGET_OBJECTS:spectral_optix_ptx>" -DOUTPUT=${SPECTRAL_PTX_CPP}
          -DSYMBOL=spectral_optix_ptx -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
  DEPENDS spectral_optix_ptx $<TARGET_OBJECTS:spectral_optix_ptx> ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake
  COMMAND_EXPAND_LISTS
  VERBATIM)

target_sources(spectral PRIVATE src/backend/cuda/cuda_backend.cpp src/backend/cuda/preview_kernels.cu ${SPECTRAL_PTX_CPP})
# `spectral` was created before CMAKE_CUDA_ARCHITECTURES was set here, so set its architectures explicitly.
set_target_properties(spectral PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                                          CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_options(spectral PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_include_directories(spectral PRIVATE ${OPTIX_INCLUDE_DIR})
target_link_libraries(spectral PRIVATE CUDA::cudart CUDA::cuda_driver)
target_compile_definitions(spectral PRIVATE SPECTRAL_HAS_CUDA=1)
