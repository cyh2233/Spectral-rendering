cmake_minimum_required(VERSION 3.20)
project(spectral VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SPECTRAL_ENABLE_CUDA "Build the CUDA/OptiX backend" OFF)
option(SPECTRAL_ENABLE_PRAGUE_SKY "Build the Prague sky model" ON)
option(SPECTRAL_BUILD_TESTS "Build unit/integration tests" ON)
option(SPECTRAL_BUILD_SHARED "Build libspectral as a shared library" OFF)
option(SPECTRAL_BUILD_PYTHON "Build the Python module (_spectral, package python/spectral_renderer)" OFF)
set(SPECTRAL_MAX_BANDS 128 CACHE STRING "Maximum number of spectral bands per path")

find_package(OpenMP)
find_package(Threads REQUIRED)
include(GNUInstallDirs)

add_subdirectory(third_party)

set(SPECTRAL_SOURCES
  src/api/renderer.cpp
  src/api/session.cpp
  src/backend/cpu/bvh.cpp
  src/backend/cpu/cpu_backend.cpp
  src/io/csv_spectrum.cpp
  src/io/image_io.cpp
  src/scene/gltf_loader.cpp
  src/scene/primitives.cpp
  src/scene/scene.cpp
  src/scene/scene_json.cpp
  src/sky/sky.cpp
  src/spectrum/cie_tables.cpp
  src/spectrum/dense_spectrum.cpp
  src/spectrum/glass.cpp
  src/spectrum/uplift_lut.cpp
  src/util/paths.cpp
)

if(SPECTRAL_BUILD_SHARED)
  add_library(spectral SHARED ${SPECTRAL_SOURCES})
else()
  add_library(spectral STATIC ${SPECTRAL_SOURCES})
endif()
add_library(spectral::spectral ALIAS spectral)

target_include_directories(spectral
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(spectral SYSTEM PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>)
target_compile_definitions(spectral
  PUBLIC SPECTRAL_MAX_BANDS=${SPECTRAL_MAX_BANDS}
  PRIVATE
    SPECTRAL_SOURCE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    SPECTRAL_INSTALL_DATA_DIR="${CMAKE_INSTALL_FULL_DATADIR}/spectral")
target_link_libraries(spectral PRIVATE spectral_thirdparty spectral_rgb2spec Threads::Threads ${CMAKE_DL_LIBS})
if(OpenMP_CXX_FOUND)
  target_link_libraries(spectral PUBLIC OpenMP::OpenMP_CXX)
endif()
if(SPECTRAL_ENABLE_PRAGUE_SKY)
  target_link_libraries(spectral PRIVATE prague_sky_model)
  target_compile_definitions(spectral PRIVATE SPECTRAL_HAS_PRAGUE=1)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(spectral PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(spectral PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-maybe-uninitialized>)
  endif()
elseif(MSVC)
  target_compile_options(spectral PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /bigobj>)
  target_compile_definitions(spectral PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

if(SPECTRAL_ENABLE_CUDA)
  include(cmake/CudaBackend.cmake)
endif()

add_executable(spectral_render apps/spectral_render/main.cpp)
target_link_libraries(spectral_render PRIVATE spectral)

add_executable(spectral_uplift_lut apps/spectral_uplift_lut/main.cpp)
target_link_libraries(spectral_uplift_lut PRIVATE spectral)

if(SPECTRAL_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  set(PYBIND11_FINDPYTHON ON)
  add_subdirectory(third_party/pybind11)
  pybind11_add_module(_spectral python/spectral_py.cpp)
  target_link_libraries(_spectral PRIVATE spectral)
  # The package directory is python/spectral_renderer; the extension is built next to a copy of it.
  set_target_properties(_spectral PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/spectral_renderer)
  # Copied by its own rule so edits to the pure-Python wrapper are picked up without relinking the extension.
  set(_py_init_src ${CMAKE_CURRENT_SOURCE_DIR}/python/spectral_renderer/__init__.py)
  set(_py_init_dst ${CMAKE_BINARY_DIR}/python/spectral_renderer/__init__.py)
  add_custom_command(OUTPUT ${_py_init_dst}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_py_init_src} ${_py_init_dst}
    DEPENDS ${_py_init_src})
  add_custom_target(spectral_renderer_py ALL DEPENDS ${_py_init_dst})
  add_dependencies(_spectral spectral_renderer_py)
endif()

if(SPECTRAL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------- install / export
set(SPECTRAL_EXPORT_TARGETS spectral spectral_thirdparty spectral_rgb2spec spectral_render spectral_uplift_lut)
if(SPECTRAL_ENABLE_PRAGUE_SKY)
  list(APPEND SPECTRAL_EXPORT_TARGETS prague_sky_model)
endif()
install(TARGETS ${SPECTRAL_EXPORT_TARGETS} EXPORT spectralTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/spectral DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES third_party/nlohmann/json.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nlohmann)
install(DIRECTORY data/ DESTINATION ${CMAKE_INSTALL_DATADIR}/spectral)
install(EXPORT spectralTargets NAMESPACE spectral:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spectral)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/spectralConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/spectralConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spectral)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/spectralConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/spectralConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/spectralConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spectral)
