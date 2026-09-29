# Vendored third-party code (see third_party/README.md for versions and licenses).
add_library(spectral_thirdparty STATIC thirdparty_impl.cpp tinyexr/miniz.c)
target_include_directories(spectral_thirdparty PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tinyexr> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stb>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tinygltf> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nlohmann>)
set_target_properties(spectral_thirdparty PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(spectral_thirdparty PRIVATE -w)
endif()

# rgb2spec optimizer (Jakob & Hanika 2019), linked as a function for on-demand LUT generation.
add_library(spectral_rgb2spec STATIC rgb2spec/rgb2spec_opt.cpp)
target_compile_definitions(spectral_rgb2spec PRIVATE main=spectral_rgb2spec_main)
target_include_directories(spectral_rgb2spec PRIVATE rgb2spec)
target_link_libraries(spectral_rgb2spec PRIVATE Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(spectral_rgb2spec PRIVATE -w)
endif()

if(SPECTRAL_ENABLE_PRAGUE_SKY)
  add_library(prague_sky_model STATIC prague_sky_model/PragueSkyModel.cpp)
  target_include_directories(prague_sky_model PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/prague_sky_model>)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(prague_sky_model PRIVATE -w)
  endif()
endif()
