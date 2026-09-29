file(GLOB SPECTRAL_TEST_SOURCES CONFIGURE_DEPENDS unit/*.cpp integration/*.cpp)
add_executable(spectral_tests test_main.cpp ${SPECTRAL_TEST_SOURCES})
target_link_libraries(spectral_tests PRIVATE spectral)
target_include_directories(spectral_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/third_party/doctest
  ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(spectral_tests PRIVATE
  SPECTRAL_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  SPECTRAL_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp"
  SPECTRAL_TEST_SKY_PLUGIN="$<TARGET_FILE:test_sky_plugin>")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tmp)

# Kernel headers must stay free of host-only constructs (they are compiled by nvcc too).
add_library(kernel_headers_strict OBJECT unit/strict/kernel_headers_strict.cpp)
target_include_directories(kernel_headers_strict PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(kernel_headers_strict PRIVATE SPECTRAL_MAX_BANDS=${SPECTRAL_MAX_BANDS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kernel_headers_strict PRIVATE -fno-exceptions -fno-rtti -Wall -Wextra -Werror -Wno-unused-parameter)
endif()
add_dependencies(spectral_tests kernel_headers_strict)

# Minimal sky plugin implementing the C ABI (constant sky), used by tests.
add_library(test_sky_plugin MODULE plugins/test_sky_plugin.c)
target_include_directories(test_sky_plugin PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(test_sky_plugin PROPERTIES C_VISIBILITY_PRESET hidden)
add_dependencies(spectral_tests test_sky_plugin)

add_test(NAME spectral_tests COMMAND spectral_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(spectral_tests PROPERTIES ENVIRONMENT "SPECTRAL_CACHE_DIR=${CMAKE_BINARY_DIR}/cache")
