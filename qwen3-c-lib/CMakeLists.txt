cmake_minimum_required(VERSION 3.16)
project(qwen3-c-lib VERSION 0.1.0 LANGUAGES C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Build configuration
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-align")
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -funroll-loops -fomit-frame-pointer -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-align")

# Architecture detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_X86_64 ON)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -m64 -march=native -mtune=native")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_ARM64 ON)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mcpu=native -mtune=native")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(ARCH_ARM ON)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mfpu=neon")
endif()

# SIMD optimizations - detect actual CPU features
if(ARCH_X86_64)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mavx2 -mfma")
    # Check for AVX-512 support
    execute_process(
        COMMAND grep -q "avx512" /proc/cpuinfo
        RESULT_VARIABLE HAS_AVX512
    )
    if(HAS_AVX512 EQUAL 0)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mavx512f -mavx512vl")
    endif()
elseif(ARCH_ARM64 OR ARCH_ARM)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mfpu=neon")
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simd
)

# Source files
set(SOURCES
    src/qwen3_inference.c
    src/model.c
    src/transformer.c
    src/attention.c
    src/matrix.c
    src/tensor.c
    src/tokenizer.c
    src/sampler.c
    src/memory.c
    src/utils.c
    src/simd/cpu_detect.c
)

# Create static library
add_library(qwen3_inference STATIC ${SOURCES})

# Create shared library
add_library(qwen3_inference_shared SHARED ${SOURCES})
set_target_properties(qwen3_inference_shared PROPERTIES OUTPUT_NAME qwen3_inference)

# Link math library
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(qwen3_inference PRIVATE ${MATH_LIBRARY})
    target_link_libraries(qwen3_inference_shared PRIVATE ${MATH_LIBRARY})
endif()

# Platform-specific linking
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(RT_LIBRARY rt)
    if(RT_LIBRARY)
        target_link_libraries(qwen3_inference PRIVATE ${RT_LIBRARY})
        target_link_libraries(qwen3_inference_shared PRIVATE ${RT_LIBRARY})
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    find_library(FOUNDATION_LIBRARY Foundation)
    if(FOUNDATION_LIBRARY)
        target_link_libraries(qwen3_inference PRIVATE ${FOUNDATION_LIBRARY})
        target_link_libraries(qwen3_inference_shared PRIVATE ${FOUNDATION_LIBRARY})
    endif()
endif()

# Compiler definitions
target_compile_definitions(qwen3_inference PRIVATE)
target_compile_definitions(qwen3_inference_shared PRIVATE)

# Compiler flags for position-independent code
set_property(TARGET qwen3_inference PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET qwen3_inference_shared PROPERTY POSITION_INDEPENDENT_CODE ON)

# Install targets
install(TARGETS qwen3_inference qwen3_inference_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/qwen3_inference.h
    DESTINATION include
)

# Testing
enable_testing()

# Test executables
add_executable(test_matrix tests/test_matrix.c)
add_executable(test_attention tests/test_attention.c)
add_executable(test_model_loading tests/test_model_loading.c)
add_executable(test_transformer tests/test_transformer.c)
add_executable(test_inference tests/test_inference.c)

# Link test executables
target_link_libraries(test_matrix qwen3_inference)
target_link_libraries(test_attention qwen3_inference)
target_link_libraries(test_model_loading qwen3_inference)
target_link_libraries(test_transformer qwen3_inference)
target_link_libraries(test_inference qwen3_inference)

# Add tests
add_test(NAME test_matrix COMMAND test_matrix)
add_test(NAME test_attention COMMAND test_attention)
add_test(NAME test_model_loading COMMAND test_model_loading)
add_test(NAME test_transformer COMMAND test_transformer)
add_test(NAME test_inference COMMAND test_inference)

# Benchmark executable
add_executable(benchmark_qwen3_c
    tests/benchmark.c
)

target_link_libraries(benchmark_qwen3_c qwen3_inference)

# Custom targets
add_custom_target(tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_matrix test_attention test_model_loading test_transformer test_inference
)

add_custom_target(benchmark
    COMMAND benchmark_qwen3_c
    DEPENDS benchmark_qwen3_c
)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C flags: ${CMAKE_C_FLAGS}")
message(STATUS "C flags (Release): ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")

# Feature summary
if(ARCH_X86_64)
    message(STATUS "x86_64 architecture detected - enabling AVX2/AVX-512")
elseif(ARCH_ARM64)
    message(STATUS "ARM64 architecture detected - enabling NEON")
elseif(ARCH_ARM)
    message(STATUS "ARM architecture detected - enabling NEON")
endif()