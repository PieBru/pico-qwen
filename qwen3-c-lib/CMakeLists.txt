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

# SIMD optimizations
if(ARCH_X86_64)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mavx2 -mfma")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mavx512f -mavx512vl -mavx512vnni")
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
    src/simd/avx2_kernels.c
    src/simd/avx512_kernels.c
    src/simd/neon_kernels.c
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

# Test executable
add_executable(test_qwen3_c
    tests/test_all.c
    tests/test_memory.c
    tests/test_tensor.c
    tests/test_matmul.c
    tests/test_attention.c
    tests/test_inference.c
)

# Link test executable
target_link_libraries(test_qwen3_c qwen3_inference)

# Add tests
add_test(NAME test_qwen3_c COMMAND test_qwen3_c)

# Benchmark executable
add_executable(benchmark_qwen3_c
    tests/benchmark.c
)

target_link_libraries(benchmark_qwen3_c qwen3_inference)

# Custom targets
add_custom_target(tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_qwen3_c
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