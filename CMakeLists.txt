# =============================================================================
# GPU-Accelerated LLM Inference Engine
# CMakeLists.txt
# =============================================================================
# Build modes:
#   CPU-only (default):  cmake -B build -DUSE_CUDA=OFF
#   With CUDA:           cmake -B build -DUSE_CUDA=ON
#
# Windows CPU-only build (no GPU required):
#   cmake -B build -G "Visual Studio 17 2022" -DUSE_CUDA=OFF
#   cmake --build build --config Release
#
# Linux/Colab CUDA build:
#   cmake -B build -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
#   cmake --build build
# =============================================================================

cmake_minimum_required(VERSION 3.18)

# Project declaration — CUDA language enabled conditionally below
project(LLMInferenceEngine VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# Options
# =============================================================================
option(USE_CUDA    "Build with CUDA GPU support"            OFF)
option(BUILD_TESTS "Build GoogleTest unit tests"            ON)
option(BUILD_BENCH "Build standalone benchmark executables" ON)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# CUDA Setup (optional)
# =============================================================================
if(USE_CUDA)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Default to sm_75 (Turing) — covers GTX 16xx / RTX 20xx and up.
    # Override on command line: -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()

    add_compile_definitions(USE_CUDA)
    message(STATUS "[CUDA] Enabled — architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "[CPU-only] Building without CUDA — all GPU paths use CPU fallback")
endif()

# =============================================================================
# Compiler Flags
# =============================================================================
if(MSVC)
    add_compile_options(/W3 /WX- /MP)          # multi-processor build
    add_compile_options(/wd4068)               # suppress unknown-pragma warnings
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

# =============================================================================
# Include Directories
# =============================================================================
include_directories(${PROJECT_SOURCE_DIR}/include)

# =============================================================================
# Source File Lists
# =============================================================================
set(CPU_SOURCES
    ${PROJECT_SOURCE_DIR}/src/tensor.cpp
    ${PROJECT_SOURCE_DIR}/src/matmul_cpu.cpp
    ${PROJECT_SOURCE_DIR}/src/attention_cpu.cpp
    ${PROJECT_SOURCE_DIR}/src/kv_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/mini_transformer.cpp
    ${PROJECT_SOURCE_DIR}/src/quantize.cpp
    ${PROJECT_SOURCE_DIR}/src/scheduler.cpp
)

if(USE_CUDA)
    set(CUDA_SOURCES
        ${PROJECT_SOURCE_DIR}/src/matmul_cuda.cu
        ${PROJECT_SOURCE_DIR}/src/attention_cuda.cu
    )
    set(ALL_SOURCES ${CPU_SOURCES} ${CUDA_SOURCES})
else()
    set(ALL_SOURCES ${CPU_SOURCES})
endif()

# =============================================================================
# Main Executable
# =============================================================================
add_executable(llm_engine ${PROJECT_SOURCE_DIR}/src/main.cpp ${ALL_SOURCES})

if(USE_CUDA)
    set_target_properties(llm_engine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(llm_engine PRIVATE cudart)
endif()

# =============================================================================
# GoogleTest (downloaded automatically on first CMake configure)
# =============================================================================
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # Prevent GTest from overriding our compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # ── test_matmul ──────────────────────────────────────────────────────────
    add_executable(test_matmul
        ${PROJECT_SOURCE_DIR}/tests/test_matmul.cpp
        ${ALL_SOURCES}
    )
    target_link_libraries(test_matmul PRIVATE GTest::gtest_main)
    if(USE_CUDA)
        set_target_properties(test_matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(test_matmul PRIVATE cudart)
    endif()

    # ── test_attention ───────────────────────────────────────────────────────
    add_executable(test_attention
        ${PROJECT_SOURCE_DIR}/tests/test_attention.cpp
        ${ALL_SOURCES}
    )
    target_link_libraries(test_attention PRIVATE GTest::gtest_main)
    if(USE_CUDA)
        set_target_properties(test_attention PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(test_attention PRIVATE cudart)
    endif()

    # ── test_kv_cache ────────────────────────────────────────────────────────
    add_executable(test_kv_cache
        ${PROJECT_SOURCE_DIR}/tests/test_kv_cache.cpp
        ${ALL_SOURCES}
    )
    target_link_libraries(test_kv_cache PRIVATE GTest::gtest_main)
    if(USE_CUDA)
        set_target_properties(test_kv_cache PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(test_kv_cache PRIVATE cudart)
    endif()

    # ── test_quantize ────────────────────────────────────────────────────────
    add_executable(test_quantize
        ${PROJECT_SOURCE_DIR}/tests/test_quantize.cpp
        ${ALL_SOURCES}
    )
    target_link_libraries(test_quantize PRIVATE GTest::gtest_main)
    if(USE_CUDA)
        set_target_properties(test_quantize PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(test_quantize PRIVATE cudart)
    endif()

    include(GoogleTest)
    gtest_discover_tests(test_matmul)
    gtest_discover_tests(test_attention)
    gtest_discover_tests(test_kv_cache)
    gtest_discover_tests(test_quantize)
endif()

# =============================================================================
# Benchmarks
# =============================================================================
if(BUILD_BENCH)
    # ── benchmark_matmul ─────────────────────────────────────────────────────
    add_executable(benchmark_matmul
        ${PROJECT_SOURCE_DIR}/benchmarks/benchmark_matmul.cpp
        ${ALL_SOURCES}
    )
    if(USE_CUDA)
        set_target_properties(benchmark_matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(benchmark_matmul PRIVATE cudart)
    endif()

    # ── benchmark_attention ──────────────────────────────────────────────────
    add_executable(benchmark_attention
        ${PROJECT_SOURCE_DIR}/benchmarks/benchmark_attention.cpp
        ${ALL_SOURCES}
    )
    if(USE_CUDA)
        set_target_properties(benchmark_attention PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(benchmark_attention PRIVATE cudart)
    endif()

    # ── benchmark_quantize ───────────────────────────────────────────────────
    add_executable(benchmark_quantize
        ${PROJECT_SOURCE_DIR}/benchmarks/benchmark_quantize.cpp
        ${ALL_SOURCES}
    )
    if(USE_CUDA)
        set_target_properties(benchmark_quantize PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(benchmark_quantize PRIVATE cudart)
    endif()

    # ── benchmark_batching ───────────────────────────────────────────────────
    add_executable(benchmark_batching
        ${PROJECT_SOURCE_DIR}/benchmarks/benchmark_batching.cpp
        ${ALL_SOURCES}
    )
    if(USE_CUDA)
        set_target_properties(benchmark_batching PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(benchmark_batching PRIVATE cudart)
    endif()
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== LLM Inference Engine Build Summary ===")
message(STATUS "  CUDA support : ${USE_CUDA}")
message(STATUS "  Build tests  : ${BUILD_TESTS}")
message(STATUS "  Build bench  : ${BUILD_BENCH}")
message(STATUS "  CXX standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "==========================================")
message(STATUS "")
