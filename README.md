# GPU-Accelerated LLM Inference Engine

A complete, from-scratch educational implementation of a miniature LLM inference
runtime in **C++20 + CUDA**. Every CUDA kernel has an equivalent CPU fallback
so the entire project compiles and runs on a machine **without an NVIDIA GPU**.

---

## What This Project Teaches

| Concept | Where |
|---------|-------|
| CUDA thread hierarchy (grid/block/thread) | `docs/cuda_kernels.md` |
| Shared memory tiling | `src/matmul_cuda.cu` |
| Parallel reduction (softmax) | `src/attention_cuda.cu` |
| Cache-friendly CPU matmul | `src/matmul_cpu.cpp` |
| Scaled dot-product attention | `src/attention_cpu.cpp` |
| Multi-head attention | `src/attention_cpu.cpp` |
| KV cache — O(n) vs O(n²) | `src/kv_cache.cpp`, `docs/optimization_notes.md` |
| Autoregressive token generation | `src/main.cpp` (MiniTransformer) |
| Numerical stability (softmax) | `docs/optimization_notes.md` |
| GoogleTest unit testing | `tests/` |
| CMake modern build system | `CMakeLists.txt` |

---

## Project Structure

```
LLMInferenceEngine/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── tensor.hpp       — 2-D Tensor class (CPU + optional GPU)
│   ├── matmul.hpp       — Matrix multiply declarations
│   ├── attention.hpp    — Softmax, SDPA, MultiHeadAttention
│   ├── kv_cache.hpp     — KV Cache for autoregressive inference
│   ├── tokenizer.hpp    — Character-level tokenizer
│   └── benchmark.hpp    — Timer, GFLOPS, results table (header-only)
├── src/
│   ├── tensor.cpp
│   ├── matmul_cpu.cpp   — Naive + cache-tiled CPU matmul
│   ├── matmul_cuda.cu   — Naive + shared-memory CUDA kernels
│   ├── attention_cpu.cpp — softmax/SDPA/MHA (CPU)
│   ├── attention_cuda.cu — softmax/SDPA CUDA kernels
│   ├── kv_cache.cpp
│   ├── tokenizer.cpp
│   └── main.cpp         — End-to-end demo (10 phases)
├── tests/
│   ├── test_matmul.cpp
│   ├── test_attention.cpp
│   └── test_kv_cache.cpp
├── benchmarks/
│   ├── benchmark_matmul.cpp
│   └── benchmark_attention.cpp
└── docs/
    ├── architecture.md      — Data flow diagrams
    ├── cuda_kernels.md      — Kernel deep-dives with diagrams
    └── optimization_notes.md — All 5 optimization techniques explained
```

---

## Quick Start — Windows CPU-Only (No GPU Required)

### Prerequisites

1. **Visual Studio 2022** (Community edition is free)
   - Install "Desktop development with C++" workload
2. **CMake 3.18+**
   - Download from cmake.org or: `winget install Kitware.CMake`
3. **Git** (for GoogleTest download during build)
   - `winget install Git.Git`

### Build

```powershell
# Open Developer PowerShell for VS 2022

cd "c:\Users\ABHINAV\Desktop\GPU Accelerated LLM Inference Engine\LLMInferenceEngine"

# Configure (CPU-only, no CUDA needed)
cmake -B build -G "Visual Studio 17 2022" -DUSE_CUDA=OFF

# Build all targets
cmake --build build --config Release

# Run the main demo
.\build\Release\llm_engine.exe

# Run unit tests
cd build
ctest -C Release --output-on-failure
cd ..

# Run benchmarks
.\build\Release\benchmark_matmul.exe
.\build\Release\benchmark_attention.exe
```

### Alternative: MinGW / MSYS2

```bash
cmake -B build -G "MinGW Makefiles" -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/llm_engine
```

---

## CUDA Build (When You Have an NVIDIA GPU)

### Prerequisites

1. **NVIDIA GPU** with Compute Capability 7.5+ (GTX 1660, RTX 20xx, 30xx, 40xx)
2. **CUDA Toolkit 11.8+**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → your Windows version → exe (local)
   - After install, verify: `nvcc --version`
3. Visual Studio 2022 (CUDA integrates with MSVC)

### Build with CUDA

```powershell
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Example output: 8.6  → that's sm_86

cmake -B build -G "Visual Studio 17 2022" ^
      -DUSE_CUDA=ON ^
      -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build --config Release
.\build\Release\llm_engine.exe
```

### Supported GPU Architectures

| GPU Family | Compute Cap | CUDA Arch Flag |
|------------|-------------|----------------|
| GTX 1660, RTX 2080 | 7.5 | `75` |
| A100, RTX 3090 | 8.0 | `80` |
| RTX 3080, RTX 3070 | 8.6 | `86` |
| RTX 4090, RTX 4080 | 8.9 | `89` |

---

## Running on Google Colab (Free GPU)

```python
# Cell 1 — Setup
!git clone <your-repo-url> engine
%cd engine/LLMInferenceEngine

# Cell 2 — Build with CUDA
!apt-get install -y cmake ninja-build
!cmake -B build -GNinja \
       -DUSE_CUDA=ON \
       -DCMAKE_CUDA_ARCHITECTURES=75 \
       -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j4

# Cell 3 — Run
!./build/llm_engine

# Cell 4 — Benchmarks
!./build/benchmark_matmul 64 128 256 512 1024
!./build/benchmark_attention
```

**Colab GPU selection:** Runtime → Change runtime type → GPU
- **T4** (free tier): sm_75, 16 GB VRAM, ~8 TFLOPS FP32
- **A100** (Colab Pro+): sm_80, 40 GB VRAM, ~19.5 TFLOPS FP32

---

## Running the Tests

```powershell
# From the build directory
cd build

# Run all tests
ctest -C Release --output-on-failure

# Run specific test executable
.\Release\test_matmul.exe
.\Release\test_attention.exe
.\Release\test_kv_cache.exe

# Verbose output
.\Release\test_matmul.exe --gtest_verbose=1

# Filter specific tests
.\Release\test_matmul.exe --gtest_filter="MatmulCPUNaive*"
```

### Test Coverage

| Test Suite | Tests | Coverage |
|------------|-------|---------|
| `test_matmul` | 15 tests | Tensor ops, naive matmul, optimised matmul, CUDA matmul |
| `test_attention` | 16 tests | Softmax, SDPA, MHA, CUDA attention |
| `test_kv_cache` | 14 tests | KV cache operations, tokenizer |

---

## Expected Output (CPU-only mode)

```
  ╔══════════════════════════════════════════════════════════╗
  ║     GPU-Accelerated LLM Inference Engine — Demo          ║
  ║     C++20 + CUDA | Educational Implementation            ║
  ╚══════════════════════════════════════════════════════════╝

  Build mode: CPU-only (no GPU required)

==================================================================
  Phase 1 — Tensor Operations
==================================================================
  ── Construction & indexing ──
  A [3 x 4]:
    [  1.0000,   2.0000,   3.0000,   4.0000 ]
    [  5.0000,   6.0000,   7.0000,   8.0000 ]
    [  9.0000,  10.0000,  11.0000,  12.0000 ]

  ── Transpose ──
  Aᵀ [4 x 3]:
    [  1.0000,   5.0000,   9.0000 ]
    [  2.0000,   6.0000,  10.0000 ]
  ...

==================================================================
  Phase 8 — Autoregressive MiniTransformer Generation
==================================================================
  [prefill] 4 tokens...
  [generate] up to 10 tokens...
  Generated sequence (IDs): [ 2 5 10 7 ... ]
```

---

## Benchmark Output Example (CPU)

```
========================================================================
  Matrix Multiplication Benchmark Results
========================================================================
Implementation       Matrix Size    Time (ms)    GFLOPS   Speedup
------------------------------------------------------------------------
CPU Naive            64x64              0.42       1.26    1.0x
CPU Optimised        64x64              0.07       7.54    6.0x
CPU Naive            256x256           53.12       0.64    1.0x
CPU Optimised        256x256            7.83       4.31    6.8x
CPU Naive            512x512          [skipped]
CPU Optimised        512x512           58.21       4.63    1.0x
========================================================================
```

---

## Architecture Deep Dive

See `docs/` for detailed documentation:

- [**architecture.md**](docs/architecture.md) — Data flow diagrams, component map, extending the engine
- [**cuda_kernels.md**](docs/cuda_kernels.md) — Every kernel explained: grid/block/thread mapping, shared memory, occupancy
- [**optimization_notes.md**](docs/optimization_notes.md) — 5 optimization techniques with code examples and benchmarks

---

## Key Concepts Reference

### Scaled Dot-Product Attention
```
Attention(Q,K,V) = softmax( Q·Kᵀ / √d_k ) · V

Q : [seq_q, d_k]   K : [seq_k, d_k]   V : [seq_k, d_v]
→ Output: [seq_q, d_v]
```

### KV Cache Complexity
```
Without cache:  O(n²)  — recompute K,V for all previous tokens each step
With cache:     O(n)   — only compute K,V for the new token
For n=512:      512× fewer matmul operations!
```

### Numerically Stable Softmax
```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
              ← subtracting max prevents exp() overflow, is mathematically equivalent
```

### Tiled MatMul Memory Savings
```
Without tiles: 2·M·N·K global memory reads
With TILE=16:  2·M·N·K / TILE global reads + 2·M·N·K / TILE shared reads
               → 16× reduction in global memory bandwidth
```

---

## License

This is an educational project — use freely for learning. For production use,
consider [cuBLAS](https://developer.nvidia.com/cublas),
[FlashAttention](https://github.com/Dao-AILab/flash-attention), or
[vLLM](https://github.com/vllm-project/vllm).
