#pragma once
// =============================================================================
// matmul.hpp — Matrix multiplication (CPU naive, CPU tiled, CUDA naive, CUDA tiled)
// =============================================================================
// Notation:
//   A : [M × K]   B : [K × N]   C : [M × N]   where C = A × B
//
// Complexity:
//   All variants compute the same 2·M·N·K FLOPs.
//   Differences are purely in cache / memory-bandwidth efficiency.
// =============================================================================

#include "tensor.hpp"

// =============================================================================
// Validation
// =============================================================================
// Throws std::invalid_argument if A.cols != B.rows.
void validate_matmul_dims(const Tensor& A, const Tensor& B);

// =============================================================================
// CPU — Naive  O(M·N·K)
// =============================================================================
// Standard triple loop.  B is accessed column-by-column → cache-unfriendly.
// Use as correctness reference and baseline for benchmarks.
Tensor matmul_cpu_naive(const Tensor& A, const Tensor& B);

// =============================================================================
// CPU — Optimised (Blocked / Cache-Tiled)
// =============================================================================
// Partitions the iteration space into BLOCK_SIZE³ sub-problems that fit in L1
// cache.  Same FLOPs, drastically better cache hit rate.
//
//   BLOCK_SIZE = 64: tile = 64×64×4 = 16 KB — two tiles fit in 32 KB L1 cache.
//
// Typical speedup over naive on modern x86: 4–10×.
Tensor matmul_cpu_optimized(const Tensor& A, const Tensor& B);

#ifdef USE_CUDA
// =============================================================================
// CUDA — Naive kernel
// =============================================================================
// Grid:   ceil(N/32) × ceil(M/32) blocks   Block: 32×32 threads
// Thread mapping:  one thread → one output element C[row][col]
// Global mem access:  A row-sequential (good), B column-sequential (bad).
// FYI: "bad" column access causes ~32 uncoalesced reads per warp → low BW util.
Tensor matmul_cuda_naive(const Tensor& A, const Tensor& B);

// =============================================================================
// CUDA — Tiled shared-memory kernel
// =============================================================================
// Grid:   ceil(N/16) × ceil(M/16) blocks   Block: 16×16 = 256 threads
//
// Algorithm:
//   1. Each thread cooperatively loads one A element and one B element into
//      __shared__ float As[16][16] and Bs[16][16].
//   2. __syncthreads() — barrier before compute.
//   3. Each thread computes partial dot product over the tile.
//   4. Advance tile, repeat for ceil(K/16) tiles.
//
// Memory savings:  Each global element is loaded once into shared mem and
//   reused 16 times by other threads in the block → 16× bandwidth reduction.
//
// Occupancy (A100, sm_80):
//   Shared mem per block: 2 × 16 × 16 × 4 = 2 KB
//   Max blocks per SM: 32 KB / 2 KB = 16 — very high occupancy.
Tensor matmul_cuda_tiled(const Tensor& A, const Tensor& B);
#endif // USE_CUDA

// =============================================================================
// Dispatch — best available
// =============================================================================
// Automatically selects: CUDA tiled (if USE_CUDA) or CPU optimised (fallback).
Tensor matmul(const Tensor& A, const Tensor& B);
