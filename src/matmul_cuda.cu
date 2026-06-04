// =============================================================================
// matmul_cuda.cu — CUDA matrix multiplication kernels
// =============================================================================
// This file is compiled ONLY when USE_CUDA=ON in CMake (nvcc target).
// It provides two kernels:
//   1. matmul_naive_kernel  — one thread per output element, simple but slow
//   2. matmul_tiled_kernel  — 16×16 shared-memory tiles, fast
//
// Both compute:   C = A × B
//   A : [M × K]  B : [K × N]  C : [M × N]
//
// Public API:
//   Tensor matmul_cuda_naive(const Tensor& A, const Tensor& B)
//   Tensor matmul_cuda_tiled(const Tensor& A, const Tensor& B)
// =============================================================================

#include "matmul.hpp"
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// Tile dimension for the shared-memory kernel
// 16×16 = 256 threads per block — good occupancy on all NVIDIA architectures
static constexpr int TILE = 16;

// =============================================================================
// KERNEL 1 — Naive
// =============================================================================
//
// Thread mapping:
//   blockDim = (32, 32)
//   gridDim  = (ceil(N/32), ceil(M/32))
//   thread (tx, ty) in block (bx, by):
//     col = bx * 32 + tx   → output column
//     row = by * 32 + ty   → output row
//
// Each thread independently computes C[row][col] = Σ_k A[row][k]*B[k][col].
//
// Memory access pattern — WHY THE NAIVE KERNEL IS SLOW:
//   • Threads in the same warp (ty fixed, tx = 0..31) all read the same
//     A[row][k] → A access is replicated 32 times per warp step.
//   • B[k][col] = B[k*N + col]: col varies across the warp → 32 consecutive
//     addresses → COALESCED read → this is actually fine!
//   • But A[row][k] = A[row*K + k]: all threads read the SAME address for
//     fixed (row,k) → every read is a broadcast from L2 after the first hit.
//     This is better than expected — the real bottleneck is the K iterations
//     with no shared-memory reuse of A or B values.
//
// Peak performance: limited by L2 bandwidth and arithmetic throughput.
// For 1024×1024: ~10 GFLOPS on a modern GPU (vs. 300+ GFLOPS for cuBLAS).
__global__
void matmul_naive_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float*       __restrict__ C,
                         int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;   // guard against out-of-bounds threads

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// =============================================================================
// KERNEL 2 — Tiled Shared-Memory
// =============================================================================
//
// Thread / block mapping:
//   blockDim = (TILE, TILE) = (16, 16)  →  256 threads per block
//   gridDim  = (ceil(N/16), ceil(M/16))
//
// Algorithm (per block):
//   Each block computes one TILE×TILE sub-matrix of C.
//   It iterates over ceil(K/TILE) tiles along the shared K dimension.
//
//   For tile t:
//     As[ty][tx] = A[row][t*TILE + tx]   (one element of the A-tile)
//     Bs[ty][tx] = B[t*TILE + ty][col]   (one element of the B-tile)
//     __syncthreads()
//     acc += As[ty][0]*Bs[0][tx] + As[ty][1]*Bs[1][tx] + ... (16 MACs)
//     __syncthreads()
//
// Why this is faster:
//   Without tiling: K global memory reads per output element = M*N*K total.
//   With TILE=16:   Each A-tile element is loaded once by one thread and
//                   reused 16 times across the row (Bs). Same for B → V-tile.
//   → (TILE-1)/TILE ≈ 93.75% reduction in global memory bandwidth for large K.
//
// Shared memory per block: 2 × 16 × 16 × 4 = 2048 bytes
// On sm_80 (A100): 192 KB shared mem per SM → up to 96 concurrent blocks.
// Achieved occupancy (256 threads, 96 blocks): very high — hides latency well.
//
// The #pragma unroll directive tells nvcc to fully unroll the inner TILE loop,
// converting 16 loop iterations into 16 MACs with no branch overhead.
__global__
void matmul_tiled_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float*       __restrict__ C,
                         int M, int N, int K)
{
    // Shared memory tiles — enough for one TILE×TILE block from A and B
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;

    // Global output coordinates for this thread
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    // Number of tiles needed to cover the K dimension
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        // ── Phase 1: cooperative load ─────────────────────────────────────────
        // Each thread loads one element; out-of-bounds threads load 0 (padding).
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // ── Phase 2: sync — all threads must finish loading before computing ──
        // Without this barrier, threads might read stale shared-mem values from
        // the previous tile iteration.
        __syncthreads();

        // ── Phase 3: partial dot product over the tile ────────────────────────
        // #pragma unroll tells nvcc to unroll this fixed-trip loop → removes
        // branch prediction overhead and enables instruction-level parallelism.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // ── Phase 4: sync before loading the next tile ────────────────────────
        // Prevents fast threads from overwriting shared mem while slow threads
        // are still reading from it.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// =============================================================================
// Internal launchers — manage GPU memory and invoke kernels
// =============================================================================

// Allocate dA, dB, dC on device; copy A,B host→device; launch kernel;
// copy C device→host; free device buffers.
static Tensor run_matmul_kernel(const Tensor& A, const Tensor& B, bool tiled) {
    validate_matmul_dims(A, B);
    const int M = A.rows, K = A.cols, N = B.cols;

    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    // Upload operands
    CUDA_CHECK(cudaMemcpy(dA, A.data.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data.data(), bytesB, cudaMemcpyHostToDevice));

    // Launch
    if (tiled) {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        matmul_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    } else {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    // Wait for GPU to finish (also propagates any kernel errors)
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download result
    Tensor C(M, N);
    CUDA_CHECK(cudaMemcpy(C.data.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return C;
}

// =============================================================================
// Public API
// =============================================================================

Tensor matmul_cuda_naive(const Tensor& A, const Tensor& B) {
    return run_matmul_kernel(A, B, /*tiled=*/false);
}

Tensor matmul_cuda_tiled(const Tensor& A, const Tensor& B) {
    return run_matmul_kernel(A, B, /*tiled=*/true);
}
