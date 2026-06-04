// =============================================================================
// matmul_cpu.cpp — CPU matrix multiplication implementations
// =============================================================================
// Two implementations for educational comparison:
//   1. Naive  — straightforward triple loop, poor cache behaviour
//   2. Tiled  — blocking for L1 cache reuse, same FLOPs, much faster
// =============================================================================
#include "matmul.hpp"
#include <stdexcept>
#include <algorithm>

// L1-cache-friendly tile size.
// For 32 KB L1 cache: two 64×64 float tiles = 2 × 16 KB — fits comfortably.
static constexpr int BLOCK = 64;

// =============================================================================
// Validation
// =============================================================================
void validate_matmul_dims(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument(
            "matmul dimension mismatch:\n"
            "  A" + A.shape() + " requires A.cols == B.rows\n"
            "  B" + B.shape() + " has B.rows = " + std::to_string(B.rows) +
            " but A.cols = " + std::to_string(A.cols));
    }
}

// =============================================================================
// Naive CPU MatMul — O(M·N·K)
// =============================================================================
//
// Memory access analysis:
//   Inner loop reads A[m][k] (sequential in k — good) and B[k][n].
//   B[k][n] changes k while n is fixed → stride-N access → cache miss every
//   iteration for large matrices.  On a 1024×1024 matrix, most B accesses
//   fall outside the 32 KB L1 cache → dominated by L2/L3 latency.
//
// Useful as: correctness reference, baseline for speedup comparison.
Tensor matmul_cpu_naive(const Tensor& A, const Tensor& B) {
    validate_matmul_dims(A, B);
    const int M = A.rows, K = A.cols, N = B.cols;

    Tensor C(M, N);   // zero-initialised

    const float* a = A.data.data();
    const float* b = B.data.data();
    float*       c = C.data.data();

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // B[k][n] = b[k*N + n] — non-sequential access (stride N)
                sum += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = sum;
        }
    }
    return C;
}

// =============================================================================
// Optimised CPU MatMul — Blocked / Cache-Tiled
// =============================================================================
//
// Key idea: partition the M×N×K computation into BLOCK³ sub-problems.
//
//   for m0 in 0, BLOCK, 2*BLOCK, ...        // tile rows of A and C
//     for n0 in 0, BLOCK, 2*BLOCK, ...      // tile columns of B and C
//       for k0 in 0, BLOCK, 2*BLOCK, ...    // tile the shared dimension
//         // A-tile [m0:m0+BLOCK, k0:k0+BLOCK] and
//         // B-tile [k0:k0+BLOCK, n0:n0+BLOCK] both fit in L1 cache.
//         for m in m0..m0+BLOCK
//           for k in k0..k0+BLOCK
//             float a_mk = A[m][k]           // load A once, broadcast to N
//             for n in n0..n0+BLOCK
//               C[m][n] += a_mk * B[k][n]   // B access is sequential in n
//
// Why the inner loop ordering (m, k, n) matters:
//   Hoisting A[m][k] out of the inner loop:
//     • reduces A reads from M*N*K to M*K (one load per inner strip)
//     • inner loop now reads B sequentially: B[k*N + n0], B[k*N + n0+1], ...
//       → cache-friendly!
//
// Speedup vs naive: typically 4–10× on modern CPUs.
// Complexity: still O(M·N·K) — identical FLOPs, better memory bandwidth use.
Tensor matmul_cpu_optimized(const Tensor& A, const Tensor& B) {
    validate_matmul_dims(A, B);
    const int M = A.rows, K = A.cols, N = B.cols;

    Tensor C(M, N);   // zero-initialised

    const float* a = A.data.data();
    const float* b = B.data.data();
    float*       c = C.data.data();

    // Iterate over macro-tiles
    for (int m0 = 0; m0 < M; m0 += BLOCK) {
        int m_end = std::min(m0 + BLOCK, M);
        for (int n0 = 0; n0 < N; n0 += BLOCK) {
            int n_end = std::min(n0 + BLOCK, N);
            for (int k0 = 0; k0 < K; k0 += BLOCK) {
                int k_end = std::min(k0 + BLOCK, K);

                // Inner tile — all three windows fit in L1 cache
                for (int m = m0; m < m_end; ++m) {
                    for (int k = k0; k < k_end; ++k) {
                        float a_mk = a[m * K + k];   // load A once per (m,k)
                        for (int n = n0; n < n_end; ++n) {
                            // Sequential read of B[k][n0..n_end) → cache hits
                            c[m * N + n] += a_mk * b[k * N + n];
                        }
                    }
                }
            }
        }
    }
    return C;
}

// =============================================================================
// Dispatch — selects best available implementation at compile time
// =============================================================================
Tensor matmul(const Tensor& A, const Tensor& B) {
#ifdef USE_CUDA
    return matmul_cuda_tiled(A, B);
#else
    return matmul_cpu_optimized(A, B);
#endif
}
