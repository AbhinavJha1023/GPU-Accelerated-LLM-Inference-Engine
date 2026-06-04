// =============================================================================
// attention_cuda.cu — CUDA Softmax and Scaled Dot-Product Attention
// =============================================================================
// Compiled ONLY when USE_CUDA=ON.
//
// This file implements:
//   1. softmax_kernel    — parallel row-wise softmax with block-reduction
//   2. scale_kernel      — in-place uniform scaling (for S / √d_k)
//   3. softmax_cuda()    — public API wrapping the kernel
//   4. sdpa_cuda()       — full SDPA using GPU kernels + device buffers
//
// Design note on efficiency:
//   sdpa_cuda keeps intermediate tensors on the GPU (device-to-device matmul)
//   to avoid expensive PCIe round-trips between steps.
// =============================================================================

#include "attention.hpp"
#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include <string>
#include <cmath>

// =============================================================================
// CUDA Matmul kernel (local copy for device-to-device use within SDPA)
// =============================================================================
// This is a copy of the tiled kernel from matmul_cuda.cu.
// Keeping it here avoids cross-translation-unit CUDA kernel linking
// complications at the cost of minor code duplication.

static constexpr int ATT_TILE = 16;

__global__
static void att_matmul_tiled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int M, int N, int K)
{
    __shared__ float As[ATT_TILE][ATT_TILE];
    __shared__ float Bs[ATT_TILE][ATT_TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * ATT_TILE + ty;
    const int col = blockIdx.x * ATT_TILE + tx;

    float acc = 0.0f;
    const int tiles = (K + ATT_TILE - 1) / ATT_TILE;

    for (int t = 0; t < tiles; ++t) {
        int a_col = t * ATT_TILE + tx;
        int b_row = t * ATT_TILE + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < ATT_TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// =============================================================================
// KERNEL: scale_kernel
// =============================================================================
// Multiply every element of data[0..n) by 'factor'.
// Launched with ceil(n/256) blocks of 256 threads.
//
// Used to apply the 1/√d_k scaling factor to the raw attention scores.
__global__
static void scale_kernel(float* __restrict__ data, float factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

// =============================================================================
// KERNEL: softmax_kernel
// =============================================================================
//
// One block per row.  Block size = 256 threads (or blockDim.x, set by caller).
//
// Algorithm (parallel reduction within a block):
//   Phase 1 — find row maximum (parallel reduction over threads)
//   Phase 2 — compute exp(x[i] - max) and sum (another reduction)
//   Phase 3 — normalise: out[i] /= sum
//
// Shared memory layout:
//   sdata[0..blockDim.x): holds thread-local max or sum during reductions.
//
// Why parallel reduction?
//   A single thread scanning all cols would take O(cols) steps sequentially.
//   With B threads, each thread handles ceil(cols/B) elements, then a
//   log2(B)-step tree reduction finds the global max/sum → O(log B) sync steps.
//
// Occupancy note:
//   shared mem per block = 256 × 4 = 1024 bytes — tiny, no occupancy pressure.
//   For rows with > 1024 elements, the loop "col += blockDim.x" handles them.
__global__
static void softmax_kernel(float*       __restrict__ out,
                           const float* __restrict__ in,
                           int rows, int cols)
{
    extern __shared__ float sdata[];   // blockDim.x floats

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = in  + row * cols;
    float*       row_out = out + row * cols;

    // ── Phase 1: find row max ──────────────────────────────────────────────────
    float local_max = -1e38f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        local_max = fmaxf(local_max, row_in[c]);

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Tree reduction: stride halves each iteration
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // ── Phase 2: exp and partial sum ──────────────────────────────────────────
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = expf(row_in[c] - row_max);
        row_out[c] = e;
        local_sum += e;
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // ── Phase 3: normalise ────────────────────────────────────────────────────
    float inv_sum = 1.0f / row_sum;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        row_out[c] *= inv_sum;
}

// =============================================================================
// softmax_cuda — public API
// =============================================================================
Tensor softmax_cuda(const Tensor& x) {
    const int rows = x.rows, cols = x.cols;
    const size_t bytes = (size_t)rows * cols * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, x.data.data(), bytes, cudaMemcpyHostToDevice));

    // One block per row; 256 threads per block (handles rows of any length)
    const int threads = 256;
    const size_t smem = threads * sizeof(float);
    softmax_kernel<<<rows, threads, smem>>>(d_out, d_in, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Tensor out(rows, cols);
    CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

// =============================================================================
// sdpa_cuda — Scaled Dot-Product Attention on GPU
// =============================================================================
//
// All intermediate tensors stay on the GPU to avoid PCIe round-trips:
//
//   Host  → GPU : Q, K, V                   (3 uploads)
//   GPU   → GPU : Kᵀ = transpose(K)         (kernel)
//   GPU   → GPU : S  = Q · Kᵀ               (tiled matmul kernel)
//   GPU   → GPU : S  *= 1/√d_k              (scale kernel)
//   GPU   → GPU : A  = softmax(S)            (softmax kernel)
//   GPU   → GPU : Out = A · V               (tiled matmul kernel)
//   GPU   → Host: Out                        (1 download)
//
// This pattern minimises bandwidth: total PCIe traffic = 3 uploads + 1 download.
// (Compare with the CPU version + data on host: 6 copies for the same pipeline.)
Tensor sdpa_cuda(const Tensor& Q, const Tensor& K, const Tensor& V) {
    if (Q.cols != K.cols)
        throw std::invalid_argument("sdpa_cuda: Q.cols != K.cols");
    if (K.rows != V.rows)
        throw std::invalid_argument("sdpa_cuda: K.rows != V.rows");

    const int seq_q = Q.rows, d_k = Q.cols;
    const int seq_k = K.rows, d_v = V.cols;

    const size_t bytesQ  = (size_t)seq_q * d_k   * sizeof(float);
    const size_t bytesK  = (size_t)seq_k * d_k   * sizeof(float);
    const size_t bytesV  = (size_t)seq_k * d_v   * sizeof(float);
    const size_t bytesKt = (size_t)d_k   * seq_k * sizeof(float);  // Kᵀ
    const size_t bytesS  = (size_t)seq_q * seq_k * sizeof(float);
    const size_t bytesO  = (size_t)seq_q * d_v   * sizeof(float);

    float *dQ=nullptr, *dK=nullptr, *dV=nullptr,
          *dKt=nullptr, *dS=nullptr, *dO=nullptr;

    CUDA_CHECK(cudaMalloc(&dQ,  bytesQ));
    CUDA_CHECK(cudaMalloc(&dK,  bytesK));
    CUDA_CHECK(cudaMalloc(&dV,  bytesV));
    CUDA_CHECK(cudaMalloc(&dKt, bytesKt));
    CUDA_CHECK(cudaMalloc(&dS,  bytesS));
    CUDA_CHECK(cudaMalloc(&dO,  bytesO));

    // Upload Q, K, V
    CUDA_CHECK(cudaMemcpy(dQ, Q.data.data(), bytesQ, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data.data(), bytesK, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data.data(), bytesV, cudaMemcpyHostToDevice));

    // ── Step 1: Kᵀ = K.transpose()  [d_k × seq_k] ────────────────────────────
    // Compute the transpose on CPU and re-upload (simpler than a CUDA transpose
    // kernel for this educational project; production would use a tiled transpose)
    Tensor Kt = K.transpose();
    CUDA_CHECK(cudaMemcpy(dKt, Kt.data.data(), bytesKt, cudaMemcpyHostToDevice));

    // ── Step 2: S = Q · Kᵀ   [seq_q × seq_k] ────────────────────────────────
    {
        dim3 block(ATT_TILE, ATT_TILE);
        dim3 grid((seq_k + ATT_TILE - 1) / ATT_TILE,
                  (seq_q + ATT_TILE - 1) / ATT_TILE);
        att_matmul_tiled<<<grid, block>>>(dQ, dKt, dS, seq_q, seq_k, d_k);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Step 3: S /= √d_k  ────────────────────────────────────────────────────
    {
        float sc = 1.0f / std::sqrt(static_cast<float>(d_k));
        int   n  = seq_q * seq_k;
        scale_kernel<<<(n + 255) / 256, 256>>>(dS, sc, n);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Step 4: A = softmax(S)   [seq_q × seq_k] ─────────────────────────────
    // Reuse dS as output (in-place softmax — need a separate output buffer
    // because softmax_kernel reads from d_in and writes to d_out)
    float* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesS));
    {
        const int threads = 256;
        softmax_kernel<<<seq_q, threads, threads * sizeof(float)>>>(
            dA, dS, seq_q, seq_k);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Step 5: Out = A · V   [seq_q × d_v] ─────────────────────────────────
    {
        dim3 block(ATT_TILE, ATT_TILE);
        dim3 grid((d_v   + ATT_TILE - 1) / ATT_TILE,
                  (seq_q + ATT_TILE - 1) / ATT_TILE);
        att_matmul_tiled<<<grid, block>>>(dA, dV, dO, seq_q, d_v, seq_k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Download result
    Tensor out(seq_q, d_v);
    CUDA_CHECK(cudaMemcpy(out.data.data(), dO, bytesO, cudaMemcpyDeviceToHost));

    cudaFree(dQ); cudaFree(dK);  cudaFree(dV);
    cudaFree(dKt); cudaFree(dS); cudaFree(dA); cudaFree(dO);

    return out;
}
