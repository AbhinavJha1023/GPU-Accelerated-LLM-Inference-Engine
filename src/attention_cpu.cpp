// =============================================================================
// attention_cpu.cpp — Softmax, SDPA, and MultiHeadAttention (CPU)
// =============================================================================
#include "attention.hpp"
#include "matmul.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>

// =============================================================================
// softmax_cpu — numerically stable row-wise softmax
// =============================================================================
// For each row r:
//   1. Find max_r = max(x[r][*])
//   2. Shift: y[r][c] = x[r][c] - max_r           (prevents overflow in exp)
//   3. Exponentiate: e[r][c] = exp(y[r][c])
//   4. Normalise: out[r][c] = e[r][c] / Σ_c e[r][c]
//
// Why subtract the max?
//   exp(x) overflows float for x > ~88. In attention, logits QKᵀ/√d can be
//   large for sharp attention distributions. Subtracting the row max is
//   mathematically equivalent (the constant cancels in the ratio) but
//   ensures all arguments to exp() are ≤ 0, preventing NaN/Inf.
//
// Complexity: O(rows × cols) — two passes per row (max, then exp+sum).
Tensor softmax_cpu(const Tensor& x) {
    Tensor out(x.rows, x.cols);

    for (int r = 0; r < x.rows; ++r) {
        // Pass 1: find row max
        float row_max = x.data[r * x.cols];
        for (int c = 1; c < x.cols; ++c)
            row_max = std::max(row_max, x.data[r * x.cols + c]);

        // Pass 2: exp(x - max) and sum
        float sum = 0.0f;
        for (int c = 0; c < x.cols; ++c) {
            float e = std::exp(x.data[r * x.cols + c] - row_max);
            out.data[r * x.cols + c] = e;
            sum += e;
        }

        // Pass 3: normalise
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < x.cols; ++c)
            out.data[r * x.cols + c] *= inv_sum;
    }
    return out;
}

// Dispatch
Tensor softmax(const Tensor& x) {
#ifdef USE_CUDA
    return softmax_cuda(x);
#else
    return softmax_cpu(x);
#endif
}

// =============================================================================
// sdpa_cpu — Scaled Dot-Product Attention
// =============================================================================
//
//   Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
//
// Step-by-step with shapes:
//   Q   : [seq_q, d_k]
//   K   : [seq_k, d_k]
//   V   : [seq_k, d_v]
//
//   Step 1:  Kᵀ  : [d_k,  seq_k]            (transpose K)
//   Step 2:  S   = Q · Kᵀ   : [seq_q, seq_k] (raw dot-product scores)
//   Step 3:  S   = S / √d_k  : [seq_q, seq_k] (scale to stabilise gradients)
//   Step 4:  A   = softmax(S) : [seq_q, seq_k] (attention weights, Σ=1 per row)
//   Step 5:  Out = A · V     : [seq_q, d_v]   (weighted sum of value vectors)
//
// The scaling factor 1/√d_k was introduced in "Attention is All You Need"
// (Vaswani et al., 2017) to prevent the dot products from growing large in
// magnitude with high d_k, which would push softmax into regions of very
// small gradients.
Tensor sdpa_cpu(const Tensor& Q, const Tensor& K, const Tensor& V) {
    if (Q.cols != K.cols)
        throw std::invalid_argument(
            "sdpa: Q.cols (" + std::to_string(Q.cols) +
            ") must equal K.cols (" + std::to_string(K.cols) + ")");
    if (K.rows != V.rows)
        throw std::invalid_argument(
            "sdpa: K.rows (" + std::to_string(K.rows) +
            ") must equal V.rows (" + std::to_string(V.rows) + ")");

    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols));

    // Step 1 + 2: S = Q · Kᵀ
    Tensor Kt = K.transpose();                    // [d_k, seq_k]
    Tensor S  = matmul_cpu_optimized(Q, Kt);     // [seq_q, seq_k]

    // Step 3: scale
    for (float& v : S.data) v *= scale;

    // Step 4: softmax
    Tensor A = softmax_cpu(S);                   // [seq_q, seq_k]

    // Step 5: output = A · V
    Tensor out = matmul_cpu_optimized(A, V);     // [seq_q, d_v]
    return out;
}

// Dispatch
Tensor sdpa(const Tensor& Q, const Tensor& K, const Tensor& V) {
#ifdef USE_CUDA
    return sdpa_cuda(Q, K, V);
#else
    return sdpa_cpu(Q, K, V);
#endif
}

// =============================================================================
// MultiHeadAttention — constructor
// =============================================================================
// Weight matrices are Xavier-initialised: U[-1/√d_model, +1/√d_model].
// In a real training system these would be loaded from a checkpoint.
MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model(d_model), num_heads(num_heads), head_dim(d_model / num_heads)
    , W_Q(d_model, d_model), W_K(d_model, d_model)
    , W_V(d_model, d_model), W_O(d_model, d_model)
{
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "MultiHeadAttention: d_model (" + std::to_string(d_model) +
            ") must be divisible by num_heads (" + std::to_string(num_heads) + ")");

    float bound = 1.0f / std::sqrt(static_cast<float>(d_model));
    W_Q.randomize(-bound, bound);
    W_K.randomize(-bound, bound);
    W_V.randomize(-bound, bound);
    W_O.randomize(-bound, bound);
}

// =============================================================================
// project — linear projection: [seq, d_model] × [d_model, out] → [seq, out]
// =============================================================================
Tensor MultiHeadAttention::project(const Tensor& input, const Tensor& W) const {
    return matmul_cpu_optimized(input, W);
}

// =============================================================================
// extract_head — slice columns [h*head_dim, (h+1)*head_dim) from x
// =============================================================================
// x     : [seq, d_model]
// return: [seq, head_dim]
Tensor MultiHeadAttention::extract_head(const Tensor& x, int h) const {
    int seq   = x.rows;
    int start = h * head_dim;
    Tensor out(seq, head_dim);
    for (int r = 0; r < seq; ++r)
        for (int c = 0; c < head_dim; ++c)
            out.data[r * head_dim + c] = x.data[r * d_model + start + c];
    return out;
}

// =============================================================================
// merge_heads — concatenate head_outs[0..num_heads-1] along the last axis
// =============================================================================
// head_outs[h]: [seq_len, head_dim]
// return      : [seq_len, d_model]
Tensor MultiHeadAttention::merge_heads(
    const std::vector<Tensor>& head_outs, int seq_len) const
{
    Tensor out(seq_len, d_model);
    for (int h = 0; h < num_heads; ++h) {
        int start = h * head_dim;
        for (int r = 0; r < seq_len; ++r)
            for (int c = 0; c < head_dim; ++c)
                out.data[r * d_model + start + c] = head_outs[h].data[r * head_dim + c];
    }
    return out;
}

// =============================================================================
// forward — full-sequence (prefill) mode
// =============================================================================
//
// Shape trace (d_model=512, num_heads=8, head_dim=64, seq_len=16):
//   input         : [16, 512]
//   Q = input·W_Q : [16, 512]
//   K = input·W_K : [16, 512]
//   V = input·W_V : [16, 512]
//   For h in 0..7:
//     Q_h = Q[:,h*64:(h+1)*64] : [16, 64]
//     K_h = K[:,h*64:(h+1)*64] : [16, 64]
//     V_h = V[:,h*64:(h+1)*64] : [16, 64]
//     head_h = SDPA(Q_h, K_h, V_h) : [16, 64]
//   concat = [head_0|...|head_7]   : [16, 512]
//   output = concat · W_O          : [16, 512]
Tensor MultiHeadAttention::forward(const Tensor& input) const {
    int seq_len = input.rows;

    // Project all tokens to Q, K, V
    Tensor Q = project(input, W_Q);   // [seq_len, d_model]
    Tensor K = project(input, W_K);   // [seq_len, d_model]
    Tensor V = project(input, W_V);   // [seq_len, d_model]

    // Per-head attention
    std::vector<Tensor> head_outs(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        Tensor Q_h = extract_head(Q, h);   // [seq_len, head_dim]
        Tensor K_h = extract_head(K, h);   // [seq_len, head_dim]
        Tensor V_h = extract_head(V, h);   // [seq_len, head_dim]
        head_outs[h] = sdpa_cpu(Q_h, K_h, V_h);   // [seq_len, head_dim]
    }

    // Concatenate heads and project
    Tensor concat = merge_heads(head_outs, seq_len);   // [seq_len, d_model]
    return project(concat, W_O);                        // [seq_len, d_model]
}

// =============================================================================
// forward_cached — single-token (generation) mode with KV cache
// =============================================================================
//
// At generation step t (cache already holds tokens 0..t-1):
//   token_emb : [1, d_model]
//   q = token_emb · W_Q : [1, d_model]
//   k = token_emb · W_K : [1, d_model]  → appended to cache → length becomes t
//   v = token_emb · W_V : [1, d_model]  → appended to cache
//   K_all = cache.get_keys()   : [t, d_model]
//   V_all = cache.get_values() : [t, d_model]
//   For each head h:
//     q_h = q[:,h*hd:(h+1)*hd] : [1, head_dim]
//     K_h = K_all[:,h*hd:...]  : [t, head_dim]
//     V_h = V_all[:,h*hd:...]  : [t, head_dim]
//     head_h = SDPA(q_h, K_h, V_h) : [1, head_dim]
//   return (concat · W_O) : [1, d_model]
//
// KV cache complexity savings:
//   Without cache: recompute K,V for all t previous tokens → O(t) matmuls/step
//   With cache: only 2 matmuls for the new token's K,V → O(1) matmuls/step
//   Total over n steps: O(n) vs O(n²)
Tensor MultiHeadAttention::forward_cached(const Tensor& token_emb,
                                          KVCache& cache) const
{
    if (token_emb.cols != d_model)
        throw std::invalid_argument("forward_cached: token_emb.cols must equal d_model");

    // Project the single new token
    Tensor q = project(token_emb, W_Q);   // [1, d_model]
    Tensor k = project(token_emb, W_K);   // [1, d_model]
    Tensor v = project(token_emb, W_V);   // [1, d_model]

    // Append this token's K, V to the cache
    cache.append(k, v);

    // Retrieve all cached K, V (includes the just-appended token)
    Tensor K_all = cache.get_keys();    // [cache_len, d_model]
    Tensor V_all = cache.get_values();  // [cache_len, d_model]

    // Per-head attention: query is [1, head_dim], context is [cache_len, head_dim]
    std::vector<Tensor> head_outs(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        Tensor q_h   = extract_head(q,     h);   // [1,          head_dim]
        Tensor K_h   = extract_head(K_all, h);   // [cache_len,  head_dim]
        Tensor V_h   = extract_head(V_all, h);   // [cache_len,  head_dim]
        head_outs[h] = sdpa_cpu(q_h, K_h, V_h); // [1,          head_dim]
    }

    Tensor concat = merge_heads(head_outs, 1);   // [1, d_model]
    return project(concat, W_O);                  // [1, d_model]
}

// =============================================================================
// forward_cached_batch — decode one new token for B requests at once
// =============================================================================
// Shape trace (B requests, d_model):
//   batch_emb : [B, d_model]
//   Q = batch_emb · W_Q : [B, d_model]   ← ONE matmul for all requests
//   K = batch_emb · W_K : [B, d_model]   ← ONE matmul
//   V = batch_emb · W_V : [B, d_model]   ← ONE matmul
//   For each request b:
//     append K[b], V[b] to caches[b]
//     K_all_b = caches[b].get_keys()   : [ctx_b, d_model]
//     V_all_b = caches[b].get_values() : [ctx_b, d_model]
//     For each head h:
//       q_bh   = Q[b, head h slice]            : [1, head_dim]
//       head   = SDPA(q_bh, K_all_b_h, V_all_b_h) : [1, head_dim]
//     hidden[b] = concat of heads             : [1, d_model]
//   output = hidden · W_O : [B, d_model]   ← ONE matmul
Tensor MultiHeadAttention::forward_cached_batch(const Tensor& batch_emb,
                                                std::vector<KVCache*>& caches) const
{
    const int B = batch_emb.rows;
    if (batch_emb.cols != d_model)
        throw std::invalid_argument("forward_cached_batch: cols must equal d_model");
    if (static_cast<int>(caches.size()) != B)
        throw std::invalid_argument("forward_cached_batch: caches.size() must equal batch rows");

    // Batched projections — one matmul each, covering every request.
    Tensor Q = project(batch_emb, W_Q);   // [B, d_model]
    Tensor K = project(batch_emb, W_K);   // [B, d_model]
    Tensor V = project(batch_emb, W_V);   // [B, d_model]

    Tensor hidden(B, d_model);

    for (int b = 0; b < B; ++b) {
        // Extract this request's single-row q/k/v projections
        Tensor q_row(1, d_model), k_row(1, d_model), v_row(1, d_model);
        for (int c = 0; c < d_model; ++c) {
            q_row.data[c] = Q.data[b * d_model + c];
            k_row.data[c] = K.data[b * d_model + c];
            v_row.data[c] = V.data[b * d_model + c];
        }

        // Append new K,V to this request's own cache, then read full context
        caches[b]->append(k_row, v_row);
        Tensor K_all = caches[b]->get_keys();    // [ctx_b, d_model]
        Tensor V_all = caches[b]->get_values();  // [ctx_b, d_model]

        // Per-head attention for this request
        std::vector<Tensor> head_outs(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            Tensor q_h   = extract_head(q_row, h);  // [1,      head_dim]
            Tensor K_h   = extract_head(K_all, h);  // [ctx_b,  head_dim]
            Tensor V_h   = extract_head(V_all, h);  // [ctx_b,  head_dim]
            head_outs[h] = sdpa_cpu(q_h, K_h, V_h); // [1,      head_dim]
        }
        Tensor concat = merge_heads(head_outs, 1);  // [1, d_model]

        // Write this request's hidden row into the batch
        for (int c = 0; c < d_model; ++c)
            hidden.data[b * d_model + c] = concat.data[c];
    }

    // Batched output projection — one matmul for all requests
    return project(hidden, W_O);   // [B, d_model]
}
