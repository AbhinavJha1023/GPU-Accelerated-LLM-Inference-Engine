#pragma once
// =============================================================================
// attention.hpp — Softmax, Scaled Dot-Product Attention, Multi-Head Attention
// =============================================================================
//
// SCALED DOT-PRODUCT ATTENTION (SDPA)
// ─────────────────────────────────────
//   Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
//
//   Q : [seq_q, d_k]    — query matrix
//   K : [seq_k, d_k]    — key   matrix  (seq_k can differ from seq_q in cached mode)
//   V : [seq_k, d_v]    — value matrix
//   →   [seq_q, d_v]    — output
//
//   Step 1: S  = Q · Kᵀ              [seq_q, seq_k]  — raw scores
//   Step 2: S  = S / √d_k            [seq_q, seq_k]  — scale (prevents vanishing grad)
//   Step 3: A  = softmax_rows(S)     [seq_q, seq_k]  — attention weights (sum=1 per row)
//   Step 4: O  = A · V               [seq_q, d_v]    — weighted sum of values
//
// MULTI-HEAD ATTENTION (MHA)
// ───────────────────────────
//   Split d_model into num_heads independent heads of size head_dim.
//   Each head attends to a different projection of the input.
//   Outputs are concatenated and projected back to d_model.
//
//   head_dim = d_model / num_heads
//
//   MHA(X) = Concat(head_1,...,head_h) · W_O
//   head_i  = SDPA( X·W_Q_i, X·W_K_i, X·W_V_i )
//
// =============================================================================

#include "tensor.hpp"
#include "kv_cache.hpp"
#include <vector>

// =============================================================================
// Softmax (row-wise, numerically stable)
// =============================================================================
// For each row r:  out[r][c] = exp(in[r][c] - max(in[r])) / sum_c(exp(...))
// Subtracting the row max prevents exp() overflow (no accuracy loss).

Tensor softmax_cpu(const Tensor& x);   // CPU implementation

#ifdef USE_CUDA
Tensor softmax_cuda(const Tensor& x);  // GPU implementation
#endif

Tensor softmax(const Tensor& x);       // dispatch: GPU if available, else CPU

// =============================================================================
// Scaled Dot-Product Attention
// =============================================================================

Tensor sdpa_cpu(const Tensor& Q, const Tensor& K, const Tensor& V);

#ifdef USE_CUDA
Tensor sdpa_cuda(const Tensor& Q, const Tensor& K, const Tensor& V);
#endif

Tensor sdpa(const Tensor& Q, const Tensor& K, const Tensor& V);  // dispatch

// =============================================================================
// Multi-Head Attention
// =============================================================================
class MultiHeadAttention {
public:
    int d_model;    // total embedding dimension
    int num_heads;  // number of independent attention heads
    int head_dim;   // d_model / num_heads

    // Projection weights — all [d_model × d_model]
    // Initialised with Xavier scaling: uniform[-1/√d_model, +1/√d_model]
    Tensor W_Q;   // query projection
    Tensor W_K;   // key   projection
    Tensor W_V;   // value projection
    Tensor W_O;   // output projection (combines head outputs)

    // ── Constructor ───────────────────────────────────────────────────────────
    // Throws if d_model % num_heads != 0.
    MultiHeadAttention(int d_model, int num_heads);

    // ── full-sequence forward (prefill mode) ──────────────────────────────────
    // input  : [seq_len, d_model]
    // return : [seq_len, d_model]
    //
    // Internally:
    //   Q = input · W_Q   [seq_len, d_model]
    //   K = input · W_K   [seq_len, d_model]
    //   V = input · W_V   [seq_len, d_model]
    //   For each head h:
    //     Q_h = Q[:, h*head_dim : (h+1)*head_dim]   [seq_len, head_dim]
    //     K_h = K[:, h*head_dim : (h+1)*head_dim]   [seq_len, head_dim]
    //     V_h = V[:, h*head_dim : (h+1)*head_dim]   [seq_len, head_dim]
    //     head_h = SDPA(Q_h, K_h, V_h)              [seq_len, head_dim]
    //   concat = Concat(head_0,...,head_{h-1})       [seq_len, d_model]
    //   output = concat · W_O                        [seq_len, d_model]
    Tensor forward(const Tensor& input) const;

    // ── single-token forward with KV cache (generation mode) ─────────────────
    // token_emb : [1, d_model]   — embedding of the NEW token only
    // cache     : KVCache        — holds K/V for all PREVIOUS tokens
    //
    // Steps:
    //   1. q = token_emb · W_Q   [1, d_model]
    //   2. k = token_emb · W_K   [1, d_model]  → append to cache
    //   3. v = token_emb · W_V   [1, d_model]  → append to cache
    //   4. K_all = cache.get_keys()    [t, d_model]
    //   5. V_all = cache.get_values()  [t, d_model]
    //   6. Per head: SDPA(q_h, K_all_h, V_all_h)
    //   7. Concat + W_O
    // return : [1, d_model]
    Tensor forward_cached(const Tensor& token_emb, KVCache& cache) const;

private:
    // Matrix multiply input [seq, d_model] by weight [d_model, d_model]
    Tensor project(const Tensor& input, const Tensor& W) const;

    // Extract columns [h*head_dim, (h+1)*head_dim) from x [seq, d_model]
    // Returns [seq, head_dim]
    Tensor extract_head(const Tensor& x, int h) const;

    // Assemble per-head outputs into [seq, d_model]
    Tensor merge_heads(const std::vector<Tensor>& head_outs, int seq_len) const;
};
