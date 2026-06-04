#pragma once
// =============================================================================
// kv_cache.hpp — Key-Value Cache for autoregressive transformer inference
// =============================================================================
// WHY KV CACHE EXISTS
// ────────────────────
// At generation step t, scaled-dot-product attention needs:
//   Q[t]  : query for the current token           shape [1,  d]
//   K[0…t]: keys for all tokens seen so far       shape [t+1, d]
//   V[0…t]: values for all tokens seen so far     shape [t+1, d]
//
// Without cache: recompute K and V for every previous token at every step.
//   Complexity: O(n²) matmul operations total.
//
// With cache: store K[i] and V[i] as they are produced; at step t, only
//   compute K[t] and V[t] and append to the cache.
//   Attention then uses (cached_K, cached_V).
//   Complexity: O(n) matmul operations + O(n) linear attention per step.
//
// Memory cost: O(num_layers × max_seq_len × 2 × d_model × sizeof(float))
//   Example: GPT-3 — 96 layers, 4096 d_model, 2048 seq → ~3 GB for KV cache.
// =============================================================================

#include "tensor.hpp"
#include <vector>
#include <stdexcept>

class KVCache {
public:
    int num_heads;     // number of attention heads
    int head_dim;      // dimension of each head  (d_model / num_heads)
    int kv_dim;        // = num_heads * head_dim  (flat dimension stored per token)
    int max_seq_len;   // hard capacity limit

    // Each element is one token's flattened key or value vector.
    // key_store[i]   : float[kv_dim]   — key   for token position i
    // value_store[i] : float[kv_dim]   — value for token position i
    std::vector<std::vector<float>> key_store;
    std::vector<std::vector<float>> value_store;

    // ── Constructor ───────────────────────────────────────────────────────────
    // num_heads  : number of attention heads
    // head_dim   : dimensionality per head
    // max_seq_len: maximum tokens this cache can hold (default 2048)
    KVCache(int num_heads, int head_dim, int max_seq_len = 2048);

    // ── append ────────────────────────────────────────────────────────────────
    // Appends one token's key and value vectors to the cache.
    //
    // key_row  : Tensor of shape [1, kv_dim]  (or [kv_dim, 1] — must have
    //            size() == kv_dim regardless of logical shape)
    // value_row: same constraint as key_row
    //
    // Throws std::overflow_error if the cache is full.
    void append(const Tensor& key_row, const Tensor& value_row);

    // ── get_keys / get_values ─────────────────────────────────────────────────
    // Returns all stored keys/values as a single Tensor [length(), kv_dim].
    // The returned tensor is built on every call — cache it if you call it
    // frequently within one decoding step.
    Tensor get_keys()   const;
    Tensor get_values() const;

    // ── Diagnostics ───────────────────────────────────────────────────────────
    int  length()  const { return static_cast<int>(key_store.size()); }
    bool is_full() const { return length() >= max_seq_len; }
    bool is_empty() const { return key_store.empty(); }

    // Reset the cache (start a new sequence)
    void clear();

    // Print a summary of the cache state
    void print_info() const;
};
