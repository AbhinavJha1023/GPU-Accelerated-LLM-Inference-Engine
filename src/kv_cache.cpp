// =============================================================================
// kv_cache.cpp — KV Cache implementation
// =============================================================================
#include "kv_cache.hpp"
#include <iostream>
#include <stdexcept>

KVCache::KVCache(int num_heads, int head_dim, int max_seq_len)
    : num_heads(num_heads)
    , head_dim(head_dim)
    , kv_dim(num_heads * head_dim)
    , max_seq_len(max_seq_len)
{
    // Reserve capacity to avoid reallocations during generation
    key_store.reserve(max_seq_len);
    value_store.reserve(max_seq_len);
}

// =============================================================================
// append — store one token's K and V
// =============================================================================
// key_row and value_row must have .size() == kv_dim.
// We copy the flat float data; layout within (num_heads × head_dim) is
// enforced by the caller (MultiHeadAttention::forward_cached).
void KVCache::append(const Tensor& key_row, const Tensor& value_row) {
    if (is_full()) {
        throw std::overflow_error(
            "KVCache is full (max_seq_len = " + std::to_string(max_seq_len) + ")");
    }
    if (key_row.size() != kv_dim || value_row.size() != kv_dim) {
        throw std::invalid_argument(
            "KVCache::append: expected size " + std::to_string(kv_dim) +
            " but got key=" + std::to_string(key_row.size()) +
            " value=" + std::to_string(value_row.size()));
    }
    key_store.emplace_back(key_row.data.begin(), key_row.data.end());
    value_store.emplace_back(value_row.data.begin(), value_row.data.end());
}

// =============================================================================
// get_keys — assemble all cached keys into a [length, kv_dim] Tensor
// =============================================================================
// Time complexity: O(length * kv_dim) — a linear scan of stored data.
// Called once per decoding step, so this is acceptable.
Tensor KVCache::get_keys() const {
    int len = length();
    if (len == 0) return Tensor(0, kv_dim);

    Tensor out(len, kv_dim);
    for (int i = 0; i < len; ++i) {
        const auto& row = key_store[i];
        for (int j = 0; j < kv_dim; ++j)
            out.data[i * kv_dim + j] = row[j];
    }
    return out;
}

// =============================================================================
// get_values — assemble all cached values into a [length, kv_dim] Tensor
// =============================================================================
Tensor KVCache::get_values() const {
    int len = length();
    if (len == 0) return Tensor(0, kv_dim);

    Tensor out(len, kv_dim);
    for (int i = 0; i < len; ++i) {
        const auto& row = value_store[i];
        for (int j = 0; j < kv_dim; ++j)
            out.data[i * kv_dim + j] = row[j];
    }
    return out;
}

// =============================================================================
// clear — reset for a new sequence
// =============================================================================
void KVCache::clear() {
    key_store.clear();
    value_store.clear();
}

// =============================================================================
// print_info — diagnostic summary
// =============================================================================
void KVCache::print_info() const {
    std::cout << "KVCache: length=" << length()
              << " / max=" << max_seq_len
              << "  (num_heads=" << num_heads
              << ", head_dim=" << head_dim
              << ", kv_dim=" << kv_dim << ")\n";

    if (length() > 0) {
        // Memory usage
        size_t bytes = 2ULL * length() * kv_dim * sizeof(float);
        std::cout << "  Memory used: " << bytes / 1024.0f << " KB\n";
    }
}
