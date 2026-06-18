#pragma once
// =============================================================================
// mini_transformer.hpp — Single-layer autoregressive transformer
// =============================================================================
// Architecture:
//   embedding   [vocab_size, d_model]   (token lookup + sinusoidal positional)
//   attention   MultiHeadAttention(d_model, num_heads)
//   output_proj [d_model, vocab_size]   (hidden -> vocab logits)
//
// Two inference modes:
//   • generate()        — single sequence, greedy argmax decoding with KV cache
//   • forward_step_batch — one decode step for B requests (continuous batching)
//
// Weights are random (educational): the model demonstrates the inference
// pipeline and optimizations, not trained text generation.
// =============================================================================

#include "tensor.hpp"
#include "attention.hpp"
#include "kv_cache.hpp"
#include <vector>

class MiniTransformer {
public:
    int vocab_size, d_model, num_heads, max_seq_len;
    Tensor embedding;          // [vocab_size, d_model]
    MultiHeadAttention mha;    // attention layer
    Tensor output_proj;        // [d_model, vocab_size]
    KVCache cache;             // cache used by the single-sequence generate()

    MiniTransformer(int vocab_size, int d_model, int num_heads, int max_seq_len);

    // Sinusoidal positional encoding for one position -> [1, d_model]
    Tensor positional_encoding(int pos) const;

    // Token id + position -> [1, d_model] (embedding lookup + positional)
    Tensor embed(int token_id, int pos) const;

    // Hidden [n, d_model] -> logits [n, vocab_size]
    Tensor get_logits(const Tensor& hidden) const;

    // Greedy argmax over a single logits row [1, vocab_size] -> token id
    int argmax_row(const Tensor& logits, int row = 0) const;

    // Single-sequence greedy generation (prefill + decode) using `cache`.
    // Returns the full token sequence (prompt + generated).
    std::vector<int> generate(const std::vector<int>& prompt_ids,
                              int max_new_tokens = 20);

    // ── Continuous-batching primitive ─────────────────────────────────────────
    // One decode step for B requests at once.
    //   tokens    : B current token ids (the last token of each request)
    //   positions : B sequence positions (for positional encoding)
    //   caches    : B per-request KV caches
    //   return    : next token id for each request (greedy argmax)
    std::vector<int> forward_step_batch(const std::vector<int>& tokens,
                                        const std::vector<int>& positions,
                                        std::vector<KVCache*>& caches) const;
};
