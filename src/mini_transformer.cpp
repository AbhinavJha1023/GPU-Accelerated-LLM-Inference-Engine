// =============================================================================
// mini_transformer.cpp — MiniTransformer implementation
// =============================================================================
#include "mini_transformer.hpp"
#include "matmul.hpp"
#include "tokenizer.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

MiniTransformer::MiniTransformer(int vocab_size, int d_model,
                                 int num_heads, int max_seq_len)
    : vocab_size(vocab_size), d_model(d_model)
    , num_heads(num_heads), max_seq_len(max_seq_len)
    , embedding(vocab_size, d_model)
    , mha(d_model, num_heads)
    , output_proj(d_model, vocab_size)
    , cache(num_heads, d_model / num_heads, max_seq_len)
{
    float emb_scale = 1.0f / std::sqrt(static_cast<float>(d_model));
    embedding.randomize(-emb_scale, emb_scale);
    output_proj.randomize(-emb_scale, emb_scale);
}

// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
Tensor MiniTransformer::positional_encoding(int pos) const {
    Tensor pe(1, d_model);
    for (int i = 0; i < d_model / 2; ++i) {
        float angle = static_cast<float>(pos) /
                      std::pow(10000.0f, 2.0f * i / d_model);
        pe.data[2 * i]     = std::sin(angle);
        pe.data[2 * i + 1] = std::cos(angle);
    }
    return pe;
}

Tensor MiniTransformer::embed(int token_id, int pos) const {
    if (token_id < 0 || token_id >= vocab_size)
        throw std::out_of_range("embed: token_id out of range");
    Tensor emb(1, d_model);
    for (int c = 0; c < d_model; ++c)
        emb.data[c] = embedding.data[token_id * d_model + c];
    Tensor pe = positional_encoding(pos);
    for (int c = 0; c < d_model; ++c)
        emb.data[c] += pe.data[c];
    return emb;
}

Tensor MiniTransformer::get_logits(const Tensor& hidden) const {
    return matmul_cpu_optimized(hidden, output_proj);
}

int MiniTransformer::argmax_row(const Tensor& logits, int row) const {
    int best = 0;
    float best_val = logits.data[row * logits.cols];
    for (int i = 1; i < logits.cols; ++i) {
        float v = logits.data[row * logits.cols + i];
        if (v > best_val) { best_val = v; best = i; }
    }
    return best;
}

std::vector<int> MiniTransformer::generate(const std::vector<int>& prompt_ids,
                                           int max_new_tokens) {
    cache.clear();
    std::vector<int> tokens = prompt_ids;

    // Prefill: process all prompt tokens, building the KV cache
    for (int pos = 0; pos < static_cast<int>(prompt_ids.size()); ++pos) {
        Tensor x = embed(prompt_ids[pos], pos);
        mha.forward_cached(x, cache);   // output discarded; cache is the goal
    }

    // Decode: one token at a time
    for (int step = 0; step < max_new_tokens; ++step) {
        int cur_pos = static_cast<int>(tokens.size()) - 1;
        int cur_tok = tokens.back();

        Tensor x      = embed(cur_tok, cur_pos);
        Tensor hidden = mha.forward_cached(x, cache);  // [1, d_model]
        Tensor logits = get_logits(hidden);             // [1, vocab_size]
        int    next   = argmax_row(logits);

        tokens.push_back(next);
        if (next == Tokenizer::EOS_ID || cache.is_full()) break;
    }
    return tokens;
}

// =============================================================================
// forward_step_batch — one decode step for B requests
// =============================================================================
std::vector<int> MiniTransformer::forward_step_batch(
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    std::vector<KVCache*>& caches) const
{
    const int B = static_cast<int>(tokens.size());
    if (static_cast<int>(positions.size()) != B ||
        static_cast<int>(caches.size())    != B)
        throw std::invalid_argument("forward_step_batch: size mismatch");

    // Build the [B, d_model] embedding matrix (one row per request)
    Tensor batch_emb(B, d_model);
    for (int b = 0; b < B; ++b) {
        Tensor e = embed(tokens[b], positions[b]);   // [1, d_model]
        for (int c = 0; c < d_model; ++c)
            batch_emb.data[b * d_model + c] = e.data[c];
    }

    // Batched attention with per-request caches -> [B, d_model]
    Tensor hidden = mha.forward_cached_batch(batch_emb, caches);

    // Batched logits -> [B, vocab_size], then greedy argmax per request
    Tensor logits = get_logits(hidden);
    std::vector<int> next(B);
    for (int b = 0; b < B; ++b)
        next[b] = argmax_row(logits, b);
    return next;
}
