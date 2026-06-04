// =============================================================================
// benchmark_attention.cpp — Attention mechanism performance benchmark
// =============================================================================
// Standalone executable — does NOT use GoogleTest.
// Run:  ./benchmark_attention
//
// Benchmarks:
//   1. SDPA CPU vs GPU across different sequence lengths
//   2. MHA full-sequence forward
//   3. MHA KV-cache single-token decode (generation speed)
//   4. KV cache growth profile
// =============================================================================

#include "tensor.hpp"
#include "attention.hpp"
#include "kv_cache.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

static constexpr int WARMUP = 2;
static constexpr int RUNS   = 5;

// =============================================================================
// SDPA benchmark across sequence lengths
// =============================================================================
static void bench_sdpa() {
    std::cout << "\n  ── SDPA Benchmark ──────────────────────────────────────\n";
    std::cout << "  d_k = d_v = 64\n\n";

    std::vector<int> seq_lens = {8, 16, 32, 64, 128, 256};
    const int d = 64;

    std::cout << std::left  << std::setw(12) << "seq_len"
              << std::right << std::setw(14) << "CPU (ms)"
#ifdef USE_CUDA
              << std::right << std::setw(14) << "GPU (ms)"
              << std::right << std::setw(10) << "Speedup"
#endif
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int seq : seq_lens) {
        Tensor Q(seq, d), K(seq, d), V(seq, d);
        Q.randomize(); K.randomize(); V.randomize();

        double cpu_ms = measure_ms([&]{ (void)sdpa_cpu(Q, K, V); }, WARMUP, RUNS);

        std::cout << std::left  << std::setw(12) << seq
                  << std::fixed << std::setprecision(3)
                  << std::right << std::setw(14) << cpu_ms;

#ifdef USE_CUDA
        double gpu_ms = measure_ms([&]{ (void)sdpa_cuda(Q, K, V); }, WARMUP, RUNS);
        std::cout << std::right << std::setw(14) << gpu_ms
                  << std::right << std::setw(9)  << (cpu_ms / gpu_ms) << "x";
#endif
        std::cout << "\n";
    }
}

// =============================================================================
// Multi-Head Attention full-sequence benchmark
// =============================================================================
static void bench_mha_prefill() {
    std::cout << "\n  ── MHA Prefill (full-sequence) Benchmark ───────────────\n";
    std::cout << "  d_model=512, num_heads=8, head_dim=64\n\n";

    const int d_model = 512, num_heads = 8;
    MultiHeadAttention mha(d_model, num_heads);

    std::vector<int> seq_lens = {8, 16, 32, 64, 128};

    std::cout << std::left  << std::setw(12) << "seq_len"
              << std::right << std::setw(14) << "Time (ms)"
              << std::right << std::setw(14) << "Tokens/sec"
              << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (int seq : seq_lens) {
        Tensor inp(seq, d_model);
        inp.randomize(-0.5f, 0.5f);

        double ms = measure_ms([&]{ (void)mha.forward(inp); }, WARMUP, RUNS);
        double toks_per_sec = seq / (ms * 1e-3);

        std::cout << std::left  << std::setw(12) << seq
                  << std::fixed << std::setprecision(3)
                  << std::right << std::setw(14) << ms
                  << std::right << std::setw(14) << static_cast<int>(toks_per_sec)
                  << "\n";
    }
}

// =============================================================================
// KV-cache decode speed: time per token vs cache length
// =============================================================================
static void bench_kv_cache_decode() {
    std::cout << "\n  ── KV Cache: Decode Speed vs Context Length ─────────────\n";
    std::cout << "  d_model=256, num_heads=4, head_dim=64\n";
    std::cout << "  (Decoding one NEW token given N cached tokens)\n\n";

    const int d_model = 256, num_heads = 4;
    MultiHeadAttention mha(d_model, num_heads);

    std::vector<int> cache_lengths = {0, 16, 32, 64, 128, 256, 512};

    std::cout << std::left  << std::setw(14) << "Cache length"
              << std::right << std::setw(14) << "Time (ms)"
              << std::right << std::setw(14) << "vs baseline"
              << "\n";
    std::cout << std::string(42, '-') << "\n";

    double baseline_ms = -1.0;

    for (int ctx : cache_lengths) {
        // Pre-fill the cache to the desired length
        KVCache cache(num_heads, d_model / num_heads, ctx + 1 + RUNS + 10);
        {
            // Fill cache with ctx tokens
            Tensor dummy(1, d_model);
            dummy.randomize();
            for (int i = 0; i < ctx; ++i) mha.forward_cached(dummy, cache);
        }

        Tensor new_tok(1, d_model);
        new_tok.randomize();

        // Measure decoding one additional token (cache grows by 1 each call)
        // Use a fresh cache for each measurement to keep context size stable
        auto run_decode = [&]() {
            KVCache c2(num_heads, d_model / num_heads, ctx + 1 + 10);
            // Re-fill to ctx tokens
            Tensor d(1, d_model);
            for (int i = 0; i < ctx; ++i) mha.forward_cached(d, c2);
            mha.forward_cached(new_tok, c2);
        };

        double ms = measure_ms(run_decode, 1, 3);

        if (baseline_ms < 0.0) baseline_ms = ms;
        double ratio = ms / baseline_ms;

        std::cout << std::left  << std::setw(14) << ctx
                  << std::fixed << std::setprecision(3)
                  << std::right << std::setw(14) << ms
                  << std::right << std::setw(13) << ratio << "x"
                  << "\n";
    }

    std::cout << "\n  Note: Decode time grows linearly with context length\n"
              << "  (attention over all cached tokens = O(n) per step).\n"
              << "  Without KV cache: would be O(n²) total for n steps.\n";
}

// =============================================================================
// Compare: w/ cache vs w/o cache for a generation sequence
// =============================================================================
static void bench_cache_vs_no_cache() {
    std::cout << "\n  ── KV Cache vs No Cache: Total Generation Time ──────────\n";
    std::cout << "  d_model=128, num_heads=4, generating 32 tokens\n\n";

    const int d_model = 128, num_heads = 4, n_gen = 32;
    MultiHeadAttention mha(d_model, num_heads);

    // WITH KV cache: each step decodes one token, attending over growing cache
    Timer t;
    t.start();
    {
        KVCache cache(num_heads, d_model / num_heads, n_gen + 10);
        Tensor tok(1, d_model);
        for (int i = 0; i < n_gen; ++i) {
            tok.randomize(-0.1f, 0.1f);
            mha.forward_cached(tok, cache);
        }
    }
    double with_cache_ms = t.elapsed_ms();

    // WITHOUT KV cache: each step processes the FULL sequence from scratch
    t.start();
    {
        std::vector<Tensor> history;
        for (int i = 0; i < n_gen; ++i) {
            Tensor tok(1, d_model);
            tok.randomize(-0.1f, 0.1f);
            history.push_back(tok);

            // Build full sequence [i+1, d_model]
            Tensor seq_tensor(i + 1, d_model);
            for (int j = 0; j <= i; ++j)
                for (int c = 0; c < d_model; ++c)
                    seq_tensor.data[j * d_model + c] = history[j].data[c];
            // Run full-sequence attention
            mha.forward(seq_tensor);
        }
    }
    double no_cache_ms = t.elapsed_ms();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  With KV cache:    " << with_cache_ms << " ms\n";
    std::cout << "  Without KV cache: " << no_cache_ms   << " ms\n";
    std::cout << "  Speedup:          " << (no_cache_ms / with_cache_ms) << "×\n";
    std::cout << "\n  (Speedup grows with sequence length — O(n) vs O(n²))\n";
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════╗\n";
    std::cout << "  ║        Attention Mechanism Benchmark                 ║\n";
#ifdef USE_CUDA
    std::cout << "  ║        Mode: CUDA + CPU                             ║\n";
#else
    std::cout << "  ║        Mode: CPU-only                               ║\n";
#endif
    std::cout << "  ╚══════════════════════════════════════════════════════╝\n";

    try {
        bench_sdpa();
        bench_mha_prefill();
        bench_kv_cache_decode();
        bench_cache_vs_no_cache();
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n  Benchmark complete.\n\n";
    return 0;
}
