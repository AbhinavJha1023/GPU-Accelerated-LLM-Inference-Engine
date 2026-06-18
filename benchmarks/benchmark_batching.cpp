// =============================================================================
// benchmark_batching.cpp — Continuous batching vs sequential throughput
// =============================================================================
// Compares two ways of serving many generation requests:
//   1. Sequential : one request at a time (baseline)
//   2. Continuous : dynamic batching with slot admission/eviction
//
// Measures requests/sec and tokens/sec on CPU. The speedup comes from batching
// the projection/logits matmuls across requests (larger matmuls are far more
// efficient than many tiny ones). On a GPU this effect is dramatically larger.
//
// Run:  ./benchmark_batching
// =============================================================================

#include "mini_transformer.hpp"
#include "scheduler.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

// Build a synthetic workload: requests with varied prompt lengths and
// generation lengths (mimics real traffic where requests differ in size).
static std::vector<Request> make_workload(int num_requests, int vocab) {
    std::vector<Request> reqs;
    reqs.reserve(num_requests);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> prompt_len(2, 8);
    std::uniform_int_distribution<int> gen_len(6, 20);
    std::uniform_int_distribution<int> tok(4, vocab - 1);

    for (int i = 0; i < num_requests; ++i) {
        Request r;
        r.id      = i;
        r.max_new = gen_len(rng);
        int plen  = prompt_len(rng);
        r.prompt.push_back(Tokenizer::BOS_ID);
        for (int j = 0; j < plen; ++j) r.prompt.push_back(tok(rng));
        reqs.push_back(std::move(r));
    }
    return reqs;
}

static void print_stats(const std::string& label, const SchedulerStats& s) {
    std::cout << std::left << std::setw(24) << label
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << s.wall_ms << " ms"
              << std::setw(12) << s.requests_per_sec << " req/s"
              << std::setw(12) << s.tokens_per_sec << " tok/s"
              << std::setw(10) << std::setprecision(1) << s.avg_batch_size << " avg-batch"
              << "\n";
}

int main() {
    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════╗\n";
    std::cout << "  ║      Continuous Batching vs Sequential (CPU)         ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════╝\n";

    const int VOCAB   = 64;
    const int D_MODEL = 128;
    const int N_HEADS = 4;
    const int MAX_LEN = 256;
    const int N_REQS  = 32;

    std::cout << "\n  Model: vocab=" << VOCAB << " d_model=" << D_MODEL
              << " heads=" << N_HEADS << "\n";
    std::cout << "  Workload: " << N_REQS << " requests, varied prompt/gen lengths\n\n";

    MiniTransformer model(VOCAB, D_MODEL, N_HEADS, MAX_LEN);

    std::cout << std::left << std::setw(24) << "  Strategy"
              << std::right << std::setw(15) << "Wall time"
              << std::setw(16) << "Throughput"
              << std::setw(16) << "Token rate"
              << std::setw(16) << "Occupancy" << "\n";
    std::cout << "  " << std::string(80, '-') << "\n";

    // ── Sequential baseline ──────────────────────────────────────────────────
    SchedulerStats seq;
    {
        std::vector<Request> reqs = make_workload(N_REQS, VOCAB);
        seq = run_sequential(model, reqs);
        print_stats("  Sequential (batch=1)", seq);
    }

    // ── Continuous batching at several max-batch sizes ───────────────────────
    for (int B : {4, 8, 16}) {
        std::vector<Request> reqs = make_workload(N_REQS, VOCAB);
        ContinuousBatchScheduler sched(model, B);
        for (auto& r : reqs) sched.add_request(&r);
        SchedulerStats st = sched.run();
        print_stats("  Continuous (max=" + std::to_string(B) + ")", st);
    }

    std::cout << "\n  Speedup is reported as tokens/sec relative to sequential.\n";
    std::cout << "  Note: on CPU the gain is modest (better matmul cache use);\n";
    std::cout << "  on a GPU, batched matmuls turn this into a large throughput win.\n\n";
    return 0;
}
