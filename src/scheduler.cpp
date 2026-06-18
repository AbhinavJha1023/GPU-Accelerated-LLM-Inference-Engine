// =============================================================================
// scheduler.cpp — Continuous batching scheduler implementation
// =============================================================================
#include "scheduler.hpp"
#include "tokenizer.hpp"
#include "benchmark.hpp"
#include <stdexcept>

ContinuousBatchScheduler::ContinuousBatchScheduler(MiniTransformer& model,
                                                   int max_batch_size)
    : model_(model), max_batch_size_(max_batch_size) {}

void ContinuousBatchScheduler::add_request(Request* req) {
    waiting_.push_back(req);
}

// Prefill: run the prompt through the model to populate this slot's KV cache.
// After prefill, the slot is ready to generate from its last prompt token.
void ContinuousBatchScheduler::prefill(Slot& slot) {
    const auto& prompt = slot.req->prompt;
    for (int i = 0; i < static_cast<int>(prompt.size()); ++i) {
        std::vector<int> tok = { prompt[i] };
        std::vector<int> pos = { i };
        std::vector<KVCache*> caches = { slot.cache.get() };
        model_.forward_step_batch(tok, pos, caches);   // builds cache entry
    }
    slot.pos       = static_cast<int>(prompt.size());
    slot.cur_token = prompt.empty() ? Tokenizer::BOS_ID : prompt.back();
    slot.generated = 0;
}

// Move waiting requests into free running slots (the "continuous" admission).
void ContinuousBatchScheduler::admit_waiting() {
    while (!waiting_.empty() &&
           static_cast<int>(running_.size()) < max_batch_size_) {
        Request* req = waiting_.front();
        waiting_.pop_front();

        Slot slot;
        slot.req   = req;
        slot.cache = std::make_unique<KVCache>(
            model_.num_heads, model_.d_model / model_.num_heads,
            model_.max_seq_len);
        prefill(slot);
        running_.push_back(std::move(slot));
    }
}

SchedulerStats ContinuousBatchScheduler::run() {
    SchedulerStats stats;
    stats.total_requests = static_cast<int>(waiting_.size());

    long long batch_size_accum = 0;

    Timer timer;
    timer.start();

    admit_waiting();   // initial fill

    while (!running_.empty()) {
        // ── Assemble the current batch from all running slots ──────────────────
        int B = static_cast<int>(running_.size());
        std::vector<int>      tokens(B), positions(B);
        std::vector<KVCache*> caches(B);
        for (int i = 0; i < B; ++i) {
            tokens[i]    = running_[i].cur_token;
            positions[i] = running_[i].pos;
            caches[i]    = running_[i].cache.get();
        }

        // ── One batched decode step (the throughput win lives here) ───────────
        std::vector<int> next = model_.forward_step_batch(tokens, positions, caches);
        stats.decode_steps++;
        batch_size_accum += B;

        // ── Apply results; mark finished requests ─────────────────────────────
        for (int i = 0; i < B; ++i) {
            Slot& slot = running_[i];
            int tok = next[i];
            slot.req->output.push_back(tok);
            slot.cur_token = tok;
            slot.pos++;
            slot.generated++;
            stats.total_tokens++;

            if (tok == Tokenizer::EOS_ID ||
                slot.generated >= slot.req->max_new ||
                slot.cache->is_full()) {
                slot.req->finished = true;
            }
        }

        // ── Evict finished slots (compact the running vector) ─────────────────
        std::vector<Slot> survivors;
        survivors.reserve(running_.size());
        for (auto& slot : running_) {
            if (!slot.req->finished) survivors.push_back(std::move(slot));
        }
        running_ = std::move(survivors);

        // ── Admit waiting requests into the slots we just freed ───────────────
        admit_waiting();
    }

    stats.wall_ms = timer.elapsed_ms();
    double secs = stats.wall_ms / 1000.0;
    stats.requests_per_sec = secs > 0 ? stats.total_requests / secs : 0.0;
    stats.tokens_per_sec   = secs > 0 ? stats.total_tokens   / secs : 0.0;
    stats.avg_batch_size   = stats.decode_steps > 0
        ? static_cast<double>(batch_size_accum) / stats.decode_steps : 0.0;
    return stats;
}

// =============================================================================
// Sequential baseline — one request fully processed before the next starts
// =============================================================================
SchedulerStats run_sequential(MiniTransformer& model,
                              std::vector<Request>& requests) {
    SchedulerStats stats;
    stats.total_requests = static_cast<int>(requests.size());

    Timer timer;
    timer.start();

    for (auto& req : requests) {
        KVCache cache(model.num_heads, model.d_model / model.num_heads,
                      model.max_seq_len);

        // Prefill
        for (int i = 0; i < static_cast<int>(req.prompt.size()); ++i) {
            std::vector<int> tok = { req.prompt[i] };
            std::vector<int> pos = { i };
            std::vector<KVCache*> caches = { &cache };
            model.forward_step_batch(tok, pos, caches);
        }

        // Decode
        int pos       = static_cast<int>(req.prompt.size());
        int cur_token = req.prompt.empty() ? Tokenizer::BOS_ID : req.prompt.back();
        for (int g = 0; g < req.max_new; ++g) {
            std::vector<int> tk = { cur_token };
            std::vector<int> ps = { pos };
            std::vector<KVCache*> caches = { &cache };
            std::vector<int> next = model.forward_step_batch(tk, ps, caches);
            cur_token = next[0];
            req.output.push_back(cur_token);
            stats.total_tokens++;
            pos++;
            if (cur_token == Tokenizer::EOS_ID || cache.is_full()) break;
        }
    }

    stats.wall_ms = timer.elapsed_ms();
    double secs = stats.wall_ms / 1000.0;
    stats.requests_per_sec = secs > 0 ? stats.total_requests / secs : 0.0;
    stats.tokens_per_sec   = secs > 0 ? stats.total_tokens   / secs : 0.0;
    stats.avg_batch_size   = 1.0;
    return stats;
}
