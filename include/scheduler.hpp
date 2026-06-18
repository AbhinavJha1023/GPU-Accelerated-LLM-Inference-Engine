#pragma once
// =============================================================================
// scheduler.hpp — Continuous batching scheduler
// =============================================================================
// THE PROBLEM
// ────────────
// Naively, an inference server handles one request at a time: a request with
// 5 tokens to generate occupies the engine for 5 steps, then the next request
// starts. The hardware sits underused, and short requests wait behind long ones.
//
// STATIC BATCHING
// ────────────────
// Group N requests and decode them together. Better, BUT the whole batch must
// wait for the SLOWEST request to finish before any new request is admitted —
// finished slots sit idle ("bubble").
//
// CONTINUOUS BATCHING (this file)
// ────────────────────────────────
// Used by vLLM / SGLang / TGI. The batch is dynamic:
//   • Each step, decode one token for every RUNNING request (batched matmuls).
//   • The moment a request finishes (EOS or max tokens), EVICT it and ADMIT a
//     waiting request into the freed slot — without stalling the others.
// This keeps every slot busy, maximizing throughput.
//
//   step:   1    2    3    4    5    6
//   slot0:  A    A    A   [B]   B    B     (A finishes at step 3, B admitted)
//   slot1:  C    C    C    C    C   [D]    (C finishes at step 5, D admitted)
//            └ no waiting for the whole batch to drain ┘
//
// This implementation runs on CPU and measures throughput (requests/sec and
// tokens/sec). The batched matmuls in MiniTransformer::forward_step_batch are
// where the speedup comes from.
// =============================================================================

#include "mini_transformer.hpp"
#include "kv_cache.hpp"
#include <vector>
#include <deque>
#include <memory>

// One generation request
struct Request {
    int              id;
    std::vector<int> prompt;       // prompt token ids
    int              max_new;      // max tokens to generate
    std::vector<int> output;       // generated token ids (filled in by scheduler)
    bool             finished = false;
};

// A running slot: a request plus its private KV cache and decode position
struct Slot {
    Request*                 req;
    std::unique_ptr<KVCache> cache;
    int                      pos;        // current sequence position
    int                      cur_token;  // last token (input to next step)
    int                      generated;  // tokens generated so far
};

struct SchedulerStats {
    int    total_requests   = 0;
    int    total_tokens     = 0;   // total tokens generated across all requests
    int    decode_steps     = 0;   // number of batched decode steps executed
    double wall_ms          = 0.0;
    double requests_per_sec = 0.0;
    double tokens_per_sec   = 0.0;
    double avg_batch_size   = 0.0; // mean running-slot occupancy per step
};

class ContinuousBatchScheduler {
public:
    ContinuousBatchScheduler(MiniTransformer& model, int max_batch_size);

    // Queue a request to be processed
    void add_request(Request* req);

    // Run until every queued request is finished. Returns timing/throughput.
    SchedulerStats run();

private:
    MiniTransformer&     model_;
    int                  max_batch_size_;
    std::deque<Request*> waiting_;     // not yet started
    std::vector<Slot>    running_;     // currently decoding

    void admit_waiting();              // fill free slots from the waiting queue
    void prefill(Slot& slot);          // process a request's prompt into its cache
};

// =============================================================================
// Baseline for comparison: process requests one at a time (no batching)
// =============================================================================
SchedulerStats run_sequential(MiniTransformer& model,
                              std::vector<Request>& requests);
