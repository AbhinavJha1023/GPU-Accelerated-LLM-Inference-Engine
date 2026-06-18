// =============================================================================
// main.cpp — GPU-Accelerated LLM Inference Engine: Full Demo
// =============================================================================
// This program walks through every component of the engine in order,
// demonstrating correctness and — where CUDA is available — GPU acceleration.
//
// Run:  ./llm_engine
//
// Phases:
//   1  Tensor operations
//   2  CPU matrix multiplication (naive vs. optimised)
//   3  Softmax
//   4  Scaled dot-product attention
//   5  Multi-head attention
//   6  KV cache
//   7  Tokenizer
//   8  Autoregressive MiniTransformer
//   9  CUDA operations (if compiled with USE_CUDA)
//  10  Benchmark summary
// =============================================================================

#include "tensor.hpp"
#include "matmul.hpp"
#include "attention.hpp"
#include "kv_cache.hpp"
#include "tokenizer.hpp"
#include "benchmark.hpp"
#include "mini_transformer.hpp"   // MiniTransformer now lives in its own TU
#include "quantize.hpp"
#include "scheduler.hpp"          // continuous batching

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>

// ── Colour / separator helpers ────────────────────────────────────────────────
static void section(const std::string& title) {
    std::cout << "\n";
    std::cout << std::string(66, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(66, '=') << "\n";
}

static void subsection(const std::string& title) {
    std::cout << "\n  ── " << title << " ──\n";
}

// MiniTransformer is now defined in include/mini_transformer.hpp +
// src/mini_transformer.cpp so it can be reused by the continuous-batching
// scheduler and benchmarks. See those files for the full implementation.

// =============================================================================
// Phase 1 — Tensor Operations
// =============================================================================
static void phase1_tensors() {
    section("Phase 1 — Tensor Operations");

    subsection("Construction & indexing");
    Tensor A(3, 4);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f; A(0, 3) = 4.0f;
    A(1, 0) = 5.0f; A(1, 1) = 6.0f; A(1, 2) = 7.0f; A(1, 3) = 8.0f;
    A(2, 0) = 9.0f; A(2, 1) =10.0f; A(2, 2) =11.0f; A(2, 3) =12.0f;
    A.print("A");

    subsection("Transpose");
    Tensor At = A.transpose();
    At.print("Aᵀ");

    subsection("Identity matrix");
    Tensor I = eye(4);
    I.print("I[4]");

    subsection("Random tensor");
    Tensor R(3, 3);
    R.randomize(-2.0f, 2.0f);
    R.print("R (random)");

    subsection("Element-wise add");
    Tensor B(3, 3, 1.0f);
    Tensor C = add(R, B);
    C.print("R + 1");

    subsection("Scale");
    Tensor S2 = scale(R, 2.0f);
    S2.print("R × 2");

    std::cout << "\n  allclose(A, A):  " << (allclose(A, A) ? "true" : "false") << "\n";
    std::cout << "  allclose(A, At): " << (allclose(A, At) ? "true" : "false") << "\n";
}

// =============================================================================
// Phase 2 — CPU Matrix Multiplication
// =============================================================================
static void phase2_matmul() {
    section("Phase 2 — CPU Matrix Multiplication");

    // Small correctness check
    subsection("2×2 correctness: [1,2;3,4] × [5,6;7,8]");
    Tensor M1(2, 2), M2(2, 2);
    M1(0,0)=1; M1(0,1)=2; M1(1,0)=3; M1(1,1)=4;
    M2(0,0)=5; M2(0,1)=6; M2(1,0)=7; M2(1,1)=8;
    Tensor naive_res  = matmul_cpu_naive(M1, M2);
    Tensor optim_res  = matmul_cpu_optimized(M1, M2);
    naive_res.print("Naive result (expect [19,22;43,50])");
    std::cout << "  Results match: " << (allclose(naive_res, optim_res) ? "YES" : "NO") << "\n";

    // Larger timing comparison
    subsection("Timing: 256×256 matrices");
    Tensor X(256, 256), Y(256, 256);
    X.randomize(); Y.randomize();

    Timer t;
    t.start();
    Tensor RN = matmul_cpu_naive(X, Y);
    double t_naive = t.elapsed_ms();

    t.start();
    Tensor RO = matmul_cpu_optimized(X, Y);
    double t_optim = t.elapsed_ms();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Naive:      " << t_naive << " ms\n";
    std::cout << "  Optimised:  " << t_optim << " ms\n";
    std::cout << "  Speedup:    " << (t_naive / t_optim) << "×\n";
    std::cout << "  Results match: " << (allclose(RN, RO) ? "YES" : "NO") << "\n";

    double gflops = matmul_gflops(256, 256, 256, t_optim);
    std::cout << "  GFLOPS (optimised): " << gflops << "\n";
}

// =============================================================================
// Phase 3 — Softmax
// =============================================================================
static void phase3_softmax() {
    section("Phase 3 — Softmax (numerically stable)");

    subsection("Row-wise softmax on 2×4 matrix");
    Tensor x(2, 4);
    x(0,0)=1.0f; x(0,1)=2.0f; x(0,2)=3.0f; x(0,3)=4.0f;
    x(1,0)=0.1f; x(1,1)=0.2f; x(1,2)=0.3f; x(1,3)=0.4f;
    x.print("Input");

    Tensor sm = softmax_cpu(x);
    sm.print("Softmax output (rows should sum to 1.0)");

    // Verify row sums
    for (int r = 0; r < sm.rows; ++r) {
        float s = 0.0f;
        for (int c = 0; c < sm.cols; ++c) s += sm(r, c);
        std::cout << "  Row " << r << " sum = " << s << " (expected 1.0)\n";
    }

    subsection("Numerical stability: large values");
    Tensor big(1, 4);
    big(0,0)=1000.0f; big(0,1)=1001.0f; big(0,2)=1002.0f; big(0,3)=999.0f;
    big.print("Input with large values");
    Tensor sm_big = softmax_cpu(big);
    sm_big.print("Stable softmax output (no NaN/Inf)");
    float sum_check = 0.0f;
    for (int c = 0; c < sm_big.cols; ++c) sum_check += sm_big(0, c);
    std::cout << "  Sum = " << sum_check << "\n";
}

// =============================================================================
// Phase 4 — Scaled Dot-Product Attention
// =============================================================================
static void phase4_sdpa() {
    section("Phase 4 — Scaled Dot-Product Attention");
    std::cout << "\n  Formula: Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V\n";

    subsection("3 tokens, d_k=4, d_v=4");
    int seq = 3, dk = 4, dv = 4;

    Tensor Q(seq, dk), K(seq, dk), V(seq, dv);
    Q.randomize(-1.0f, 1.0f);
    K.randomize(-1.0f, 1.0f);
    V.randomize(-1.0f, 1.0f);

    Q.print("Q [3×4]");
    K.print("K [3×4]");
    V.print("V [3×4]");

    Tensor out = sdpa_cpu(Q, K, V);
    out.print("Attention output [3×4]");

    std::cout << "\n  Step-by-step:\n";
    Tensor Kt    = K.transpose();
    Tensor scores = matmul_cpu_naive(Q, Kt);
    std::cout << "  1. Q·Kᵀ:\n";
    scores.print("", 3, 3);

    float sc = 1.0f / std::sqrt(static_cast<float>(dk));
    for (float& v : scores.data) v *= sc;
    std::cout << "  2. Scaled (/ √" << dk << " = " << sc << "):\n";
    scores.print("", 3, 3);

    Tensor weights = softmax_cpu(scores);
    std::cout << "  3. Softmax (attention weights):\n";
    weights.print("", 3, 3);
}

// =============================================================================
// Phase 5 — Multi-Head Attention
// =============================================================================
static void phase5_mha() {
    section("Phase 5 — Multi-Head Attention");

    const int d_model   = 64;
    const int num_heads = 4;   // head_dim = 16
    const int seq_len   = 5;

    std::cout << "\n  d_model=" << d_model
              << "  num_heads=" << num_heads
              << "  head_dim=" << (d_model/num_heads)
              << "  seq_len=" << seq_len << "\n";

    Tensor input(seq_len, d_model);
    input.randomize(-0.5f, 0.5f);
    input.print("Input [5×64]", 3, 8);

    MultiHeadAttention mha(d_model, num_heads);
    Tensor output = mha.forward(input);
    output.print("MHA output [5×64]", 3, 8);

    std::cout << "\n  Shape check: "
              << input.shape() << " → " << output.shape() << "\n";
    assert(output.rows == seq_len && output.cols == d_model);
    std::cout << "  Shape correct!\n";
}

// =============================================================================
// Phase 6 — KV Cache
// =============================================================================
static void phase6_kv_cache() {
    section("Phase 6 — KV Cache");

    std::cout << "\n  Demonstrating sequence growth and O(n) vs O(n²) complexity.\n";

    const int num_heads = 2, head_dim = 8, max_len = 16;
    KVCache cache(num_heads, head_dim, max_len);

    std::cout << "\n  Appending tokens 1..6 to the cache:\n";
    for (int t = 0; t < 6; ++t) {
        Tensor k(1, num_heads * head_dim);
        Tensor v(1, num_heads * head_dim);
        k.randomize(-1.0f, 1.0f);
        v.randomize(-1.0f, 1.0f);
        cache.append(k, v);
        std::cout << "    After append " << (t+1) << ": cache length = "
                  << cache.length() << "\n";
    }

    std::cout << "\n";
    cache.print_info();

    Tensor all_keys   = cache.get_keys();
    Tensor all_values = cache.get_values();
    std::cout << "\n  get_keys()  → " << all_keys.shape()   << "\n";
    std::cout << "  get_values()→ " << all_values.shape() << "\n";

    std::cout << "\n  Complexity comparison:\n"
              << "    Without KV cache: O(n²) matmul operations (recompute all K,V each step)\n"
              << "    With    KV cache: O(n)  matmul operations (only new token's K,V)\n"
              << "    For n=512 tokens: 512× speedup in matmul count!\n";

    cache.clear();
    std::cout << "\n  After clear: length = " << cache.length() << "\n";
}

// =============================================================================
// Phase 7 — Tokenizer
// =============================================================================
static void phase7_tokenizer() {
    section("Phase 7 — Character-level Tokenizer");

    Tokenizer tok;
    std::string corpus = "Hello, world! This is an LLM inference engine.";
    tok.build_vocab(corpus);

    std::cout << "\n  Corpus: \"" << corpus << "\"\n";
    tok.print_vocab();

    std::string test = "Hello";
    std::vector<int> ids = tok.encode(test, /*bos=*/true, /*eos=*/true);
    std::cout << "\n  Encode(\"" << test << "\"): [ ";
    for (int id : ids) std::cout << id << " ";
    std::cout << "]\n";

    std::string decoded = tok.decode(ids);
    std::cout << "  Decode back: \"" << decoded << "\"\n";

    std::cout << "  Round-trip: " << (decoded == test ? "PASS" : "FAIL") << "\n";
}

// =============================================================================
// Phase 8 — Autoregressive Generation
// =============================================================================
static void phase8_autoregressive() {
    section("Phase 8 — Autoregressive MiniTransformer Generation");

    const int VOCAB    =  32;   // small vocabulary for demo speed
    const int D_MODEL  =  32;   // embedding dimension (tiny, for speed)
    const int N_HEADS  =   4;   // 4 heads × 8 head_dim = 32
    const int MAX_LEN  =  64;

    std::cout << "\n  MiniTransformer: vocab=" << VOCAB
              << " d_model=" << D_MODEL
              << " num_heads=" << N_HEADS << "\n";
    std::cout << "  (Weights are random — output is token IDs, not real text)\n";

    MiniTransformer transformer(VOCAB, D_MODEL, N_HEADS, MAX_LEN);

    // Prompt: BOS + a few token IDs
    std::vector<int> prompt = {Tokenizer::BOS_ID, 5, 10, 7};
    std::cout << "\n  Prompt token IDs: [ ";
    for (int id : prompt) std::cout << id << " ";
    std::cout << "]\n";

    auto generated = transformer.generate(prompt, /*max_new_tokens=*/10);

    std::cout << "\n  Generated sequence (IDs): [ ";
    for (int id : generated) std::cout << id << " ";
    std::cout << "]\n";
    std::cout << "  Prompt length:   " << prompt.size() << "\n";
    std::cout << "  Generated tokens: " << (generated.size() - prompt.size()) << "\n";
    std::cout << "  Cache length after generation: " << transformer.cache.length() << "\n";

    std::cout << "\n  How autoregressive inference works:\n"
              << "    1. Prefill:  process all N prompt tokens at once\n"
              << "       (builds KV cache entries for positions 0..N-1)\n"
              << "    2. Generate: for each step t = N, N+1, ...\n"
              << "       a. Embed token[t-1]\n"
              << "       b. MHA.forward_cached → append K,V to cache\n"
              << "       c. Project hidden → logits [vocab_size]\n"
              << "       d. Next token = argmax(logits)\n"
              << "       e. Append next token to sequence\n";
}

// =============================================================================
// Phase 9 — CUDA Section (only when compiled with USE_CUDA)
// =============================================================================
static void phase9_cuda() {
#ifdef USE_CUDA
    section("Phase 9 — CUDA GPU Operations");

    // Query device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    std::cout << "\n  GPU: " << props.name << "\n";
    std::cout << "  Compute capability: " << props.major << "." << props.minor << "\n";
    std::cout << "  Global memory: " << props.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "  SM count: " << props.multiProcessorCount << "\n";
    std::cout << "  Max threads/block: " << props.maxThreadsPerBlock << "\n";

    subsection("CUDA matmul correctness: 64×64");
    Tensor A(64, 64), B(64, 64);
    A.randomize(); B.randomize();
    Tensor cpu_res  = matmul_cpu_optimized(A, B);
    Tensor cuda_nav = matmul_cuda_naive(A, B);
    Tensor cuda_til = matmul_cuda_tiled(A, B);
    std::cout << "  CPU == CUDA naive: " << (allclose(cpu_res, cuda_nav) ? "PASS" : "FAIL") << "\n";
    std::cout << "  CPU == CUDA tiled: " << (allclose(cpu_res, cuda_til) ? "PASS" : "FAIL") << "\n";

    subsection("CUDA softmax correctness: 8×32");
    Tensor sx(8, 32);
    sx.randomize(-2.0f, 2.0f);
    Tensor sm_cpu  = softmax_cpu(sx);
    Tensor sm_cuda = softmax_cuda(sx);
    std::cout << "  CPU == CUDA softmax: " << (allclose(sm_cpu, sm_cuda) ? "PASS" : "FAIL") << "\n";

    subsection("CUDA SDPA correctness: seq=4, dk=dv=16");
    Tensor Q(4,16), K(4,16), V(4,16);
    Q.randomize(); K.randomize(); V.randomize();
    Tensor sdpa_cpu_out  = sdpa_cpu(Q, K, V);
    Tensor sdpa_cuda_out = sdpa_cuda(Q, K, V);
    std::cout << "  CPU == CUDA SDPA: " << (allclose(sdpa_cpu_out, sdpa_cuda_out) ? "PASS" : "FAIL") << "\n";

#else
    section("Phase 9 — CUDA GPU Operations");
    std::cout << "\n  [SKIPPED] Compiled without CUDA.\n"
              << "  To enable: cmake -B build -DUSE_CUDA=ON && cmake --build build\n"
              << "  On Google Colab: see docs/architecture.md for setup instructions.\n";
#endif
}

// =============================================================================
// Phase 10 — Benchmark
// =============================================================================
static void phase10_benchmark() {
    section("Phase 10 — Performance Benchmark");

    std::vector<int> sizes = {64, 128, 256, 512};

    std::cout << "\n  Running matrix multiplication benchmark...\n";
    std::cout << "  (Measuring wall-clock time; 2 warmup + 5 measured runs each)\n\n";

    std::vector<BenchmarkResult> results;

    for (int N : sizes) {
        Tensor A(N, N), B(N, N);
        A.randomize(); B.randomize();

        // CPU Naive
        {
            double ms = measure_ms([&]{ auto r = matmul_cpu_naive(A, B); }, 1, 3);
            results.push_back({"CPU Naive", N, ms, matmul_gflops(N,N,N,ms), 1.0});
        }
        // CPU Optimised
        {
            double ms = measure_ms([&]{ auto r = matmul_cpu_optimized(A, B); }, 1, 3);
            results.push_back({"CPU Optimised", N, ms, matmul_gflops(N,N,N,ms), 1.0});
        }
#ifdef USE_CUDA
        // CUDA Naive
        {
            double ms = measure_ms([&]{ auto r = matmul_cuda_naive(A, B); }, 1, 3);
            results.push_back({"CUDA Naive", N, ms, matmul_gflops(N,N,N,ms), 1.0});
        }
        // CUDA Tiled
        {
            double ms = measure_ms([&]{ auto r = matmul_cuda_tiled(A, B); }, 1, 3);
            results.push_back({"CUDA Tiled", N, ms, matmul_gflops(N,N,N,ms), 1.0});
        }
#endif
    }

    // Compute speedups per size group
    size_t group = results.size() / sizes.size();
    for (size_t g = 0; g < sizes.size(); ++g) {
        size_t start = g * group;
        double base = results[start].time_ms;
        for (size_t i = start; i < start + group; ++i)
            results[i].speedup = base / results[i].time_ms;
    }

    print_results_table(results);

    // Attention benchmark
    std::cout << "  Running attention benchmark (seq_len=32, d_model=64, heads=4)...\n\n";
    const int seq = 32, dm = 64, nh = 4;
    Tensor inp(seq, dm);
    inp.randomize();
    MultiHeadAttention mha(dm, nh);

    std::vector<BenchmarkResult> att_results;
    {
        double ms = measure_ms([&]{ mha.forward(inp); }, 1, 5);
        att_results.push_back({"MHA CPU", seq, ms, 0.0, 1.0});
    }
    for (auto& r : att_results) r.speedup = att_results[0].time_ms / r.time_ms;

    std::cout << "  MHA forward (seq=32, d=64, heads=4): "
              << std::fixed << std::setprecision(3)
              << att_results[0].time_ms << " ms\n";
}

// =============================================================================
// Phase 11 — Quantization (INT8 / INT4)
// =============================================================================
static void phase11_quantization() {
    section("Phase 11 — Weight Quantization (INT8 / INT4)");
    std::cout << "\n  Symmetric linear quantization: scale = max|x| / qmax\n";

    Tensor W(128, 128);
    W.randomize(-1.0f, 1.0f);
    size_t fp32_bytes = (size_t)W.size() * sizeof(float);

    QTensorInt8 q8 = quantize_int8(W);
    QTensorInt4 q4 = quantize_int4(W);
    QuantError e8 = quant_error(W, dequantize_int8(q8));
    QuantError e4 = quant_error(W, dequantize_int4(q4));

    std::cout << "\n  Weight matrix 128x128 (" << W.size() << " weights):\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "    FP32 : " << std::setw(7) << fp32_bytes/1024  << " KB   (baseline)\n";
    std::cout << "    INT8 : " << std::setw(7) << q8.bytes()/1024  << " KB   ("
              << 100.0*q8.bytes()/fp32_bytes << "% of FP32)   max err " << e8.max_abs_error << "\n";
    std::cout << "    INT4 : " << std::setw(7) << q4.bytes()/1024  << " KB   ("
              << 100.0*q4.bytes()/fp32_bytes << "% of FP32)   max err " << e4.max_abs_error << "\n";

    std::cout << "\n  INT8 is 4x smaller with ~1e-3 error; INT4 is 8x smaller with ~1e-1 error.\n";
    std::cout << "  (Full report: run benchmark_quantize)\n";
}

// =============================================================================
// Phase 12 — Continuous Batching
// =============================================================================
static void phase12_continuous_batching() {
    section("Phase 12 — Continuous Batching");
    std::cout << "\n  Serving many requests: sequential (batch=1) vs continuous batching.\n";

    const int VOCAB = 48, D_MODEL = 64, N_HEADS = 4, MAX_LEN = 128, N_REQS = 16;
    MiniTransformer model(VOCAB, D_MODEL, N_HEADS, MAX_LEN);

    auto make_reqs = [&]() {
        std::vector<Request> v;
        for (int i = 0; i < N_REQS; ++i) {
            Request r; r.id = i; r.max_new = 8 + (i % 6);
            r.prompt = { Tokenizer::BOS_ID, 5 + (i % 10), 12, 7 };
            v.push_back(std::move(r));
        }
        return v;
    };

    std::vector<Request> seq_reqs = make_reqs();
    SchedulerStats seq = run_sequential(model, seq_reqs);

    std::vector<Request> batch_reqs = make_reqs();
    ContinuousBatchScheduler sched(model, /*max_batch=*/8);
    for (auto& r : batch_reqs) sched.add_request(&r);
    SchedulerStats batched = sched.run();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  " << N_REQS << " requests, d_model=" << D_MODEL << ", max_batch=8\n";
    std::cout << "    Sequential : " << std::setw(8) << seq.wall_ms     << " ms   "
              << seq.tokens_per_sec     << " tok/s\n";
    std::cout << "    Continuous : " << std::setw(8) << batched.wall_ms << " ms   "
              << batched.tokens_per_sec << " tok/s   (avg batch "
              << std::setprecision(1) << batched.avg_batch_size << ")\n";
    if (batched.tokens_per_sec > 0 && seq.tokens_per_sec > 0)
        std::cout << "    Throughput speedup: " << std::setprecision(2)
                  << (batched.tokens_per_sec / seq.tokens_per_sec) << "x\n";
    std::cout << "\n  (Full sweep over batch sizes: run benchmark_batching)\n";
}

// =============================================================================
// main
// =============================================================================
int main() {
    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "  ║     GPU-Accelerated LLM Inference Engine — Demo          ║\n";
    std::cout << "  ║     C++20 + CUDA | Educational Implementation            ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════════╝\n";

#ifdef USE_CUDA
    std::cout << "\n  Build mode: CUDA enabled\n";
#else
    std::cout << "\n  Build mode: CPU-only (no GPU required)\n";
#endif

    try {
        phase1_tensors();
        phase2_matmul();
        phase3_softmax();
        phase4_sdpa();
        phase5_mha();
        phase6_kv_cache();
        phase7_tokenizer();
        phase8_autoregressive();
        phase9_cuda();
        phase10_benchmark();
        phase11_quantization();
        phase12_continuous_batching();
    } catch (const std::exception& ex) {
        std::cerr << "\n[ERROR] " << ex.what() << "\n";
        return 1;
    }

    std::cout << "\n" << std::string(66, '=') << "\n";
    std::cout << "  All phases complete.\n";
    std::cout << std::string(66, '=') << "\n\n";
    return 0;
}
