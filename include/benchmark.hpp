#pragma once
// =============================================================================
// benchmark.hpp — Timing, GFLOPS calculation, and result reporting
// =============================================================================
// All timing uses std::chrono::high_resolution_clock for maximum portability.
// GFLOPS calculation for matrix multiplication:
//   FLOPs = 2 × M × N × K  (each output element: K multiplies + K adds)
//   GFLOPS = FLOPs / (time_s × 1e9)
// =============================================================================

#include <chrono>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// =============================================================================
// Timer
// =============================================================================
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TP    = Clock::time_point;

    TP t0_;

    void start() { t0_ = Clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    }

    double elapsed_s() const { return elapsed_ms() / 1000.0; }
};

// =============================================================================
// BenchmarkResult — one row in the results table
// =============================================================================
struct BenchmarkResult {
    std::string name;          // e.g. "CPU Naive", "CUDA Tiled"
    int         matrix_size;   // square dimension N
    double      time_ms;       // average wall-clock time over all runs
    double      gflops;        // effective GFLOPS
    double      speedup;       // relative to the first (baseline) result
};

// =============================================================================
// Core measurement function
// =============================================================================
// Runs fn() warmup times (discarded), then runs times and returns the mean.
inline double measure_ms(std::function<void()> fn,
                         int warmup = 2,
                         int runs   = 5)
{
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> samples;
    samples.reserve(runs);
    Timer t;
    for (int i = 0; i < runs; ++i) {
        t.start();
        fn();
        samples.push_back(t.elapsed_ms());
    }
    // Return median to reduce noise from OS scheduling
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// =============================================================================
// GFLOPS for matmul: 2 × M × N × K operations
// =============================================================================
inline double matmul_gflops(int M, int N, int K, double time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);   // ms → s: /1000; /1e9 → G: combined /1e6
}

// =============================================================================
// Pretty-print a results table
// =============================================================================
inline void print_results_table(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) return;

    const int W = 72;
    std::string hline(W, '=');

    std::cout << "\n" << hline << "\n";
    std::cout << std::left
              << std::setw(22) << "Implementation"
              << std::setw(14) << "Size"
              << std::setw(14) << "Time (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(10) << "Speedup"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left  << std::setw(22) << r.name
                  << std::left  << std::setw(14) << (std::to_string(r.matrix_size) + "×" + std::to_string(r.matrix_size))
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << r.time_ms
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.gflops
                  << std::right << std::setw(9)  << std::fixed << std::setprecision(1) << r.speedup << "×"
                  << "\n";
    }
    std::cout << hline << "\n\n";
}

// =============================================================================
// Compute speedups relative to results[0] and fill the speedup field
// =============================================================================
inline void compute_speedups(std::vector<BenchmarkResult>& results) {
    if (results.empty()) return;
    double base = results[0].time_ms;
    for (auto& r : results) {
        r.speedup = (r.time_ms > 0.0) ? base / r.time_ms : 0.0;
    }
}
