// =============================================================================
// benchmark_matmul.cpp — Matrix multiplication performance benchmark
// =============================================================================
// Standalone executable — does NOT use GoogleTest.
// Run:  ./benchmark_matmul [size1 size2 ...]
//
// Default sizes: 64, 128, 256, 512, 1024
//
// Output example:
//   Implementation          Size          Time (ms)    GFLOPS   Speedup
//   ────────────────────────────────────────────────────────────────────
//   CPU Naive               256×256       142.31        0.23    1.0×
//   CPU Optimised           256×256        18.44        1.79    7.7×
//   CUDA Naive              256×256         1.23       26.80  115.7×    (if GPU)
//   CUDA Tiled              256×256         0.54       61.12  263.5×    (if GPU)
// =============================================================================

#include "tensor.hpp"
#include "matmul.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

// Warmup + measured runs per configuration
static constexpr int WARMUP = 2;
static constexpr int RUNS   = 5;

// =============================================================================
// Run a full benchmark for one matrix size
// =============================================================================
static void benchmark_size(int N, std::vector<BenchmarkResult>& all_results) {
    std::cout << "  Benchmarking N=" << N
              << " (" << N << "×" << N << "×" << N << ") ... " << std::flush;

    Tensor A(N, N), B(N, N);
    A.randomize(-1.0f, 1.0f);
    B.randomize(-1.0f, 1.0f);

    std::vector<BenchmarkResult> group;

    // ── CPU Naive ─────────────────────────────────────────────────────────────
    // Only run for small sizes — O(N³) is painfully slow for N≥512
    if (N <= 256) {
        double ms = measure_ms([&]{ (void)matmul_cpu_naive(A, B); }, WARMUP, RUNS);
        group.push_back({"CPU Naive", N, ms, matmul_gflops(N,N,N,ms), 1.0});
    } else {
        group.push_back({"CPU Naive", N, -1.0, 0.0, 0.0});   // skipped marker
    }

    // ── CPU Optimised ─────────────────────────────────────────────────────────
    {
        double ms = measure_ms([&]{ (void)matmul_cpu_optimized(A, B); }, WARMUP, RUNS);
        group.push_back({"CPU Optimised", N, ms, matmul_gflops(N,N,N,ms), 1.0});
    }

#ifdef USE_CUDA
    // ── CUDA Naive ───────────────────────────────────────────────────────────
    {
        double ms = measure_ms([&]{ (void)matmul_cuda_naive(A, B); }, WARMUP, RUNS);
        group.push_back({"CUDA Naive", N, ms, matmul_gflops(N,N,N,ms), 1.0});
    }

    // ── CUDA Tiled ───────────────────────────────────────────────────────────
    {
        double ms = measure_ms([&]{ (void)matmul_cuda_tiled(A, B); }, WARMUP, RUNS);
        group.push_back({"CUDA Tiled", N, ms, matmul_gflops(N,N,N,ms), 1.0});
    }
#endif

    // Compute speedup relative to the first valid result
    double base_ms = -1.0;
    for (auto& r : group)
        if (r.time_ms > 0.0 && base_ms < 0.0) base_ms = r.time_ms;
    for (auto& r : group)
        r.speedup = (r.time_ms > 0.0) ? base_ms / r.time_ms : 0.0;

    for (auto& r : group) all_results.push_back(r);
    std::cout << "done\n";
}

// =============================================================================
// Pretty table with separator lines between sizes
// =============================================================================
static void print_full_table(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) return;

    const int W = 74;
    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << "  Matrix Multiplication Benchmark Results\n";
    std::cout << std::string(W, '=') << "\n";
    std::cout << std::left  << std::setw(20) << "Implementation"
              << std::left  << std::setw(14) << "Matrix Size"
              << std::right << std::setw(12) << "Time (ms)"
              << std::right << std::setw(10) << "GFLOPS"
              << std::right << std::setw(10) << "Speedup"
              << std::right << std::setw(8)  << "vs CPU"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    int last_N = -1;
    for (const auto& r : results) {
        if (r.matrix_size != last_N && last_N != -1)
            std::cout << std::string(W, '-') << "\n";
        last_N = r.matrix_size;

        std::string size_str = std::to_string(r.matrix_size) + "x" +
                               std::to_string(r.matrix_size);

        if (r.time_ms < 0.0) {
            std::cout << std::left  << std::setw(20) << r.name
                      << std::left  << std::setw(14) << size_str
                      << std::right << std::setw(12) << "[skipped]"
                      << "\n";
        } else {
            std::cout << std::left  << std::setw(20) << r.name
                      << std::left  << std::setw(14) << size_str
                      << std::fixed << std::setprecision(2)
                      << std::right << std::setw(12) << r.time_ms
                      << std::right << std::setw(10) << r.gflops
                      << std::right << std::setw(9)  << r.speedup << "x"
                      << "\n";
        }
    }
    std::cout << std::string(W, '=') << "\n\n";
}

// =============================================================================
// Validation: verify CPU and GPU produce the same result
// =============================================================================
static void run_correctness_check() {
    std::cout << "\n  Correctness check (CPU naive == CPU optimised)...\n";
    Tensor A(64, 64), B(64, 64);
    A.randomize(); B.randomize();
    Tensor Cn = matmul_cpu_naive(A, B);
    Tensor Co = matmul_cpu_optimized(A, B);
    std::cout << "    CPU: " << (allclose(Cn, Co, 1e-3f) ? "PASS" : "FAIL") << "\n";

#ifdef USE_CUDA
    Tensor Cg = matmul_cuda_tiled(A, B);
    std::cout << "    GPU tiled vs CPU: " << (allclose(Co, Cg, 1e-3f) ? "PASS" : "FAIL") << "\n";
#endif
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════╗\n";
    std::cout << "  ║           Matrix Multiplication Benchmark            ║\n";
#ifdef USE_CUDA
    std::cout << "  ║           Mode: CUDA + CPU                          ║\n";
#else
    std::cout << "  ║           Mode: CPU-only                            ║\n";
#endif
    std::cout << "  ╚══════════════════════════════════════════════════════╝\n\n";

    // Parse sizes from command line, or use defaults
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        try { sizes.push_back(std::stoi(argv[i])); }
        catch (...) {}
    }
    if (sizes.empty()) sizes = {64, 128, 256, 512};

    std::cout << "  Matrix sizes:";
    for (int s : sizes) std::cout << " " << s;
    std::cout << "\n  Warmup=" << WARMUP << " runs=" << RUNS << "\n\n";

    run_correctness_check();
    std::cout << "\n  Running benchmarks:\n";

    std::vector<BenchmarkResult> all_results;
    for (int N : sizes) {
        benchmark_size(N, all_results);
    }

    print_full_table(all_results);

    // Memory bandwidth analysis
    std::cout << "  Memory analysis (for largest size N=" << sizes.back() << "):\n";
    int N = sizes.back();
    size_t bytes = 3ULL * N * N * sizeof(float);   // read A,B + write C
    std::cout << "    Data movement: " << bytes / (1024.0 * 1024.0) << " MB\n";

    // Find best time for largest N
    for (auto it = all_results.rbegin(); it != all_results.rend(); ++it) {
        if (it->matrix_size == N && it->time_ms > 0.0) {
            double bw = bytes / (it->time_ms * 1e-3) / 1e9;
            std::cout << "    Best achieved BW (" << it->name << "): "
                      << std::fixed << std::setprecision(1) << bw << " GB/s\n";
            break;
        }
    }

    return 0;
}
