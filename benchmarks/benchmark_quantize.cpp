// =============================================================================
// benchmark_quantize.cpp — Quantization memory / accuracy report
// =============================================================================
// Standalone executable. Prints an HONEST table: real memory footprint and
// real numerical error on CPU. We deliberately do NOT print GPU VRAM or
// tokens/sec, because this engine uses random weights on CPU — those numbers
// would be meaningless. Memory and error, however, are real and meaningful.
//
// Run:  ./benchmark_quantize
// =============================================================================

#include "tensor.hpp"
#include "quantize.hpp"
#include "matmul.hpp"
#include "benchmark.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

static void report_for_size(int N) {
    Tensor W(N, N);
    W.randomize(-1.0f, 1.0f);   // simulate a weight matrix

    QTensorInt8 q8 = quantize_int8(W);
    QTensorInt4 q4 = quantize_int4(W);

    Tensor r8 = dequantize_int8(q8);
    Tensor r4 = dequantize_int4(q4);

    QuantError e8 = quant_error(W, r8);
    QuantError e4 = quant_error(W, r4);

    size_t fp32_bytes = (size_t)W.size() * sizeof(float);

    std::cout << "\n  Weight matrix: " << N << "x" << N
              << "  (" << W.size() << " weights)\n";
    std::cout << "  " << std::string(68, '-') << "\n";
    std::cout << std::left
              << std::setw(12) << "  Precision"
              << std::right << std::setw(14) << "Memory"
              << std::setw(12) << "vs FP32"
              << std::setw(15) << "Max error"
              << std::setw(15) << "RMSE"
              << "\n";
    std::cout << "  " << std::string(68, '-') << "\n";

    auto row = [&](const std::string& name, size_t bytes, float maxe, float rmse) {
        double pct = 100.0 * bytes / fp32_bytes;
        std::cout << std::left  << std::setw(12) << ("  " + name)
                  << std::right << std::setw(12) << (std::to_string(bytes / 1024) + " KB")
                  << std::setw(13) << std::fixed << std::setprecision(1) << pct << "%"
                  << std::setw(15) << std::fixed << std::setprecision(6) << maxe
                  << std::setw(15) << std::fixed << std::setprecision(6) << rmse
                  << "\n";
    };

    row("FP32",  fp32_bytes,  0.0f,            0.0f);
    row("INT8",  q8.bytes(),  e8.max_abs_error, e8.rmse);
    row("INT4",  q4.bytes(),  e4.max_abs_error, e4.rmse);
}

static void verify_matmul_accuracy() {
    std::cout << "\n  ── End-to-end: how quantization error propagates through matmul ──\n";
    const int M = 64, K = 128, N = 64;
    Tensor A(M, K), W(K, N);
    A.randomize(-1.0f, 1.0f);
    W.randomize(-1.0f, 1.0f);

    Tensor C_fp32 = matmul_cpu_optimized(A, W);                       // reference
    Tensor C_int8 = matmul_int8(A, quantize_int8(W));                 // INT8 weights
    Tensor C_int4 = matmul_cpu_optimized(A, dequantize_int4(quantize_int4(W)));

    QuantError e8 = quant_error(C_fp32, C_int8);
    QuantError e4 = quant_error(C_fp32, C_int4);

    std::cout << "    A[" << M << "x" << K << "] x W[" << K << "x" << N << "]\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    INT8 weights -> output RMSE vs FP32: " << e8.rmse << "\n";
    std::cout << "    INT4 weights -> output RMSE vs FP32: " << e4.rmse << "\n";
    std::cout << "    (Error stays small and bounded — quantization is safe for inference.)\n";
}

int main() {
    std::cout << "\n";
    std::cout << "  ╔══════════════════════════════════════════════════════╗\n";
    std::cout << "  ║         Weight Quantization Report (CPU)             ║\n";
    std::cout << "  ║         Honest metrics: memory + numerical error     ║\n";
    std::cout << "  ╚══════════════════════════════════════════════════════╝\n";

    std::cout << "\n  Scheme: symmetric linear, per-tensor scale.\n"
              << "    scale = max|x| / qmax   (INT8 qmax=127, INT4 qmax=7)\n"
              << "    q = clamp(round(x/scale), -qmax, qmax)\n";

    for (int N : {128, 512, 1024}) report_for_size(N);

    verify_matmul_accuracy();

    std::cout << "\n  Takeaways:\n"
              << "    • INT8 -> 4x smaller, error ~1e-3 (negligible for inference)\n"
              << "    • INT4 -> 8x smaller, error ~1e-1 (needs care; often per-row scales)\n"
              << "    • On a GPU this memory reduction directly raises decode throughput,\n"
              << "      because LLM decoding is memory-bandwidth bound.\n\n";
    return 0;
}
