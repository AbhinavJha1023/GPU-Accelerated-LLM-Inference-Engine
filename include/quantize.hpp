#pragma once
// =============================================================================
// quantize.hpp — Weight quantization (FP32 -> INT8 / INT4)
// =============================================================================
// WHY QUANTIZATION
// ─────────────────
// LLM weights dominate memory. Storing them in fewer bits shrinks the model
// and reduces memory bandwidth (the real bottleneck in decoding), at the cost
// of a small, bounded numerical error.
//
//   FP32 : 4 bytes/weight  (baseline)
//   INT8 : 1 byte/weight   -> 4x smaller
//   INT4 : 0.5 byte/weight -> 8x smaller (two weights packed per byte)
//
// SCHEME: symmetric linear quantization, per-tensor scale.
//   scale = max(|x|) / qmax           (qmax = 127 for INT8, 7 for INT4)
//   q     = clamp(round(x / scale), -qmax, qmax)
//   x'    = q * scale                 (dequantization)
//
// "Symmetric" means zero maps exactly to 0 (no zero-point offset), which keeps
// the math simple and is the standard choice for weight quantization.
//
// HONEST METRICS (this project, CPU):
//   We report MEMORY footprint and NUMERICAL ERROR — not GPU VRAM or tok/s,
//   because the model uses random weights and runs on CPU. The memory and
//   error numbers are real and meaningful.
// =============================================================================

#include "tensor.hpp"
#include <vector>
#include <cstdint>

// =============================================================================
// INT8 quantized tensor (per-tensor symmetric scale)
// =============================================================================
struct QTensorInt8 {
    std::vector<int8_t> data;   // quantized values in [-127, 127]
    float scale = 1.0f;         // x ≈ q * scale
    int   rows  = 0;
    int   cols  = 0;

    int    size()  const { return rows * cols; }
    size_t bytes() const { return data.size() * sizeof(int8_t) + sizeof(float); }
};

// =============================================================================
// INT4 quantized tensor (per-tensor symmetric scale, 2 values packed per byte)
// =============================================================================
struct QTensorInt4 {
    std::vector<uint8_t> packed;  // each byte holds two 4-bit signed values
    float scale = 1.0f;           // x ≈ q * scale
    int   rows  = 0;
    int   cols  = 0;
    int   count = 0;              // logical element count (rows*cols)

    int    size()  const { return count; }
    size_t bytes() const { return packed.size() * sizeof(uint8_t) + sizeof(float); }
};

// =============================================================================
// Quantize / Dequantize
// =============================================================================
QTensorInt8 quantize_int8(const Tensor& x);
Tensor      dequantize_int8(const QTensorInt8& q);

QTensorInt4 quantize_int4(const Tensor& x);
Tensor      dequantize_int4(const QTensorInt4& q);

// =============================================================================
// Error metrics (real, CPU-measurable)
// =============================================================================
struct QuantError {
    float max_abs_error;   // max_i |x_i - x'_i|
    float mean_abs_error;  // mean_i |x_i - x'_i|
    float rmse;            // sqrt(mean_i (x_i - x'_i)^2)
};

QuantError quant_error(const Tensor& original, const Tensor& reconstructed);

// =============================================================================
// Quantized matmul: C = A (FP32) x B (INT8 weights)
// =============================================================================
// Dequantizes B on the fly and multiplies. Demonstrates how a quantized weight
// matrix is consumed at inference time. (A production INT8 kernel would do
// integer accumulation; here we dequantize for clarity and correctness.)
Tensor matmul_int8(const Tensor& A, const QTensorInt8& B);
