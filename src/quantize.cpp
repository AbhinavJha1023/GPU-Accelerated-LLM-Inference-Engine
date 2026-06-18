// =============================================================================
// quantize.cpp — INT8 / INT4 weight quantization implementation
// =============================================================================
#include "quantize.hpp"
#include "matmul.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// =============================================================================
// Helper: find max absolute value (the quantization range)
// =============================================================================
static float max_abs(const Tensor& x) {
    float m = 0.0f;
    for (float v : x.data) m = std::max(m, std::fabs(v));
    return m;
}

// =============================================================================
// INT8 quantization (symmetric, per-tensor)
// =============================================================================
//   qmax  = 127
//   scale = max|x| / 127
//   q_i   = clamp(round(x_i / scale), -127, 127)
QTensorInt8 quantize_int8(const Tensor& x) {
    QTensorInt8 out;
    out.rows = x.rows;
    out.cols = x.cols;
    out.data.resize(x.size());

    const float m = max_abs(x);
    // Guard against an all-zero tensor (scale would be 0 -> division by zero)
    out.scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
    const float inv = 1.0f / out.scale;

    for (int i = 0; i < x.size(); ++i) {
        int q = static_cast<int>(std::lround(x.data[i] * inv));
        q = std::clamp(q, -127, 127);
        out.data[i] = static_cast<int8_t>(q);
    }
    return out;
}

Tensor dequantize_int8(const QTensorInt8& q) {
    Tensor out(q.rows, q.cols);
    for (int i = 0; i < q.size(); ++i)
        out.data[i] = static_cast<float>(q.data[i]) * q.scale;
    return out;
}

// =============================================================================
// INT4 quantization (symmetric, per-tensor, 2 values packed per byte)
// =============================================================================
//   qmax  = 7   (4-bit signed range is [-8, 7]; we use [-7,7] for symmetry)
//   scale = max|x| / 7
//
// Packing: each byte stores two values.
//   low  nibble = value at even index   (stored as unsigned 0..15 via +8 bias)
//   high nibble = value at odd index
QTensorInt4 quantize_int4(const Tensor& x) {
    QTensorInt4 out;
    out.rows  = x.rows;
    out.cols  = x.cols;
    out.count = x.size();
    out.packed.resize((x.size() + 1) / 2, 0);

    const float m = max_abs(x);
    out.scale = (m > 0.0f) ? (m / 7.0f) : 1.0f;
    const float inv = 1.0f / out.scale;

    for (int i = 0; i < x.size(); ++i) {
        int q = static_cast<int>(std::lround(x.data[i] * inv));
        q = std::clamp(q, -7, 7);
        // Bias by +8 so the value fits an unsigned 4-bit nibble (range 1..15)
        uint8_t nibble = static_cast<uint8_t>(q + 8) & 0x0F;
        if ((i & 1) == 0) {
            out.packed[i / 2] |= nibble;          // low nibble
        } else {
            out.packed[i / 2] |= (nibble << 4);   // high nibble
        }
    }
    return out;
}

Tensor dequantize_int4(const QTensorInt4& q) {
    Tensor out(q.rows, q.cols);
    for (int i = 0; i < q.count; ++i) {
        uint8_t byte = q.packed[i / 2];
        uint8_t nibble = ((i & 1) == 0) ? (byte & 0x0F) : (byte >> 4);
        int val = static_cast<int>(nibble) - 8;   // undo the +8 bias
        out.data[i] = static_cast<float>(val) * q.scale;
    }
    return out;
}

// =============================================================================
// Error metrics
// =============================================================================
QuantError quant_error(const Tensor& a, const Tensor& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("quant_error: size mismatch");

    float max_e = 0.0f, sum_abs = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < a.size(); ++i) {
        float e = std::fabs(a.data[i] - b.data[i]);
        max_e   = std::max(max_e, e);
        sum_abs += e;
        sum_sq  += e * e;
    }
    int n = a.size();
    return QuantError{
        max_e,
        n ? sum_abs / n : 0.0f,
        n ? std::sqrt(sum_sq / n) : 0.0f
    };
}

// =============================================================================
// Quantized matmul: A (FP32) x B (INT8) -> C (FP32)
// =============================================================================
// We reconstruct B in FP32 and reuse the optimised CPU matmul. This keeps the
// result numerically identical to "dequantize then matmul", which is the
// correctness contract we test against.
Tensor matmul_int8(const Tensor& A, const QTensorInt8& B) {
    Tensor Bf = dequantize_int8(B);
    return matmul_cpu_optimized(A, Bf);
}
