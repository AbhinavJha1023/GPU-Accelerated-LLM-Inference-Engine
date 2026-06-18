// =============================================================================
// test_quantize.cpp — GoogleTest tests for INT8 / INT4 quantization
// =============================================================================
#include <gtest/gtest.h>
#include "tensor.hpp"
#include "quantize.hpp"
#include "matmul.hpp"
#include <cmath>

// =============================================================================
// INT8
// =============================================================================
TEST(QuantizeInt8, ShapePreserved) {
    Tensor x(7, 13);
    x.randomize(-2.0f, 2.0f);
    QTensorInt8 q = quantize_int8(x);
    EXPECT_EQ(q.rows, 7);
    EXPECT_EQ(q.cols, 13);
    EXPECT_EQ(q.size(), 91);
}

TEST(QuantizeInt8, MemoryIsQuarterOfFP32) {
    Tensor x(64, 64);
    x.randomize();
    QTensorInt8 q = quantize_int8(x);
    size_t fp32_bytes = (size_t)x.size() * sizeof(float);   // 4 bytes each
    // INT8 payload is 1 byte each (plus a tiny scalar for the scale)
    EXPECT_EQ(q.data.size(), (size_t)x.size());
    EXPECT_LT(q.bytes(), fp32_bytes / 3);   // comfortably under 1/3
}

TEST(QuantizeInt8, RoundTripErrorBounded) {
    Tensor x(32, 32);
    x.randomize(-1.0f, 1.0f);
    QTensorInt8 q = quantize_int8(x);
    Tensor recon = dequantize_int8(q);
    QuantError e = quant_error(x, recon);
    // INT8 step size is max|x|/127 ≈ 1/127 ≈ 0.0079; error ≤ half a step.
    EXPECT_LT(e.max_abs_error, 0.01f);
}

TEST(QuantizeInt8, AllZeroTensorIsSafe) {
    Tensor x(8, 8, 0.0f);
    QTensorInt8 q = quantize_int8(x);   // must not divide by zero
    Tensor recon = dequantize_int8(q);
    for (int i = 0; i < recon.size(); ++i)
        EXPECT_FLOAT_EQ(recon.data[i], 0.0f);
}

TEST(QuantizeInt8, ValuesWithinRange) {
    Tensor x(16, 16);
    x.randomize(-5.0f, 5.0f);
    QTensorInt8 q = quantize_int8(x);
    for (int8_t v : q.data) {
        EXPECT_GE(v, -127);
        EXPECT_LE(v,  127);
    }
}

TEST(QuantizeInt8, MatmulMatchesDequantThenMatmul) {
    Tensor A(8, 16), W(16, 8);
    A.randomize(); W.randomize();
    QTensorInt8 Wq = quantize_int8(W);

    Tensor via_helper = matmul_int8(A, Wq);
    Tensor via_manual = matmul_cpu_optimized(A, dequantize_int8(Wq));
    ASSERT_EQ(via_helper.rows, via_manual.rows);
    for (int i = 0; i < via_helper.size(); ++i)
        EXPECT_NEAR(via_helper.data[i], via_manual.data[i], 1e-4f);
}

// =============================================================================
// INT4
// =============================================================================
TEST(QuantizeInt4, ShapePreserved) {
    Tensor x(5, 9);
    x.randomize();
    QTensorInt4 q = quantize_int4(x);
    EXPECT_EQ(q.rows, 5);
    EXPECT_EQ(q.cols, 9);
    EXPECT_EQ(q.count, 45);
}

TEST(QuantizeInt4, MemoryIsRoughlyEighthOfFP32) {
    Tensor x(64, 64);   // 4096 elements
    x.randomize();
    QTensorInt4 q = quantize_int4(x);
    // Two values per byte -> ceil(4096/2) = 2048 bytes payload
    EXPECT_EQ(q.packed.size(), (size_t)(x.size() + 1) / 2);
    size_t fp32_bytes = (size_t)x.size() * sizeof(float);
    EXPECT_LT(q.bytes(), fp32_bytes / 7);   // under 1/7 (≈1/8 + scale)
}

TEST(QuantizeInt4, RoundTripErrorBounded) {
    Tensor x(32, 32);
    x.randomize(-1.0f, 1.0f);
    QTensorInt4 q = quantize_int4(x);
    Tensor recon = dequantize_int4(q);
    QuantError e = quant_error(x, recon);
    // INT4 step size is max|x|/7 ≈ 1/7 ≈ 0.143; error ≤ half a step ≈ 0.072.
    EXPECT_LT(e.max_abs_error, 0.10f);
}

TEST(QuantizeInt4, OddCountPacksCorrectly) {
    // 3x3 = 9 elements (odd) -> ceil(9/2) = 5 bytes, last nibble unused
    Tensor x(3, 3);
    x.randomize();
    QTensorInt4 q = quantize_int4(x);
    EXPECT_EQ(q.packed.size(), 5u);
    Tensor recon = dequantize_int4(q);
    EXPECT_EQ(recon.size(), 9);
}

TEST(QuantizeInt4, Int8MoreAccurateThanInt4) {
    Tensor x(64, 64);
    x.randomize(-3.0f, 3.0f);
    QuantError e8 = quant_error(x, dequantize_int8(quantize_int8(x)));
    QuantError e4 = quant_error(x, dequantize_int4(quantize_int4(x)));
    // More bits -> smaller error
    EXPECT_LT(e8.rmse, e4.rmse);
}
