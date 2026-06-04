// =============================================================================
// test_attention.cpp — GoogleTest tests for softmax, SDPA, and MHA
// =============================================================================

#include <gtest/gtest.h>
#include "tensor.hpp"
#include "attention.hpp"
#include "matmul.hpp"
#include <cmath>

static void expect_allclose(const Tensor& A, const Tensor& B, float tol=1e-4f) {
    ASSERT_EQ(A.rows, B.rows);
    ASSERT_EQ(A.cols, B.cols);
    for (int r = 0; r < A.rows; ++r)
        for (int c = 0; c < A.cols; ++c)
            EXPECT_NEAR(A(r,c), B(r,c), tol)
                << "at (" << r << "," << c << ")";
}

// =============================================================================
// Softmax
// =============================================================================
TEST(SoftmaxCPU, SingleRow_SumsToOne) {
    Tensor x(1, 4);
    x(0,0)=1.0f; x(0,1)=2.0f; x(0,2)=3.0f; x(0,3)=4.0f;
    Tensor y = softmax_cpu(x);
    float sum = 0.0f;
    for (int c = 0; c < 4; ++c) sum += y(0, c);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxCPU, MultipleRows_EachSumsToOne) {
    Tensor x(3, 5);
    x.randomize(-3.0f, 3.0f);
    Tensor y = softmax_cpu(x);
    for (int r = 0; r < 3; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < 5; ++c) sum += y(r, c);
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Row " << r << " sum != 1";
    }
}

TEST(SoftmaxCPU, AllElementsPositive) {
    Tensor x(2, 4);
    x.randomize(-10.0f, 10.0f);
    Tensor y = softmax_cpu(x);
    for (int i = 0; i < y.size(); ++i)
        EXPECT_GT(y.data[i], 0.0f) << "Element " << i << " is not positive";
}

TEST(SoftmaxCPU, NumericalStability_LargeValues) {
    // Without stability trick: exp(1000) = Inf → NaN
    // With subtract-max: exp(0) = 1.0 → safe
    Tensor x(1, 3);
    x(0,0)=1000.0f; x(0,1)=1001.0f; x(0,2)=999.0f;
    Tensor y = softmax_cpu(x);
    float sum = 0.0f;
    for (int c = 0; c < 3; ++c) sum += y(0,c);
    EXPECT_FALSE(std::isnan(sum)) << "NaN detected — numerical instability!";
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxCPU, UniformInput_EqualOutput) {
    // softmax([k,k,k,k]) = [0.25, 0.25, 0.25, 0.25]
    Tensor x(1, 4, 5.0f);
    Tensor y = softmax_cpu(x);
    for (int c = 0; c < 4; ++c)
        EXPECT_NEAR(y(0,c), 0.25f, 1e-5f);
}

TEST(SoftmaxCPU, OneHot_SharpPeak) {
    // softmax([0,0,100,0]) ≈ [0,0,1,0]
    Tensor x(1, 4, 0.0f);
    x(0, 2) = 100.0f;
    Tensor y = softmax_cpu(x);
    EXPECT_NEAR(y(0,2), 1.0f, 1e-3f);
    EXPECT_NEAR(y(0,0), 0.0f, 1e-3f);
}

// =============================================================================
// Scaled Dot-Product Attention
// =============================================================================
TEST(SDPA_CPU, OutputShape) {
    Tensor Q(3, 8), K(5, 8), V(5, 12);
    Q.randomize(); K.randomize(); V.randomize();
    Tensor out = sdpa_cpu(Q, K, V);
    EXPECT_EQ(out.rows, 3);    // seq_q
    EXPECT_EQ(out.cols, 12);   // d_v
}

TEST(SDPA_CPU, IdentityAttention_AllKeysEqual) {
    // When all K rows are identical, attention weights are uniform (1/seq_k).
    // Output should be the weighted average of V rows (all equal → same as V[0]).
    int seq_q=2, seq_k=4, d=4;
    Tensor Q(seq_q, d), K(seq_k, d), V(seq_k, d);
    // Fill K with same row
    for (int r = 0; r < seq_k; ++r)
        for (int c = 0; c < d; ++c)
            K(r,c) = 1.0f;
    // Fill V with same row
    for (int r = 0; r < seq_k; ++r)
        for (int c = 0; c < d; ++c)
            V(r,c) = static_cast<float>(c + 1);
    Q.randomize(-0.1f, 0.1f);

    Tensor out = sdpa_cpu(Q, K, V);
    // Output row should equal V[0] = [1,2,3,4]
    for (int r = 0; r < seq_q; ++r)
        for (int c = 0; c < d; ++c)
            EXPECT_NEAR(out(r,c), static_cast<float>(c+1), 1e-3f)
                << "Row " << r << " col " << c;
}

TEST(SDPA_CPU, DimensionMismatch_QK) {
    Tensor Q(3, 8), K(5, 6), V(5, 8);   // K has wrong d_k
    EXPECT_THROW(sdpa_cpu(Q, K, V), std::invalid_argument);
}

TEST(SDPA_CPU, DimensionMismatch_KV) {
    Tensor Q(3, 8), K(5, 8), V(4, 8);   // V has wrong seq_k
    EXPECT_THROW(sdpa_cpu(Q, K, V), std::invalid_argument);
}

TEST(SDPA_CPU, SelfAttention_Symmetric) {
    // Self-attention with Q==K==V: output should be a non-trivial mix of V rows
    Tensor X(4, 8);
    X.randomize(-1.0f, 1.0f);
    Tensor out = sdpa_cpu(X, X, X);
    // Just check shapes and no NaN
    EXPECT_EQ(out.rows, 4);
    EXPECT_EQ(out.cols, 8);
    for (int i = 0; i < out.size(); ++i)
        EXPECT_FALSE(std::isnan(out.data[i])) << "NaN at index " << i;
}

// =============================================================================
// Multi-Head Attention
// =============================================================================
TEST(MHA_CPU, OutputShape) {
    MultiHeadAttention mha(64, 4);
    Tensor inp(5, 64);
    inp.randomize();
    Tensor out = mha.forward(inp);
    EXPECT_EQ(out.rows, 5);
    EXPECT_EQ(out.cols, 64);
}

TEST(MHA_CPU, InvalidDivision) {
    // d_model=65 is not divisible by num_heads=4
    EXPECT_THROW(MultiHeadAttention mha(65, 4), std::invalid_argument);
}

TEST(MHA_CPU, NoNaN) {
    MultiHeadAttention mha(32, 4);
    Tensor inp(6, 32);
    inp.randomize(-1.0f, 1.0f);
    Tensor out = mha.forward(inp);
    for (int i = 0; i < out.size(); ++i)
        EXPECT_FALSE(std::isnan(out.data[i])) << "NaN at " << i;
}

TEST(MHA_CPU, SingleToken) {
    // Single-token forward (seq_len=1)
    MultiHeadAttention mha(16, 2);
    Tensor inp(1, 16);
    inp.randomize();
    Tensor out = mha.forward(inp);
    EXPECT_EQ(out.rows, 1);
    EXPECT_EQ(out.cols, 16);
}

// =============================================================================
// CUDA attention tests
// =============================================================================
#ifdef USE_CUDA

TEST(SoftmaxCUDA, MatchesCPU) {
    Tensor x(4, 16);
    x.randomize(-2.0f, 2.0f);
    Tensor cpu_out  = softmax_cpu(x);
    Tensor cuda_out = softmax_cuda(x);
    expect_allclose(cpu_out, cuda_out, 1e-4f);
}

TEST(SoftmaxCUDA, LargeValues) {
    Tensor x(2, 8);
    x(0,0)=500.0f; x(0,1)=501.0f;
    Tensor cpu_out  = softmax_cpu(x);
    Tensor cuda_out = softmax_cuda(x);
    expect_allclose(cpu_out, cuda_out, 1e-4f);
}

TEST(SDPA_CUDA, MatchesCPU) {
    Tensor Q(4, 16), K(4, 16), V(4, 16);
    Q.randomize(); K.randomize(); V.randomize();
    Tensor cpu_out  = sdpa_cpu(Q, K, V);
    Tensor cuda_out = sdpa_cuda(Q, K, V);
    expect_allclose(cpu_out, cuda_out, 1e-3f);
}

TEST(SDPA_CUDA, CrossAttention) {
    // Different seq lengths for Q and K/V
    Tensor Q(2, 8), K(6, 8), V(6, 8);
    Q.randomize(); K.randomize(); V.randomize();
    Tensor cpu_out  = sdpa_cpu(Q, K, V);
    Tensor cuda_out = sdpa_cuda(Q, K, V);
    expect_allclose(cpu_out, cuda_out, 1e-3f);
}

#endif // USE_CUDA
