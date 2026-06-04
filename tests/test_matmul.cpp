// =============================================================================
// test_matmul.cpp — GoogleTest unit tests for matrix multiplication
// =============================================================================
// Run:  ctest --output-on-failure
//   or:  ./test_matmul
// =============================================================================

#include <gtest/gtest.h>
#include "tensor.hpp"
#include "matmul.hpp"
#include <cmath>

// Helper: fill a 2×2 tensor with known values
static Tensor make_2x2(float a, float b, float c, float d) {
    Tensor T(2, 2);
    T(0,0)=a; T(0,1)=b; T(1,0)=c; T(1,1)=d;
    return T;
}

// Helper: check two tensors element-wise within tolerance
static void expect_allclose(const Tensor& A, const Tensor& B,
                            float tol = 1e-4f) {
    ASSERT_EQ(A.rows, B.rows) << "Row mismatch";
    ASSERT_EQ(A.cols, B.cols) << "Col mismatch";
    for (int r = 0; r < A.rows; ++r)
        for (int c = 0; c < A.cols; ++c)
            EXPECT_NEAR(A(r,c), B(r,c), tol)
                << "Mismatch at (" << r << "," << c << ")";
}

// =============================================================================
// Tensor operations
// =============================================================================
TEST(TensorTest, DefaultConstructor) {
    Tensor T;
    EXPECT_EQ(T.rows, 0);
    EXPECT_EQ(T.cols, 0);
    EXPECT_EQ(T.size(), 0);
}

TEST(TensorTest, ShapeConstructor) {
    Tensor T(3, 5);
    EXPECT_EQ(T.rows, 3);
    EXPECT_EQ(T.cols, 5);
    EXPECT_EQ(T.size(), 15);
    // Should be zero-filled
    for (int i = 0; i < T.size(); ++i)
        EXPECT_FLOAT_EQ(T.data[i], 0.0f);
}

TEST(TensorTest, FillConstructor) {
    Tensor T(2, 3, 7.0f);
    for (int i = 0; i < T.size(); ++i)
        EXPECT_FLOAT_EQ(T.data[i], 7.0f);
}

TEST(TensorTest, Indexing) {
    Tensor T(3, 4);
    T(0,0)=1.0f; T(1,2)=5.5f; T(2,3)=9.9f;
    EXPECT_FLOAT_EQ(T(0,0), 1.0f);
    EXPECT_FLOAT_EQ(T(1,2), 5.5f);
    EXPECT_FLOAT_EQ(T(2,3), 9.9f);
    EXPECT_FLOAT_EQ(T(0,1), 0.0f);   // untouched — still 0
}

TEST(TensorTest, IndexingOutOfBounds) {
    Tensor T(3, 3);
    EXPECT_THROW(T(3, 0), std::out_of_range);
    EXPECT_THROW(T(0, 3), std::out_of_range);
    EXPECT_THROW(T(-1, 0), std::out_of_range);
}

TEST(TensorTest, Transpose) {
    Tensor T(2, 3);
    T(0,0)=1; T(0,1)=2; T(0,2)=3;
    T(1,0)=4; T(1,1)=5; T(1,2)=6;
    Tensor Tt = T.transpose();
    EXPECT_EQ(Tt.rows, 3);
    EXPECT_EQ(Tt.cols, 2);
    EXPECT_FLOAT_EQ(Tt(0,0), 1.0f);
    EXPECT_FLOAT_EQ(Tt(0,1), 4.0f);
    EXPECT_FLOAT_EQ(Tt(2,0), 3.0f);
    EXPECT_FLOAT_EQ(Tt(2,1), 6.0f);
}

TEST(TensorTest, Identity) {
    Tensor I = eye(4);
    EXPECT_EQ(I.rows, 4);
    EXPECT_EQ(I.cols, 4);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_FLOAT_EQ(I(r,c), (r == c) ? 1.0f : 0.0f);
}

TEST(TensorTest, AddSub) {
    Tensor A(2,2,1.0f), B(2,2,2.0f);
    Tensor C = add(A, B);
    Tensor D = sub(B, A);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(C.data[i], 3.0f);
        EXPECT_FLOAT_EQ(D.data[i], 1.0f);
    }
}

TEST(TensorTest, Scale) {
    Tensor A(2, 3, 3.0f);
    Tensor B = scale(A, 4.0f);
    for (int i = 0; i < B.size(); ++i)
        EXPECT_FLOAT_EQ(B.data[i], 12.0f);
}

TEST(TensorTest, Allclose) {
    Tensor A(2, 2, 1.0f);
    Tensor B(2, 2, 1.0f);
    EXPECT_TRUE(allclose(A, B));
    B(0,0) = 1.5f;
    EXPECT_FALSE(allclose(A, B, 0.1f));
    EXPECT_TRUE(allclose(A, B, 1.0f));
}

// =============================================================================
// Naive CPU matmul
// =============================================================================
TEST(MatmulCPUNaive, Identity_2x2) {
    // A × I = A
    Tensor A = make_2x2(1,2,3,4);
    Tensor I = eye(2);
    Tensor C = matmul_cpu_naive(A, I);
    expect_allclose(A, C);
}

TEST(MatmulCPUNaive, Known_2x2) {
    // [1,2;3,4] × [5,6;7,8] = [19,22;43,50]
    Tensor A = make_2x2(1,2,3,4);
    Tensor B = make_2x2(5,6,7,8);
    Tensor C = matmul_cpu_naive(A, B);
    EXPECT_NEAR(C(0,0), 19.0f, 1e-4f);
    EXPECT_NEAR(C(0,1), 22.0f, 1e-4f);
    EXPECT_NEAR(C(1,0), 43.0f, 1e-4f);
    EXPECT_NEAR(C(1,1), 50.0f, 1e-4f);
}

TEST(MatmulCPUNaive, NonSquare_2x3_times_3x4) {
    // A[2×3] × B[3×4] → C[2×4]
    Tensor A(2, 3), B(3, 4);
    for (int i = 0; i < 6; ++i)  A.data[i] = static_cast<float>(i+1);
    for (int i = 0; i < 12; ++i) B.data[i] = static_cast<float>(i+1);
    // A = [[1,2,3],[4,5,6]]
    // B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    // C[0][0] = 1*1+2*5+3*9 = 1+10+27 = 38
    Tensor C = matmul_cpu_naive(A, B);
    EXPECT_EQ(C.rows, 2);
    EXPECT_EQ(C.cols, 4);
    EXPECT_NEAR(C(0,0), 38.0f, 1e-4f);
}

TEST(MatmulCPUNaive, DimensionMismatch) {
    Tensor A(2, 3), B(4, 2);   // incompatible
    EXPECT_THROW(matmul_cpu_naive(A, B), std::invalid_argument);
}

TEST(MatmulCPUNaive, Zeros) {
    Tensor A(4, 4, 0.0f), B(4, 4, 5.0f);
    Tensor C = matmul_cpu_naive(A, B);
    for (int i = 0; i < 16; ++i)
        EXPECT_FLOAT_EQ(C.data[i], 0.0f);
}

// =============================================================================
// Optimised CPU matmul — results must match naive exactly
// =============================================================================
TEST(MatmulCPUOptimised, MatchesNaive_Random_64x64) {
    Tensor A(64, 64), B(64, 64);
    A.randomize(-1.0f, 1.0f);
    B.randomize(-1.0f, 1.0f);
    Tensor C_naive = matmul_cpu_naive(A, B);
    Tensor C_optim = matmul_cpu_optimized(A, B);
    expect_allclose(C_naive, C_optim, 1e-3f);
}

TEST(MatmulCPUOptimised, MatchesNaive_NonSquare) {
    Tensor A(17, 33), B(33, 25);
    A.randomize(); B.randomize();
    Tensor C_naive = matmul_cpu_naive(A, B);
    Tensor C_optim = matmul_cpu_optimized(A, B);
    expect_allclose(C_naive, C_optim, 1e-3f);
}

TEST(MatmulCPUOptimised, MatchesNaive_Large) {
    Tensor A(128, 128), B(128, 128);
    A.randomize(); B.randomize();
    Tensor C_naive = matmul_cpu_naive(A, B);
    Tensor C_optim = matmul_cpu_optimized(A, B);
    expect_allclose(C_naive, C_optim, 1e-2f);
}

// =============================================================================
// CUDA matmul tests (only built when USE_CUDA is defined)
// =============================================================================
#ifdef USE_CUDA

TEST(MatmulCUDANaive, MatchesCPU_32x32) {
    Tensor A(32, 32), B(32, 32);
    A.randomize(); B.randomize();
    Tensor C_cpu  = matmul_cpu_optimized(A, B);
    Tensor C_cuda = matmul_cuda_naive(A, B);
    expect_allclose(C_cpu, C_cuda, 1e-3f);
}

TEST(MatmulCUDATiled, MatchesCPU_64x64) {
    Tensor A(64, 64), B(64, 64);
    A.randomize(); B.randomize();
    Tensor C_cpu  = matmul_cpu_optimized(A, B);
    Tensor C_cuda = matmul_cuda_tiled(A, B);
    expect_allclose(C_cpu, C_cuda, 1e-3f);
}

TEST(MatmulCUDATiled, MatchesCPU_NonSquare) {
    Tensor A(48, 64), B(64, 32);
    A.randomize(); B.randomize();
    Tensor C_cpu  = matmul_cpu_optimized(A, B);
    Tensor C_cuda = matmul_cuda_tiled(A, B);
    expect_allclose(C_cpu, C_cuda, 1e-3f);
}

#endif // USE_CUDA
