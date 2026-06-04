// =============================================================================
// tensor.cpp — Tensor class implementation
// =============================================================================
#include "tensor.hpp"
#include <cstring>
#include <cassert>
#include <stdexcept>

// =============================================================================
// Constructors
// =============================================================================

Tensor::Tensor(int n)
    : rows(n), cols(n), data(static_cast<size_t>(n) * n, 0.0f) {}

Tensor::Tensor(int r, int c)
    : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0f) {}

Tensor::Tensor(int r, int c, float v)
    : rows(r), cols(c), data(static_cast<size_t>(r) * c, v) {}

// =============================================================================
// Destructor
// =============================================================================
Tensor::~Tensor() {
#ifdef USE_CUDA
    gpu_free();
#endif
}

// =============================================================================
// Copy constructor
// =============================================================================
Tensor::Tensor(const Tensor& o)
    : rows(o.rows), cols(o.cols), data(o.data)
{
#ifdef USE_CUDA
    d_data = nullptr;
    on_gpu = false;
    if (o.on_gpu) {
        size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMemcpy(d_data, o.d_data, bytes, cudaMemcpyDeviceToDevice));
        on_gpu = true;
    }
#endif
}

// =============================================================================
// Copy assignment
// =============================================================================
Tensor& Tensor::operator=(const Tensor& o) {
    if (this == &o) return *this;
#ifdef USE_CUDA
    gpu_free();
#endif
    rows = o.rows;
    cols = o.cols;
    data = o.data;
#ifdef USE_CUDA
    d_data = nullptr;
    on_gpu = false;
    if (o.on_gpu) {
        size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMemcpy(d_data, o.d_data, bytes, cudaMemcpyDeviceToDevice));
        on_gpu = true;
    }
#endif
    return *this;
}

// =============================================================================
// Move constructor
// =============================================================================
Tensor::Tensor(Tensor&& o) noexcept
    : rows(o.rows), cols(o.cols), data(std::move(o.data))
{
#ifdef USE_CUDA
    d_data   = o.d_data;
    on_gpu   = o.on_gpu;
    o.d_data = nullptr;
    o.on_gpu = false;
#endif
    o.rows = 0;
    o.cols = 0;
}

// =============================================================================
// Move assignment
// =============================================================================
Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this == &o) return *this;
#ifdef USE_CUDA
    gpu_free();
    d_data   = o.d_data;
    on_gpu   = o.on_gpu;
    o.d_data = nullptr;
    o.on_gpu = false;
#endif
    rows = o.rows;
    cols = o.cols;
    data = std::move(o.data);
    o.rows = 0;
    o.cols = 0;
    return *this;
}

// =============================================================================
// Element access — bounds-checked
// =============================================================================
float& Tensor::operator()(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range(
            "Tensor(" + std::to_string(r) + "," + std::to_string(c) +
            ") out of range for " + shape());
    }
    return data[r * cols + c];
}

const float& Tensor::operator()(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::out_of_range(
            "Tensor(" + std::to_string(r) + "," + std::to_string(c) +
            ") out of range for " + shape());
    }
    return data[r * cols + c];
}

// =============================================================================
// Initialisation
// =============================================================================
void Tensor::randomize(float lo, float hi) {
    // Fixed seed for reproducibility in tests; change seed for variety
    static std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& v : data) v = dist(rng);
}

void Tensor::zeros() {
    std::fill(data.begin(), data.end(), 0.0f);
}

void Tensor::fill(float val) {
    std::fill(data.begin(), data.end(), val);
}

void Tensor::identity() {
    if (rows != cols)
        throw std::invalid_argument("identity() requires a square tensor, got " + shape());
    zeros();
    for (int i = 0; i < rows; ++i)
        data[i * cols + i] = 1.0f;
}

// =============================================================================
// Shape string
// =============================================================================
std::string Tensor::shape() const {
    return "[" + std::to_string(rows) + " x " + std::to_string(cols) + "]";
}

// =============================================================================
// Transpose — creates a new tensor (O(rows*cols) copy)
// =============================================================================
Tensor Tensor::transpose() const {
    Tensor result(cols, rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            result.data[c * rows + r] = data[r * cols + c];
    return result;
}

// =============================================================================
// Print
// =============================================================================
void Tensor::print(const std::string& label, int max_rows, int max_cols) const {
    if (!label.empty())
        std::cout << label << " " << shape() << ":\n";
    else
        std::cout << "Tensor " << shape() << ":\n";

    int sr = std::min(rows, max_rows);
    int sc = std::min(cols, max_cols);

    for (int r = 0; r < sr; ++r) {
        std::cout << "  [ ";
        for (int c = 0; c < sc; ++c) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(9) << data[r * cols + c];
            if (c < sc - 1) std::cout << ", ";
        }
        if (sc < cols) std::cout << " ...";
        std::cout << " ]\n";
    }
    if (sr < rows) std::cout << "  [ ... ]\n";
    std::cout << std::flush;
}

// =============================================================================
// GPU memory management (only compiled when USE_CUDA is defined)
// =============================================================================
#ifdef USE_CUDA

void Tensor::to_gpu() {
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    if (!on_gpu) {
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        on_gpu = true;
    }
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice));
}

void Tensor::from_gpu() {
    if (!on_gpu || d_data == nullptr) return;
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    CUDA_CHECK(cudaMemcpy(data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
}

void Tensor::gpu_free() {
    if (d_data) {
        cudaFree(d_data);   // not CUDA_CHECK — destructor must not throw
        d_data = nullptr;
        on_gpu = false;
    }
}

void Tensor::sync_to_gpu() {
    if (!on_gpu) {
        to_gpu();
        return;
    }
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice));
}

#endif // USE_CUDA

// =============================================================================
// Free helpers
// =============================================================================

Tensor zeros(int rows, int cols) {
    return Tensor(rows, cols, 0.0f);
}

Tensor eye(int n) {
    Tensor t(n, n, 0.0f);
    for (int i = 0; i < n; ++i) t.data[i * n + i] = 1.0f;
    return t;
}

Tensor add(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols)
        throw std::invalid_argument("add: shape mismatch " + A.shape() + " vs " + B.shape());
    Tensor C(A.rows, A.cols);
    for (int i = 0; i < A.size(); ++i)
        C.data[i] = A.data[i] + B.data[i];
    return C;
}

Tensor sub(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols)
        throw std::invalid_argument("sub: shape mismatch " + A.shape() + " vs " + B.shape());
    Tensor C(A.rows, A.cols);
    for (int i = 0; i < A.size(); ++i)
        C.data[i] = A.data[i] - B.data[i];
    return C;
}

Tensor elementwise_mul(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols)
        throw std::invalid_argument("elementwise_mul: shape mismatch");
    Tensor C(A.rows, A.cols);
    for (int i = 0; i < A.size(); ++i)
        C.data[i] = A.data[i] * B.data[i];
    return C;
}

Tensor scale(const Tensor& A, float s) {
    Tensor C(A.rows, A.cols);
    for (int i = 0; i < A.size(); ++i)
        C.data[i] = A.data[i] * s;
    return C;
}

float vdot(const Tensor& a, const Tensor& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("vdot: size mismatch");
    float s = 0.0f;
    for (int i = 0; i < a.size(); ++i)
        s += a.data[i] * b.data[i];
    return s;
}

bool allclose(const Tensor& A, const Tensor& B, float tol) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (int i = 0; i < A.size(); ++i) {
        if (std::abs(A.data[i] - B.data[i]) > tol) return false;
    }
    return true;
}
