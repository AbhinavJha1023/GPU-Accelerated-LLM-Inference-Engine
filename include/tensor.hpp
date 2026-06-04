#pragma once
// =============================================================================
// tensor.hpp — 2-D Tensor with optional CUDA device memory
// =============================================================================
// The Tensor class is the fundamental data container for this project.
// It stores data in row-major order in a std::vector<float> on the CPU,
// and (when compiled with USE_CUDA) mirrors that data to device memory.
//
// Layout:  element (r, c)  →  data[r * cols + c]
//
// GPU rule:  After calling to_gpu(), d_data holds the authoritative copy.
//            Call from_gpu() to synchronise back to the host vector.
// =============================================================================

#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

#ifdef USE_CUDA
#include <cuda_runtime.h>

// Throw a std::runtime_error on any CUDA API failure
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            throw std::runtime_error(                                            \
                std::string("[CUDA] ") + cudaGetErrorString(_e) +               \
                "  @" + __FILE__ + ":" + std::to_string(__LINE__));             \
        }                                                                        \
    } while (0)

#endif // USE_CUDA

// =============================================================================
class Tensor {
public:
    // ── CPU storage ──────────────────────────────────────────────────────────
    // rows/cols are declared before data so the constructor initialiser list
    // order matches declaration order (avoids -Wreorder warnings).
    int rows = 0;
    int cols = 0;
    std::vector<float> data;   // row-major flattened elements

#ifdef USE_CUDA
    // ── GPU storage (valid only when USE_CUDA is defined) ────────────────────
    float* d_data = nullptr;   // device pointer (nullptr if not allocated)
    bool   on_gpu = false;     // true ↔ d_data is allocated and up-to-date
#endif

    // ── Constructors ─────────────────────────────────────────────────────────
    Tensor() = default;
    explicit Tensor(int n);                      // n×n zero-filled square
    Tensor(int rows, int cols);                  // rows×cols zero-filled
    Tensor(int rows, int cols, float fill_val);  // rows×cols constant

    // ── Rule of Five ─────────────────────────────────────────────────────────
    ~Tensor();
    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // ── Element access ────────────────────────────────────────────────────────
    // Bounds-checked; throws std::out_of_range on bad indices.
    float&       operator()(int r, int c);
    const float& operator()(int r, int c) const;

    // ── Initialisation ────────────────────────────────────────────────────────
    void randomize(float lo = -1.0f, float hi = 1.0f);  // uniform random
    void zeros();                                         // fill with 0
    void fill(float val);                                 // fill with val
    void identity();                                      // I (must be square)

    // ── Shape helpers ─────────────────────────────────────────────────────────
    int         size()  const { return rows * cols; }
    std::string shape() const;                   // e.g. "[4 × 8]"

    // ── Linear algebra ────────────────────────────────────────────────────────
    Tensor transpose() const;

    // ── Printing ──────────────────────────────────────────────────────────────
    // Shows up to max_rows × max_cols elements; truncates with "..." otherwise.
    void print(const std::string& label = "",
               int max_rows = 8,
               int max_cols = 8) const;

#ifdef USE_CUDA
    // ── GPU memory management ─────────────────────────────────────────────────
    void to_gpu();       // allocate device mem + upload host → device
    void from_gpu();     // download device → host (keeps device allocation)
    void gpu_free();     // free device allocation (safe to call if nullptr)
    void sync_to_gpu();  // re-upload after host-side modifications
#endif
};

// =============================================================================
// Free-function helpers
// =============================================================================

Tensor zeros(int rows, int cols);
Tensor eye(int n);

// Element-wise arithmetic (shapes must match)
Tensor add(const Tensor& A, const Tensor& B);
Tensor sub(const Tensor& A, const Tensor& B);
Tensor elementwise_mul(const Tensor& A, const Tensor& B);

// Scale every element by a scalar
Tensor scale(const Tensor& A, float s);

// Flat dot product (both tensors treated as 1-D vectors, sizes must match)
float vdot(const Tensor& a, const Tensor& b);

// Returns true if every element of |A - B| < tol
bool allclose(const Tensor& A, const Tensor& B, float tol = 1e-4f);
