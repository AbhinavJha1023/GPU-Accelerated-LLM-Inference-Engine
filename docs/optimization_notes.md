# Optimization Notes — GPU-Accelerated LLM Inference Engine

## Optimization Levels Implemented

This project implements four levels of optimization, each teaching a
distinct concept in high-performance computing.

---

## Optimization 1 — CPU Cache-Friendly Blocking

**File:** `src/matmul_cpu.cpp` → `matmul_cpu_optimized()`

**Problem:** The naive triple loop accesses matrix B in stride-N order —
every inner iteration jumps `N × 4` bytes, thrashing the L1 cache.

**Solution:** Partition the computation into `BLOCK_SIZE × BLOCK_SIZE` tiles
that fit in L1 cache (32 KB on most modern CPUs).

```cpp
// Outer loops: select tiles
for (int m0 = 0; m0 < M; m0 += BLOCK) {
  for (int n0 = 0; n0 < N; n0 += BLOCK) {
    for (int k0 = 0; k0 < K; k0 += BLOCK) {

      // Inner loops: compute within tile — all accesses hit L1
      for (int m = m0; m < min(m0+BLOCK, M); ++m) {
        for (int k = k0; k < min(k0+BLOCK, K); ++k) {
          float a_mk = A[m][k];         // load once, reuse BLOCK times
          for (int n = n0; n < min(n0+BLOCK, N); ++n) {
            C[m][n] += a_mk * B[k][n]; // sequential → cache line hits
          }
        }
      }
    }
  }
}
```

**Why `BLOCK_SIZE = 64`:**
- A-tile: 64×64 × 4B = 16 KB
- B-tile: 64×64 × 4B = 16 KB
- Total: 32 KB — fits exactly in most L1 caches
- If BLOCK is too large → tile spills to L2, losing the benefit

**Measured speedup:** 4–10× over naive on modern x86 CPUs.

---

## Optimization 2 — CUDA Shared Memory Tiling

**File:** `src/matmul_cuda.cu` → `matmul_tiled_kernel`

**Problem:** Each thread in the naive CUDA kernel independently loads K
elements from global memory (200-cycle latency).

**Solution:** A 16×16 block of threads cooperatively loads a 16×16 tile of A
and B into `__shared__` memory (4-cycle latency) before computing.

**Shared memory as a software-managed cache:**
```
Global memory bandwidth: ~2 TB/s (A100)
Shared memory bandwidth: ~19 TB/s (A100, no bank conflicts)
Ratio: ~9.5× — this is the theoretical gain from shared memory
```

**Bank conflict avoidance:**
Shared memory is organised into 32 banks (one float per bank per address).
If all 32 threads in a warp access different banks → no conflict (best case).

In our kernel:
- `As[ty][tx]`: threads in same warp share ty, vary tx → each accesses a
  different bank column ✓
- `Bs[ty][tx]`: same reasoning ✓

No bank conflicts → full shared memory bandwidth utilised.

**`__syncthreads()` — why two barriers per tile:**
```
Load As and Bs from global mem
  ↓
__syncthreads()   ← barrier 1: all loads complete before any thread reads
  ↓
Compute partial dot product using As, Bs
  ↓
__syncthreads()   ← barrier 2: all computes complete before next tile load
                    (prevents fast threads from overwriting As/Bs before
                     slower threads finish reading them)
  ↓
Advance to next tile
```

---

## Optimization 3 — Loop Unrolling

**File:** `src/matmul_cuda.cu`, `src/attention_cuda.cu`

**Applied as:** `#pragma unroll` on the inner TILE loop.

```cuda
#pragma unroll
for (int k = 0; k < TILE; ++k) {   // TILE=16, known at compile time
    acc += As[ty][k] * Bs[k][tx];
}
```

**What the compiler generates** (conceptually):
```cuda
// No loop — 16 explicit FMA instructions:
acc += As[ty][0] * Bs[0][tx];
acc += As[ty][1] * Bs[1][tx];
acc += As[ty][2] * Bs[2][tx];
// ... × 16
```

**Benefits:**
1. Eliminates `k < TILE` comparison per iteration (branch removal)
2. Eliminates `k++` increment per iteration
3. Enables instruction-level parallelism: GPU can issue multiple FMAs
   while waiting for shared memory latency
4. Better register allocation: compiler can freely reorder loads/stores

**When to unroll:** Only when the trip count is small and known at compile
time. Unrolling large loops increases code size → instruction cache pressure.

---

## Optimization 4 — Memory Coalescing

**Definition:** A warp (32 threads) accesses global memory in one transaction
if the 32 addresses are consecutive and aligned. This is "coalesced access".

**In `matmul_naive_kernel`:**
```
Warp threads: tx = 0, 1, 2, ..., 31  (same ty → same row)
Accessing B[k][col] = B[k*N + col] where col = bx*32 + tx

B addresses: B[k*N + bx*32], B[k*N + bx*32+1], ..., B[k*N + bx*32+31]
             ← 32 consecutive floats = 1 cache line = COALESCED ✓
```

**In `matmul_tiled_kernel`, shared mem load phase:**
```
Loading Bs[ty][tx] = B[b_row * N + col] where col = bx*TILE + tx

B addresses for fixed b_row: B[b_row*N + bx*TILE], ..., B[b_row*N + bx*TILE+15]
← 16 consecutive floats = COALESCED ✓
```

**Anti-pattern: column-first access (strided):**
```
B[k][col] where k varies (stride N between threads) → UNCOALESCED ✗
Each thread in the warp accesses B[0*N+col], B[1*N+col], ..., B[31*N+col]
= 32 addresses spaced N floats apart → 32 separate cache line fetches
```

This is exactly why column-major matrix storage would be better for B here,
or why transposing K before the Q·Kᵀ matmul can improve performance.

---

## Optimization 5 — Vectorised Loads (float4)

**Not fully implemented in this project, shown here for education.**

CUDA's `float4` type loads 4 floats (128 bits) in a single instruction,
matching the cache line width and maximising throughput.

```cuda
// Standard load: 4 separate 32-bit loads
float a0 = A[i+0];
float a1 = A[i+1];
float a2 = A[i+2];
float a3 = A[i+3];

// Vectorised load: 1 × 128-bit load
float4 a = reinterpret_cast<const float4*>(A)[i/4];
// Access: a.x, a.y, a.z, a.w
```

**Requirements:**
- Array must be 16-byte aligned (`cudaMalloc` guarantees this)
- Accesses must be 4-float-aligned (pad dimensions to multiples of 4)

**Expected gain:** ~1.5–2× better global memory throughput for memory-bound
kernels.

---

## Softmax Numerical Stability

**Without stability fix (DANGEROUS):**
```
x = [1000, 1001, 1002]
exp(x) = [e^1000, e^1001, e^1002] = [Inf, Inf, Inf]   ← overflow!
sum = Inf
output = Inf/Inf = NaN
```

**With stability fix (correct):**
```
max_x = 1002
x - max_x = [-2, -1, 0]
exp(x - max_x) = [e^-2, e^-1, e^0] = [0.135, 0.368, 1.0]
sum = 1.503
output = [0.090, 0.245, 0.665]   ← correct!
```

**Mathematical proof the fix is equivalent:**
```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
             = exp(x_i - c) × exp(c) / (Σ_j exp(x_j - c) × exp(c))
             = exp(x_i - c) / Σ_j exp(x_j - c)    ← c cancels!
```

Any constant c preserves correctness; using `c = max(x)` minimises overflow.

---

## KV Cache Complexity Analysis

**Without KV cache:**
At generation step t, we process a sequence of length t:
- K, V projections: O(t × d²) each
- Attention: O(t² × d)
- Total per step: O(t × d²)
- Total for n steps: O(n² × d²)

**With KV cache:**
At step t, only the new token's K, V are computed:
- K, V projections for new token: O(d²) each
- Attention over cache: O(t × d)
- Total per step: O(d² + t × d)
- Total for n steps: O(n × d²) — linear!

**Practical impact:**
For GPT-3 (d=12288, n=2048):
- Without cache: 2048² × 12288² ≈ 6.3 × 10¹⁴ FLOPs
- With cache:    2048  × 12288² ≈ 3.1 × 10¹¹ FLOPs
- **Speedup: 2048×**

---

## Future Optimizations (Beyond This Project)

| Optimization | Benefit | Complexity |
|-------------|---------|------------|
| FlashAttention | O(1) memory, faster softmax | High |
| Tensor Cores (WMMA API) | 16× FP16 throughput | High |
| Multi-Stream CUDA | Overlap compute+transfer | Medium |
| FP16 quantisation | 2× memory, 2× compute | Medium |
| INT8 quantisation | 4× memory, 4× compute | High |
| Speculative decoding | 3–4× generation speedup | Very High |
| Continuous batching | GPU utilisation | Very High |
| PagedAttention (vLLM) | Dynamic KV cache | Very High |

---

## Benchmark Interpretation Guide

**GFLOPS calculation for N×N×N matmul:**
```
FLOPs = 2 × N³   (N² outputs × N multiply-add operations each)
Time  = measured wall-clock seconds (include host-device transfer for GPU)
GFLOPS = FLOPs / (Time × 1e9)
```

**Why GPU benchmarks include PCIe transfer:**
Our `matmul_cuda_tiled()` copies A, B to device and C back to host.
For N=256: data = 3 × 256² × 4B = 768 KB; PCIe at 12 GB/s → 0.06 ms.
For N=1024: data = 12 MB; PCIe → 1 ms — significant vs ~0.1 ms kernel time.

In production inference engines, data stays on GPU across layers — zero PCIe
cost. Benchmark with `--include-transfer=false` for kernel-only timing.

**Reading the speedup column:**
All speedups are relative to CPU Naive = 1.0×. Typical values:
- CPU Tiled: 5–10× (cache efficiency)
- CUDA Naive: 20–100× (GPU parallelism, despite uncoalesced issues)
- CUDA Tiled: 50–200× (GPU + shared memory)
