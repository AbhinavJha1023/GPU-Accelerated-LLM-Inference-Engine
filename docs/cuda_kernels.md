# CUDA Kernels — GPU-Accelerated LLM Inference Engine

## CUDA Execution Model Primer

```
GPU
└── SM (Streaming Multiprocessor) × N
    └── Warp × 32 (concurrent hardware threads)
        └── Thread
```

**Grid → Block → Thread hierarchy:**

```
gridDim  = (Gx, Gy)          ← number of blocks
blockDim = (Bx, By)          ← threads per block

Thread global index:
  row = blockIdx.y * blockDim.y + threadIdx.y
  col = blockIdx.x * blockDim.x + threadIdx.x
```

**Memory hierarchy (fastest → slowest):**

| Level | Size | Latency | Scope |
|-------|------|---------|-------|
| Registers | 32-64 KB/SM | 1 cycle | Thread |
| L1 / Shared mem | 48-228 KB/SM | ~4 cycles | Block |
| L2 cache | 4-40 MB | ~30 cycles | All SMs |
| Global (HBM) | 16-80 GB | ~200 cycles | Device |

---

## Kernel 1 — `matmul_naive_kernel`

### Purpose
One thread computes one output element `C[row][col]`.

### Launch Configuration
```
blockDim = (32, 32)   = 1024 threads/block
gridDim  = (ceil(N/32), ceil(M/32))
```

### Thread → Data Mapping
```
Thread (tx=2, ty=1) in block (bx=0, by=0):
  col = 0*32 + 2 = 2
  row = 0*32 + 1 = 1
  computes C[1][2] = Σ_k A[1][k] * B[k][2]
```

### Memory Access Pattern
```
Warp: ty=0, tx=0..31  (all same row, different columns)

A[row][k]:  row is same for all threads in warp → same address
            → broadcast from L2 after first hit (good)

B[k][col]:  col varies as tx=0..31 → B[k*N+0], B[k*N+1], ..., B[k*N+31]
            → 32 consecutive floats = 1 cache line → COALESCED ✓
```

### Why It's Slow
For each output element, we read K elements of A and K elements of B from
global memory. Without reuse, total reads = `M*N*K*2` — no data is shared
between adjacent threads.

### Pseudo-code
```cuda
__global__ void matmul_naive_kernel(float* A, float* B, float* C,
                                    int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row*K + k] * B[k*N + col];
    C[row*N + col] = acc;
}
```

---

## Kernel 2 — `matmul_tiled_kernel`

### Purpose
Use `__shared__` memory to cache tiles of A and B, reducing global memory
accesses by a factor of `TILE`.

### Launch Configuration
```
blockDim = (TILE=16, TILE=16) = 256 threads/block
gridDim  = (ceil(N/16), ceil(M/16))
```

### Shared Memory Layout
```
__shared__ float As[16][16];   // 1024 bytes
__shared__ float Bs[16][16];   // 1024 bytes
                               // Total: 2048 bytes per block
```

### Algorithm (per block)

```
Block (bx, by) computes the sub-matrix C[by*16:(by+1)*16, bx*16:(bx+1)*16]

                 K
         ┌──────┬──────┬──────┐
         │  t=0 │  t=1 │  t=2 │  ← tiles of A  [M × K]
  M      ├──────┼──────┼──────┤
         │      │      │      │
         └──────┴──────┴──────┘

                 N
         ┌──────┬──────┬──────┐
         │  t=0 │      │      │
  K      ├──────┤      │      │  ← tiles of B  [K × N]
         │  t=1 │      │      │
         ├──────┤      │      │
         │  t=2 │      │      │
         └──────┴──────┴──────┘

For each tile t (0 → ceil(K/TILE)):
  1. Load As[ty][tx] = A[row][t*TILE + tx]   (1 global read per thread)
  2. Load Bs[ty][tx] = B[t*TILE + ty][col]   (1 global read per thread)
  3. __syncthreads()                          (barrier)
  4. acc += Σ_k (As[ty][k] * Bs[k][tx])     (16 MACs, all from shared mem)
  5. __syncthreads()
```

### Memory Savings
```
Without tiles: each thread reads K elements of A and K elements of B
               = 2*K global reads per output element

With TILE=16:  Each A element loaded once into As, then READ 16 TIMES by
               the 16 threads with tx=0..15 in the same block.
               Same for B → 16× fewer global reads.

Savings = (TILE - 1) / TILE * 100% = 93.75% for TILE=16
```

### Occupancy Analysis (sm_80, A100)
```
Resources per block:
  Threads:     256
  Registers:   ~32 per thread → 8192 total
  Shared mem:  2048 bytes

SM limits (sm_80):
  Max threads/SM:   2048   → 2048/256 = 8 blocks simultaneously
  Max blocks/SM:    32
  Shared mem/SM:    192 KB → 192*1024/2048 = 96 blocks possible
  Register limit:   65536  → 65536/8192 = 8 blocks

Limiting factor: registers → 8 concurrent blocks = 2048 threads/SM
Theoretical occupancy: 2048/2048 = 100% ← excellent!
```

### `#pragma unroll`
```cuda
#pragma unroll
for (int k = 0; k < TILE; ++k) {
    acc += As[ty][k] * Bs[k][tx];
}
```
The compiler unrolls this 16-iteration loop into 16 explicit FMA instructions,
eliminating branch prediction overhead and enabling the hardware to issue
multiple instructions per cycle.

---

## Kernel 3 — `scale_kernel`

### Purpose
Multiply every element by a scalar in-place. Used to divide attention scores
by √d_k.

### Launch Configuration
```
blockDim = 256
gridDim  = ceil(n / 256)
```

### Code
```cuda
__global__ void scale_kernel(float* data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}
```

This is a **memory-bandwidth-bound** kernel: each thread loads 4 bytes,
multiplies (negligible), writes 4 bytes. Peak BW on A100: 2 TB/s.

---

## Kernel 4 — `softmax_kernel`

### Purpose
Row-wise numerically-stable softmax using parallel block reduction.

### Launch Configuration
```
gridDim  = rows            (one block per row)
blockDim = 256             (fixed thread count)
shared   = 256 * 4 bytes   (one float per thread)
```

### Algorithm (per block / per row)

**Phase 1 — Find row maximum:**
```
Each thread handles cols/256 elements:
  local_max = max(row[threadIdx.x], row[threadIdx.x + 256], ...)

Store local_max in sdata[threadIdx.x]
__syncthreads()

Tree reduction (log2(256) = 8 steps):
  stride=128: sdata[i] = max(sdata[i], sdata[i+128])  __syncthreads()
  stride=64:  sdata[i] = max(sdata[i], sdata[i+64])   __syncthreads()
  ...
  stride=1:   sdata[0] = global max
```

**Phase 2 — Exp and sum:**
```
row_out[c] = exp(row_in[c] - sdata[0])    ← subtract max (stability)
local_sum += row_out[c]
...same tree reduction for sum...
```

**Phase 3 — Normalise:**
```
row_out[c] /= row_sum
```

### Why the Tree Reduction is O(log B)
```
256 threads → 8 synchronisation steps to reduce to 1 value:
  256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
```
A naive serial scan would take 256 steps — 32× slower.

---

## Memory Transfer Pattern (SDPA on GPU)

```
CPU side             PCIe (12-32 GB/s)        GPU side
─────────────────────────────────────────────────────
Q [seq_q × d_k]     ──────────────────►      dQ
K [seq_k × d_k]     ──────────────────►      dK
V [seq_k × d_v]     ──────────────────►      dV

                          (on GPU, no PCIe)
                    dKᵀ = transpose(dK)
                    dS  = dQ · dKᵀ           ← tiled matmul
                    dS  *= 1/√d_k            ← scale kernel
                    dA  = softmax(dS)        ← softmax kernel
                    dO  = dA · dV            ← tiled matmul

Out [seq_q × d_v]   ◄──────────────────      dO
─────────────────────────────────────────────────────
PCIe traffic: 3 uploads + 1 download (vs 6+ if CPU round-trips used)
```

---

## Performance Expectations

| Size | Kernel | Expected GFLOPS | Notes |
|------|--------|-----------------|-------|
| 1024×1024 | CPU Naive | 0.1–0.5 | L2/L3 bound |
| 1024×1024 | CPU Tiled | 1–8 | L1 cache efficient |
| 1024×1024 | CUDA Naive | 5–30 | Global mem BW |
| 1024×1024 | CUDA Tiled | 30–100 | Shared mem reuse |
| 1024×1024 | cuBLAS | 100–300 | Theoretical peak |

GPU peak (A100 FP32): ~19.5 TFLOPS — our tiled kernel achieves ~5%.
The gap is closed by vectorised loads (`float4`), tensor cores, and more
aggressive register file management — see `optimization_notes.md`.

---

## Profiling with Nsight Systems

```bash
# Profile the main engine
nsys profile --stats=true ./build/llm_engine

# Profile with CUDA trace
nsys profile --cuda-memory-usage=true ./build/llm_engine

# Nsight Compute for kernel-level analysis
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    --target-processes all ./build/benchmark_matmul 1024
```

Key metrics to watch:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM utilisation
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` — shared mem conflicts
- `gpu__time_duration.sum` — total kernel time
