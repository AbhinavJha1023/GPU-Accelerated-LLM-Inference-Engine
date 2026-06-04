# Architecture — GPU-Accelerated LLM Inference Engine

## Overview

This project implements a miniature LLM inference runtime from scratch using
C++20 and CUDA. Every component has a CPU fallback so the entire codebase
compiles and runs on a machine without an NVIDIA GPU.

---

## Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        MiniTransformer                          │
│  (main.cpp — combines all components into an inference loop)    │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │  Tokenizer   │  │  MultiHeadAttention  │  │   KV Cache   │  │
│  │(tokenizer.*)│  │    (attention.*)     │  │ (kv_cache.*) │  │
│  └──────────────┘  └──────────────────────┘  └──────────────┘  │
│                             │                                   │
│              ┌──────────────┴──────────────┐                   │
│              │         Softmax + SDPA       │                   │
│              │      (attention_cpu.cpp /    │                   │
│              │       attention_cuda.cu)     │                   │
│              └──────────────┬──────────────┘                   │
│                             │                                   │
│                    ┌────────┴────────┐                          │
│                    │    MatMul        │                          │
│                    │ matmul_cpu.cpp  │                          │
│                    │ matmul_cuda.cu  │                          │
│                    └────────┬────────┘                          │
│                             │                                   │
│                    ┌────────┴────────┐                          │
│                    │     Tensor      │                          │
│                    │  tensor.cpp     │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Token → Logits

```
Token ID
    │
    ▼
Embedding Table [vocab_size × d_model]
    │  lookup row token_id
    ▼
Token Embedding [1 × d_model]
    │
    +── Positional Encoding [1 × d_model]
    │      sin/cos at position pos
    ▼
Input [1 × d_model]
    │
    ▼ ─────────────────────────────────────────────────────────
    │              MultiHeadAttention
    │
    │   W_Q, W_K, W_V : [d_model × d_model]
    │   Q = Input · W_Q   [1 × d_model]
    │   K = Input · W_K   [1 × d_model]  ─► append to KV cache
    │   V = Input · W_V   [1 × d_model]  ─► append to KV cache
    │
    │   K_all = cache.get_keys()   [t × d_model]
    │   V_all = cache.get_values() [t × d_model]
    │
    │   For each head h (0..num_heads-1):
    │     Q_h = Q[:, h*hd:(h+1)*hd]     [1 × head_dim]
    │     K_h = K_all[:, h*hd:...]      [t × head_dim]
    │     V_h = V_all[:, h*hd:...]      [t × head_dim]
    │
    │     S_h = Q_h · K_hᵀ / √head_dim  [1 × t]
    │     A_h = softmax(S_h)             [1 × t]
    │     out_h = A_h · V_h              [1 × head_dim]
    │
    │   concat = [out_0 | out_1 | ... | out_{h-1}]  [1 × d_model]
    │   hidden = concat · W_O                        [1 × d_model]
    │
    ▼ ─────────────────────────────────────────────────────────
    │
    ▼
Output Projection [d_model × vocab_size]
    │  hidden · output_proj
    ▼
Logits [1 × vocab_size]
    │
    ▼
argmax → Next Token ID
```

---

## Autoregressive Inference Loop

```
┌────────────────────────────────────────────────────────────────┐
│ Prefill Phase (process entire prompt once)                     │
│                                                                │
│  for token in prompt[0..N-1]:                                  │
│    x = embed(token, position)                                  │
│    MHA.forward_cached(x, kv_cache)   ← builds cache           │
│                                                                │
│ Generation Phase (one token at a time)                         │
│                                                                │
│  for step in 0..max_new_tokens:                                │
│    x       = embed(current_token, current_position)            │
│    hidden  = MHA.forward_cached(x, kv_cache)   ← O(1) new K,V │
│    logits  = hidden · output_proj                              │
│    next_id = argmax(logits)                                    │
│    current_token = next_id                                     │
│    if next_id == EOS: break                                    │
└────────────────────────────────────────────────────────────────┘
```

**Complexity comparison:**

| Approach | K/V matmuls per step | Total for n steps |
|----------|---------------------|-------------------|
| No cache | O(t) — recompute all | O(n²) |
| KV cache | O(1) — only new token | O(n) |

---

## CPU vs GPU Fallback

Every operation has two implementations selected at compile time:

```
#ifdef USE_CUDA
    result = operation_cuda(inputs);    // GPU path
#else
    result = operation_cpu(inputs);     // CPU path (always available)
#endif
```

The `matmul()`, `softmax()`, and `sdpa()` dispatch functions handle this
automatically. You never need to call the CUDA versions directly unless
you want to benchmark them explicitly.

---

## File Map

| File | Responsibility |
|------|----------------|
| `include/tensor.hpp` | 2-D Tensor class + GPU memory management |
| `include/matmul.hpp` | MatMul function declarations |
| `include/attention.hpp` | Softmax, SDPA, MHA class |
| `include/kv_cache.hpp` | KV Cache class |
| `include/tokenizer.hpp` | Character-level tokenizer |
| `include/benchmark.hpp` | Timer, GFLOPS, result printing (header-only) |
| `src/tensor.cpp` | Tensor implementation |
| `src/matmul_cpu.cpp` | CPU naive + tiled matmul |
| `src/matmul_cuda.cu` | CUDA naive + tiled matmul kernels |
| `src/attention_cpu.cpp` | softmax, SDPA, MHA (CPU) |
| `src/attention_cuda.cu` | softmax, SDPA kernels (CUDA) |
| `src/kv_cache.cpp` | KV cache grow + retrieve |
| `src/tokenizer.cpp` | Character tokenizer |
| `src/main.cpp` | End-to-end demo of all phases |
| `tests/test_matmul.cpp` | Matmul unit tests (GoogleTest) |
| `tests/test_attention.cpp` | Attention unit tests |
| `tests/test_kv_cache.cpp` | KV cache + tokenizer tests |
| `benchmarks/benchmark_matmul.cpp` | Matmul benchmark executable |
| `benchmarks/benchmark_attention.cpp` | Attention benchmark executable |

---

## Running on Google Colab (GPU)

```python
# 1. Install CUDA toolkit (pre-installed on Colab GPU runtime)
# 2. Clone or upload this project
# 3. Build with CUDA enabled:

!apt-get install -y cmake ninja-build
!cmake -B build -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -GNinja
!cmake --build build -j4

# 4. Run:
!./build/llm_engine
!./build/benchmark_matmul 64 128 256 512
!./build/benchmark_attention
```

Select Runtime → Change runtime type → GPU (T4 or A100).
The T4 is sm_75; the A100 is sm_80.

---

## Extending the Engine

To add a new operation (e.g., Layer Normalisation):

1. Declare in `include/attention.hpp` (or a new header)
2. Implement `layernorm_cpu()` in `attention_cpu.cpp`
3. Implement `layernorm_cuda()` in `attention_cuda.cu`
4. Add a dispatch function `layernorm()`
5. Add a test in `tests/test_attention.cpp`
6. Integrate into `MultiHeadAttention::forward()`
