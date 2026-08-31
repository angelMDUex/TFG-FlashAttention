# FlashAttention CUDA & Triton

Implementation, benchmarking and profiling of FlashAttention-style kernels on NVIDIA GPUs using **CUDA**, **Triton** and **PyTorch SDPA**.

This repository contains the implementation developed as part of a Bachelor's Thesis focused on understanding how GPU kernel design decisions affect attention performance. The project compares a manually optimized CUDA kernel against Triton and PyTorch reference implementations, with detailed profiling using NVIDIA Nsight Compute.

## Overview

The project studies the forward pass of non-causal attention:

$$
O =
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d}}
\right)V
$$

without materializing the complete attention matrix in global memory.

The CUDA implementation follows the tiled computation and online-softmax principles introduced by FlashAttention. The kernel is designed specifically around NVIDIA Ampere mechanisms such as:

* BF16 Tensor Core operations
* FP32 accumulation
* `mma.sync`
* `ldmatrix`
* `cp.async`
* shared-memory buffering
* shared-memory swizzling
* coalesced global-memory accesses
* online softmax

A Triton implementation of the same operation is included for comparison, together with PyTorch SDPA and its cuDNN backend.

> The custom CUDA implementation reproduces the core tiled FlashAttention computation, but it should not be interpreted as a complete reimplementation of all FlashAttention-2 work-partitioning strategies.

---

## Experimental configuration

The main experiments presented in the thesis use:

| Parameter          | Value                 |
| ------------------ | --------------------- |
| GPU                | NVIDIA A100-SXM4-40GB |
| Architecture       | Ampere                |
| Compute capability | `sm_80`               |
| Input type         | BF16                  |
| Accumulation       | FP32                  |
| Head dimension     | 128                   |
| Batch size         | 1                     |
| Number of heads    | 1                     |
| Attention          | Non-causal            |
| Sequence length    | Multiples of 16       |

The profiling analysis focuses primarily on:

```text
N = 8192
d = 128
B = 1
H = 1
```

while timing experiments cover sequence lengths from `8192` up to `1048576`.

---

## Implementations

The repository compares four execution paths.

### CUDA

Custom Ampere-specific implementation.

The final kernel uses:

* 4 warps per CTA
* 16 query rows per warp
* 64 query rows per CTA
* 16 × 128 K/V tiles
* four shared-memory K/V buffers
* `cp.async` for global-to-shared transfers
* `ldmatrix` for shared-to-register matrix loads
* `mma.sync.m16n8k16` Tensor Core operations
* BF16 inputs and FP32 accumulators
* online softmax
* XOR-based shared-memory swizzling

### Triton

FlashAttention-style tiled implementation written in Triton.

Triton manages several low-level implementation decisions automatically and explores configurations using parameters such as:

* block size
* number of warps
* number of pipeline stages

### PyTorch SDPA

Reference implementation based on:

```python
torch.nn.functional.scaled_dot_product_attention
```

The repository includes measurements for both the standard optimized SDPA path and the cuDNN backend.

---

## Requirements

The project uses:

* Linux
* NVIDIA GPU
* CUDA Toolkit
* CMake
* C++ compiler
* Python
* `uv`
* PyTorch
* Triton
* pytest
* NVIDIA Nsight Compute

The reference environment used for the final experiments included:

```text
Ubuntu 22.04
CUDA Toolkit 13.0
PyTorch 2.13 + CUDA 13.0
Triton 3.7
Nsight Compute 2025.3
```

Exact environment information can be collected using the provided script.

---

## Python environment

The Python environment is managed using [uv](https://docs.astral.sh/uv/).

Install the dependencies with:

```bash
uv sync
```

The lock file is committed to the repository to make the Python environment reproducible.

---

## CMake configuration

A CMake preset named:

```text
angel-wsl
```

is included in the repository.

The preset contains the CUDA architecture and compiler paths used during development.

In particular, the CUDA build targets Ampere:

```text
sm_80
```

The preset is provided as a reproducible starting configuration and can be modified according to the local installation of CUDA, CMake and the host compilers.

### Configure

```bash
cmake --preset angel-wsl
```

### Build

```bash
cmake --build --preset angel-wsl
```

Generated build files are stored under:

```text
build/
```

---

## Running the tests

### pytest

Install the environment and run all tests with:

```bash
uv sync
uv run pytest
```

For verbose test output:

```bash
uv run pytest -s
```

### Performance tests

Performance benchmarks are marked with:

```text
tiempo
```

Run them with:

```bash
uv run pytest -s -m tiempo
```

All benchmark implementations use the **same random seed and the same generated inputs** so that CUDA, Triton, SDPA and cuDNN operate on equivalent data.

The benchmark currently uses:

```python
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```

Timing is performed using CUDA Events after warm-up. The reported values are based on repeated executions rather than Nsight Compute kernel durations.

### Profiling tests

Tests intended for profiling are marked with:

```text
profile
```

They can be executed with:

```bash
uv run pytest -s -m profile
```

These tests perform warm-up before entering the NVTX profiling region.

---

## CTest

The CMake build also integrates tests through CTest.

After configuring and building the project:

```bash
ctest --test-dir build
```

For verbose output:

```bash
ctest --test-dir build --output-on-failure
```

---

## Profiling with Nsight Compute

A profiling script is included:

```text
profile.sh
```

Make it executable:

```bash
chmod u+x ./profile.sh
```

Run:

```bash
./profile.sh
```

The script profiles the CUDA, Triton, SDPA and cuDNN implementations using NVIDIA Nsight Compute.

Reports are stored under:

```text
profile/
```

Typical generated files include:

```text
profile/
├── NVIDIA_A100-SXM4-40GB_cuda_ampere.ncu-rep
├── NVIDIA_A100-SXM4-40GB_triton.ncu-rep
├── NVIDIA_A100-SXM4-40GB_sdpa.ncu-rep
└── NVIDIA_A100-SXM4-40GB_sdpa_cudnn.ncu-rep
```

Nsight Compute is intentionally kept separate from the timing benchmarks.

Profiling can require kernel replay and instrumentation, so **Nsight Compute execution times should not be interpreted as benchmark timings**.

---

## Environment report

The repository includes:

```text
environment_report.sh
```

It collects information about the machine used for the experiments, including:

* operating system
* CPU
* system memory
* GPU
* compute capability
* number of SMs
* GPU clocks
* power limit
* NVIDIA driver
* CUDA Toolkit
* compiler versions
* Python
* PyTorch
* Triton
* cuDNN
* Nsight Compute

Run it with:

```bash
chmod u+x ./environment_report.sh
./environment_report.sh
```

This is useful when reproducing benchmarks on another machine.

---

## Project layout

```text
.
├── build/                  # CMake build output
├── profile/                # Nsight Compute reports
├── test/                   # pytest tests and benchmarks
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
├── uv.lock
├── profile.sh
└── environment_report.sh
```

Additional CUDA, C++ and Triton source files contain the actual kernel implementations and PyTorch bindings.

---

## Validation

Numerical correctness is checked against the PyTorch reference using:

```python
torch.allclose(...)
```

with tolerances appropriate for BF16 arithmetic.

The validation and benchmark inputs are generated deterministically from a fixed random seed.

---

## Benchmark results

For large sequence lengths on the A100, the measured sustained throughput is approximately:

| Implementation |   Throughput |
| -------------- | -----------: |
| Custom CUDA    | ~101 TFLOP/s |
| Triton         | ~126 TFLOP/s |
| cuDNN          | ~167 TFLOP/s |
| PyTorch SDPA   | ~177 TFLOP/s |

The custom CUDA kernel therefore reaches approximately **100 TFLOP/s sustained** in the tested configuration.

The objective of the project is not to outperform production libraries, but to understand which architectural decisions explain the observed differences.

---

## Profiling observations

The Nsight Compute analysis shows several interesting properties of the final CUDA implementation.

### Global-memory coalescing

The main kernels show essentially no excessive sectors caused by uncoalesced accesses.

For the custom CUDA implementation:

```text
requested sectors = ideal sectors
excessive sectors = 0
```

Therefore, poor global-memory coalescing is not responsible for the observed performance difference.

### Shared-memory bank conflicts

The custom CUDA kernel reports:

```text
bank conflicts = 0
```

confirming that the shared-memory swizzling strategy successfully avoids bank conflicts for the measured configuration.

### Register pressure

Approximate register usage:

| Implementation | Registers/thread |
| -------------- | ---------------: |
| CUDA           |              148 |
| Triton         |              199 |
| SDPA           |              240 |
| cuDNN          |              209 |

The custom CUDA kernel uses fewer registers than the compared implementations and does not spill registers in the final profiled configuration.

### Occupancy

For `N = 8192`, the CUDA implementation has:

```text
theoretical occupancy : 18.75 %
achieved occupancy    : 7.56 %
```

The kernel allows several CTAs to reside simultaneously on an SM, but the grid does not contain enough blocks to fully exploit that theoretical residency.

### Global sectors

One of the most important differences is the total number of requested global-memory sectors:

| Implementation |  Sectors |
| -------------- | -------: |
| Triton         |  ~8.52 M |
| cuDNN          |  ~8.52 M |
| CUDA           | ~16.91 M |
| SDPA           | ~17.37 M |

The accesses themselves are coalesced, but the CUDA implementation requests approximately twice as many sectors as Triton and cuDNN.

The Triton mapping observed during profiling is consistent with greater K/V reuse per CTA. For cuDNN, the measured traffic is also consistent with greater effective data reuse, although its internal work partitioning is not inferred solely from this measurement.

---

## Main lesson

A major conclusion of the project is that optimizing local GPU properties is not enough by itself.

The CUDA kernel achieves:

* coalesced global accesses
* zero measured bank conflicts
* no register spilling
* relatively low register usage
* controlled shared-memory usage
* explicit Tensor Core execution

and yet remains slower than the strongest reference implementations.

The results emphasize the importance of **global work partitioning, data reuse and available parallelism**, in addition to instruction-level optimization.

---

## Limitations

The results in this repository should be interpreted within the exact scope of the experiment.

The final evaluation:

* uses one NVIDIA A100-SXM4-40GB;
* uses BF16 inputs;
* uses head dimension `128`;
* uses `B = 1`;
* uses `H = 1`;
* evaluates non-causal attention;
* requires sequence lengths compatible with the implemented tiling;
* evaluates the forward attention operation only;
* does not evaluate backward propagation;
* does not evaluate complete Transformer training or inference;
* does not demonstrate performance portability to other GPU architectures.

The detailed Nsight Compute analysis is mainly performed for `N = 8192`, while timing scalability is evaluated across a larger range of sequence lengths.

---

## Thesis

The repository accompanies the Bachelor's Thesis:

**FlashAttention kernel implementation and profiling using CUDA, Triton and PyTorch**

The thesis discusses the algorithm, implementation decisions, profiling methodology, benchmark results and lessons learned in greater detail.

---

## References

The technical background of the project is based primarily on:

* Vaswani et al., *Attention Is All You Need*
* Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
* Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
* NVIDIA CUDA C++ Programming Guide
* NVIDIA PTX ISA
* NVIDIA Nsight Compute documentation
* Triton documentation
* PyTorch SDPA documentation
* *Modern GPU Programming for MLSys*

---
