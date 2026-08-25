# GPU Kernels

Learning GPU performance engineering from scratch: CUDA first, then HIP/ROCm on MI300X.
Working toward a benchmarked GEMM optimization ladder.

## Background

CS B.S./M.S. from UNC Chapel Hill. Previously built
[TorchFuzz](link), a GPU-accelerated RTL fuzzing framework. Comfortable with
parallelism and digital design; this repo is me closing the gap on kernel-level
performance work.

## Contents

| Kernel | Status | Notes |
|---|---|---|
| `01-vector-add/` | ✅ correct | Memory-bound baseline. Bandwidth numbers pending. |
| `02-gemm-naive/` | planned | `C = αAB + βC`, one thread per output element |
| `03-gemm-optimized/` | planned | coalescing → shared memory tiling → register tiling → tensor cores |

## Hardware

RTX 6000 Ada / A6000 (UNC CS cluster, SLURM). MI300X via AMD Developer Cloud once
capacity frees up.

## Build

```bash
nvcc -O3 -arch=native -lineinfo vector_add.cu -o vector_add
./vector_add
compute-sanitizer ./vector_add   # correctness ≠ memory safety
```

## Benchmarking methodology

Timing uses CUDA events, not wall clock, since kernel launches are asynchronous.
Discarded warmup iterations, median over 100 runs (median rather than mean because
clocks aren't locked on a shared cluster). Host↔device transfers excluded from kernel
time and reported separately. Vector add is reported as effective bandwidth
(`3 * n * sizeof(float) / time`), not FLOP/s — it does one flop per twelve bytes moved.

## Notes

See [`takes.md`](takes.md) for opinions and things that surprised me.