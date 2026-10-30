# CPU against GPU benchmark v1

Run 2026-10-30, to decide whether future runs should move to the GPU. Timing only:
synthetic data, no scientific output.

- **Script:** [`benchmarks/cpu_gpu_v1/bench.py`](../benchmarks/cpu_gpu_v1/bench.py), driven by
  [`run_all.sh`](../benchmarks/cpu_gpu_v1/run_all.sh). It is standalone and imports nothing
  from the repo.
- **Artifacts:**
  - `reports/cpu-gpu-benchmark-v1-results.jsonl`, one line per configuration;
  - `reports/cpu-gpu-benchmark-v1-environment.json`, the hardware and library versions;
  - `reports/cpu-gpu-benchmark-v1-run_all.log.gz`, the driver's log.
- **Command:** `PY=<python with torch 2.14.1+cu130> benchmarks/cpu_gpu_v1/run_all.sh`, run
  from that directory. It refuses to start while a fan-in pilot run is active.

## What one step is

A training step shaped like the sequence and fan-in pilots' network:

- encode B images (28 × 28) and B tap signals (128 steps) with the pilots' two encoders;
- project each to the 16-dimension code;
- run `g` on code pairs with a relation token, and `P` on code pairs;
- a binary cross-entropy plus a squared-error loss, backward, one Adam step.

float32. Warm-up of 20 steps, then 200 timed steps. GPU steps are synchronized before
timing.

This is a simplification of the real update. The real one builds several losses, matches
against candidate collections, assembles sequence rows in Python and renders inputs on the
CPU. Those parts have many small operations and Python overhead that a GPU does not speed
up.

## Hardware

AMD Ryzen 5 3600 (6 cores, 12 threads), 15 GB RAM; NVIDIA GeForce GTX 1650 (4 GB), driver
580.173.02; torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.24.

## Results

### One process

Steps per second (median step time in parentheses).

| Batch | CPU, 1 thread | CPU, 6 threads | GPU | GPU ÷ CPU 6 threads |
| --- | --- | --- | --- | --- |
| 64 | 103 (9.3 ms) | 157 (6.2 ms) | 377 (2.6 ms) | 2.4× |
| 270 | 27.6 (36.5 ms) | 52.7 (19.3 ms) | 307 (3.2 ms) | 5.8× |
| 1,024 | 6.5 (152 ms) | 13.4 (75 ms) | 153 (6.6 ms) | 11× |
| 4,096 | 1.3 (780 ms) | 2.6 (389 ms) | 40.4 (24.8 ms) | 16× |
| 16,384 | 0.26 (3.8 s) | 0.46 (1.8 s) | 10.4 (96 ms) | 23× |

### Several processes at once

Aggregate steps per second over W concurrent processes, each with 1 CPU thread. GPU
processes share the card.

| Batch | W | CPU | GPU | GPU ÷ CPU |
| --- | --- | --- | --- | --- |
| 270 | 1 | 25.6 | 303 | 12× |
| 270 | 3 | 51.8 | 406 | 7.8× |
| 270 | 6 | 63.5 | 403 | 6.3× |
| 1,024 | 1 | 6.8 | 149 | 22× |
| 1,024 | 3 | 12.6 | 141 | 11× |
| 1,024 | 6 | 14.4 | 141 | 9.8× |

## Reading

- **On this step the GPU is far faster.** At batch 270, one GPU process does about five
  times the work of six CPU workers together, and three GPU processes about six times.
- **The GPU saturates at about three processes.** Above that, aggregate throughput stays
  flat. The CPU gains little beyond three workers too (6 cores, memory bandwidth).
- **Larger batches favour the GPU more.** At batch 64 the gain is 2.4×; per-step overhead
  dominates.
- **The real speedup will be smaller.** The benchmark leaves out the Python-heavy parts of
  the update (candidate matching, sequence rows, rendering, evaluation). Its CPU step at
  batch 270 and 1 thread (37 ms) is of the same order as the profiled real update (about
  29 ms), so the arithmetic is representative; the overhead is not. The real gain has to
  be measured on the pilot itself.

## Before switching

- **Device handling.** No runner places tensors on a device today. Models, data tensors
  and the encoders' inputs would need to move to the GPU, with rendering kept on the CPU.
- **Determinism.** CPU runs are deterministic across repeats. On the GPU this needs
  `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`, at some speed
  cost. GPU results will not equal CPU results, so CPU and GPU arms must not be compared
  within one experiment.
- **A real timing.** One seed of the sequence pilot on the GPU, against its CPU runtime,
  before planning runs around the benchmark's figures.
- **Memory.** 4 GB holds several processes of this size; each CUDA context costs a few
  hundred MB.
