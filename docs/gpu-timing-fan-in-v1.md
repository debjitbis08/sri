# GPU port and real-pilot timing for the fan-in runner

Run 2026-10-30. Follows the [CPU against GPU benchmark](cpu-gpu-benchmark-v1.md), which timed a simplified step and
left "device handling, determinism and a real timing" open. **Timing and a device check; not scientific evidence.**

- **Code:** `experiments/device.py` (device layer, `NFI_DEVICE=cuda`, default CPU and unchanged), small hooks in
  `experiments/fan_in_pilot.py` (`DEV.install()` in each worker, models built on the CPU then moved, device fields in the
  runtime JSON, `NFI_ONLY=config:seed,...` and `NFI_TAG` to run a few tasks into separate reports).
- **Environment:** `requirements-cuda.txt` (torch 2.14.1+cu130, same numpy, scipy and pillow as `requirements.txt`), in
  `/data/venvs/sri-cuda`. NVIDIA GeForce GTX 1650 (4 GB), driver 580.173.02, CUDA 13.0, cuDNN 9.24; the CPU is the
  Ryzen 5 3600 (6 cores) of the benchmark.
- **Commands:**
  - `NFI_DEVICE=cuda NSY_WORKERS=6 NFI_ONLY=read-q16-untied-rigl:230 NFI_TAG=-gputiming /data/venvs/sri-cuda/bin/python -m experiments.fan_in_pilot pilot reset`;
  - the same with seeds 231–233 (`-gputiming3`, 3 workers) and 234–239 (`-gputiming6`, 6 workers);
  - `.venv/bin/python -m experiments.fan_in_gpu_check` (the tables below).
- **Artifacts:** `reports/fan-in-pilot-v1-pilot-reset-gputiming{,3,6}-{results.json.gz,runtime.json}`; checkpoints in
  `runs/` (untracked).

## What the port does

Models are built on the CPU exactly as before, so initial weights and masks equal the CPU runs', then moved to the GPU.
`torch.from_numpy` returns device tensors and `Tensor.numpy` copies to the host, so the scientific modules are
unchanged. Checkpoints are saved as CPU tensors. cuDNN and cuBLAS use deterministic algorithms
(`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). The CPU path is unchanged; a CPU smoke
run after the change still finishes.

## Determinism

At smoke scale (70 updates), the same configuration and seed run twice on the GPU gave identical results and
bit-identical weights, for random rewiring (`read-q16-untied-rewire`) and for RigL (`read-q16-untied-rigl`); the
dense-reset path (`dense-inplace`) runs. This has not been repeated at full length on one task twice. GPU results do not
equal CPU results, and the GPU runs use a different library build.

## Throughput

`read-q16-untied-rigl`, 6,200 updates per run, one configuration.

| Runs | Processes | Wall per run | Effective seconds per run |
| --- | --- | --- | --- |
| GPU | 1 | 101 s | 102 |
| GPU | 3 | 140 s | 47 |
| GPU | 6 | 223 s | 37 |
| CPU (reset stage) | 6 workers × 2 threads | 373 s | 60 |

- **Gain over six CPU workers: 1.6× at six GPU processes**, 1.3× at three. The benchmark's 5.8× was for a simplified
  step. Each GPU process uses about one CPU core for the whole run (process CPU time ≈ wall time), so the work is bound by
  Python and kernel-launch overhead, which is not what a GPU speeds up.
- Peak GPU memory per process 91 MB; peak host memory per process about 2.5 GB.
- The six-process figure comes from one batch of six runs, and the CPU figure from the 80-run reset stage (60 s effective).

## Do the results agree?

Seeds 230–239, RigL arm, last block's evaluation, 103 measures with variation: **6 measures differ by |z| > 2 (about 5
expected by chance), mean z −0.53.** The six are correlated composition measures in the descending direction (for
example Devanagari forms two steps down, 0.27 on CPU and 0.13 on GPU). The transfer measures agree within seed noise
(taps → Chinese forms, withheld: 0.66 CPU, 0.57 GPU). Ten seeds cannot exclude a modest device effect, and the composition
measures should be watched in any GPU study. This is an agreement check on one arm, not an equivalence test.

## Consequences for the limited-memory study

- Running on the GPU would cut the study from about 2.9 hours to about 1.8 hours ([computation plan](limited-memory-proposal.md#computation-plan)).
- GPU arms cannot be compared with the earlier CPU runs. Six processes ran with the 15 GB machine otherwise idle; peak
  host memory was not recorded for the whole job, only per process (about 2.5 GB, part of it shared by the fork).
- The saving is about an hour. Whether it is worth a second environment and the composition-measure watch is for the
  researchers to decide.
