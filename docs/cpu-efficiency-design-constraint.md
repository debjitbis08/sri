# CPU efficiency as an architectural constraint

2026-11-01. Research-priority clarification from the user: CPU efficiency is
fundamental, and potentially harder than assembling the proposed capabilities.
This is not a new experiment contract, a selected implementation, or permission
to launch a benchmark sweep.

## Current position

The project has measured computation, but has not established a CPU-efficient
architecture at a meaningful capability scale. CPU viability must constrain
representation, reader execution, memory addressing, context and learning
choices; it cannot be deferred until after adding larger encoders and attention.
Representability and CPU feasibility are independent obligations.

Existing evidence:

- [CPU/GPU benchmark](cpu-gpu-benchmark-v1.md): a simplified training step strongly
  favored the GPU (about 5.8x versus six CPU threads at batch 270).
- [Real fan-in timing](gpu-timing-fan-in-v1.md): aggregate throughput improved
  about 1.6x with six GPU processes against the historical six-worker CPU run.
  This was one arm, different run batches and substantial host overhead; it is
  not an inference benchmark or a general CPU/GPU scaling claim.
- `experiments/fan_in_pilot.py::MaskedLinear` multiplies a dense weight tensor by
  a mask and calls a dense linear operation. Sparse connectivity does not imply
  skipped arithmetic in that implementation. Reader later layers and encoders
  remain dense, and RigL growth can collect dense gradients.

Historical artifacts remain unchanged. The benchmark's outstanding real-pilot
question was answered by the later timing note, within that note's limited scope.

## Three separate resource targets

1. **Inference:** latency and sustained throughput for the intended interaction,
   including perception, context, memory search, repeated core calls and outputs.
2. **Continued learning:** work and temporary state per new experience, including
   optimization, rewiring, replay if any, and background consolidation.
3. **Initial construction/training:** pretraining and model/feature selection.

CPU inference alone does not establish CPU-feasible continual learning or initial
training. Reusing a pretrained sensory model may reduce local training, but its
inference cost and provenance must remain visible. No decision to allow or forbid
off-device initial training was made in this discussion.

## Design questions that follow

- **Actually executed work:** do inactive connections/units avoid arithmetic,
  or merely carry zeros in dense tensors? What is the cost of finding the active
  set, and of maintaining it during learning?
- **Memory traffic and locality:** how many bytes are fetched per response, and
  are weights/addresses visited in a predictable layout? Fewer arithmetic
  operations need not win if gathering and bookkeeping dominate.
- **Growth with stored experience:** selecting a few memories is not cheap if
  finding them scans the whole store. Count index construction, maintenance,
  lookup, misses and any approximation error.
- **Context cost:** bound or amortize access to history and measure the quality
  sacrificed by compression, windows or selective retrieval. Naming attention
  or ATLAS is not a resource budget.
- **Learning locality:** determine whether one new fact touches a small amount
  of state or causes a global forward/backward pass. Local memory writes do not
  establish cheap global consolidation.
- **Whole-system bottlenecks:** sensory processing and output generation may
  dominate even if the relational core becomes cheap.

These are engineering hypotheses to profile, not claims that irregular sparsity
always loses or that dense matrix operations should be excluded irrespective of
measured cost. Structured sparsity, compact dense modules, batching, quantization,
cached features and bounded active sets are candidates, not selected solutions.

**[derived]** If the reader accounts for fraction f of total latency and is
accelerated by s, with all other costs unchanged, whole-system speedup is
1/((1-f)+f/s). Even eliminating reader cost cannot exceed 1/(1-f). Profiling the
fraction is therefore necessary before spending effort on a sparse kernel.

## Next measurement obligation, when authorized

Declare representative hardware and workloads before choosing a latency/RAM/
update budget; no numeric user requirements have yet been supplied. Separate
interactive batch-one latency from batched throughput and training throughput.
Profile the current system by phase and compare dense versus genuinely sparse
execution at matched outputs or declared quality. Sweep relevant code width,
active fan-in, context length and memory size instead of extrapolating tiny-task
timings. Compare quality versus total CPU seconds, latency distribution and
persistent/peak bytes. Include all selection/tuning costs and distinguish
historical missing data from zero cost.

The governing standard remains [computation reporting](computation-reporting.md).
This note adds no measurements or speed claims and records no new external
literature findings; any substantive optimization contract needs its own prior
work and theoretical/measurement plan.
