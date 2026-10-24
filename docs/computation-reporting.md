# Computation reporting for future studies

Effective 2026-10-24. Applies to new experiment contracts and runners and
substantive extensions. Historical frozen results remain unchanged.

Accuracy, independent-example efficiency, and computational efficiency are
separate claims. Report a quality–cost frontier; a win at equal examples
does not establish a win at equal compute.

## Before a run

Declare dimensions, sample counts, code rank, sparsity/support size, number
of candidate groupings, update count, batch size, solver tolerance, and
maximum iterations. Derive time and storage orders for construction,
selection, training, update, and inference separately. Name the algorithm:
full eigendecomposition, truncated eigensolver, and iterative settling have
different costs. Give operation counts when feasible, labeling an estimate
as such; do not equate a parameter count with a FLOP count.

Account for parameters, persistent learned state, optimizer state,
sufficient statistics, replay, and peak temporary workspace separately.
State units and dtype. Include total retained bytes and whether raw samples
are retained. A covariance is learned history, even when it is not replay.

## At execution

Write a separate runtime JSON containing:

- source/contract hashes, revision and dirty status, command, seed policy;
- interpreter, library versions, OS, CPU/device, numeric dtype, BLAS backend,
  thread counts, worker counts, and relevant hardware limits;
- wall seconds and process CPU seconds by phase; for parallel runs aggregate
  worker CPU time or explicitly mark it unavailable (parent CPU is not total);
- process peak RSS with units and platform convention; distinguish the parent
  maximum from whole-job memory; accelerator peak memory where applicable;
- executed fits, candidates scored, independent examples, exposures, optimizer
  steps, solver iterations, convergence failures, and inference calls;
- total selection/tuning cost including rejected candidates, confirmation cost,
  and deployment cost after selection. Include feature/covariance construction,
  probes, validation, and compilation in named phases rather than hiding them.

For short inference/update timings, warm up, repeat, and report repeat count,
median and spread. Disclose whether allocation, batching, transfer, compilation,
and synchronization are included. For a one-off analytical script, whole-run
timing with an explicit scope suffices; it is not an inference benchmark.
Use null plus a reason for unavailable measurements; never substitute zero.

## Comparison and interpretation

Compare quality at equal independent examples and plot/report quality versus
total training/selection time, inference latency and persistent bytes. Account
for validation examples too. For repeated deployment report amortization:
`total_cost(Q) = construction + selection + training + Q * inference`.
If iterative modules are used, count their work to a common error tolerance
and report truncated/nonconverged cases. A direct equivalent is mandatory
when available. Speed claims require representative dimensions and measured
scaling, not extrapolation from a 32-dimensional task.

Keep scientific JSON deterministic and timing JSON separate. Small analytical
checks should record their own cost, but need not run a large timing sweep.
