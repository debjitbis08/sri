# Sparse updates when most of the shared-surface input stays unchanged

**Status:** analytical check, 2026-10-24. No sampled experiment or timing
benchmark. Extends [related-input-analysis.md](related-input-analysis.md).

## 1. The opportunity

If the module keeps its previous settled state, a new input does not require
computing the entire state from zero. Only the change needs to be resolved.
If input, internal constraints, and context are all unchanged, an exact
equilibrium requires no further numerical updates. Detecting changes and
maintaining physical state can still have a cost.

For a fixed quadratic system `Hu=f`, let the old equilibrium be `u0` and
the change in external forcing be `delta_f`. The exact new response obeys

```
H delta_u = delta_f,      u_new = u0 + delta_u.
```

This is the mathematical basis for reusing state. It does not by itself
guarantee sparsity: the inverse of a sparse connected matrix is generally
dense. A small change can eventually affect the entire network. Local or
approximately local response requires additional structure.

## 2. Exact sparsity in the current coarse hierarchy

Use the 32 -> 8 -> 4 aggregate hierarchy, with unit `lambda,kappa,eta` and
fixed coarse context. Let `A,B,C=BA` be its averaging restrictions. For an
input change `delta`, the exact incremental state is

```
delta_c = C delta
Delta x = delta - 1/4 C^T delta_c
Delta z = A delta - 1/2 B^T delta_c
Delta y = 1/4 delta_c.
```

This can be evaluated from changed entries and cached aggregates. It needs
neither a full new input projection nor a full equilibrium solve.

| Input change | Surface states changed | Middle states changed | Core states changed | Total, out of 44 |
| --- | --- | --- | --- | --- |
| Nothing changes | 0 | 0 | 0 | 0 |
| Add to one site and subtract equally from its neighbour within a four-site group | 2 | 0 | 0 | 2 |
| Add to one site and subtract equally in the next four-site group, same coarse group | 2 | 2 | 0 | 4 |
| Change a single site | 8 | 2 | 1 | 11 |
| Change all eight sites of one coarse group equally | 8 | 2 | 1 | 11 |

For a unit change at site zero, the surface response is `31/32` at that
site and `-1/32` at the other seven sites in its coarse group. Its two
middle-state changes are `3/8` and `-1/8`, and its core change is
`1/(4 sqrt(8))`. Everything in the other coarse groups stays unchanged.

The example has four independent coarse groups when context is fixed.
That separation is why a one-site change has an exactly bounded effect.
Adding couplings between cores, overlapping broad features, or a global
constraint can spread the response further. These counts are not universal
claims about circular or recurrent networks.

Changed-state counts are also not operation counts. Detecting cancellation,
finding the affected region, updating it, and checking convergence all cost
work. The cancellation examples assume changes arrive as a known batch;
processing them separately can cause temporary internal updates before they
cancel. A conservative graph-based scheduler may visit all 11 states of a
group even when only two ultimately change.

## 3. A safe way to stop local computation

For an approximate incremental solution `v`, maintain its global residual

```
r = delta_f - H v.
```

If `m` is the smallest eigenvalue of the positive-definite `H`, then

```
||delta_u-v||_2 <= ||r||_2/m.
```

Here `m=0.3819660`. The residual supplies an explicit error bound instead
of assuming that an inactive unit is irrelevant. Updating one coordinate
changes residuals only at that coordinate and its directly connected
neighbours, so a sparse scheduler can maintain residuals incrementally.
Nonzero residuals at the edge of an active region indicate that the region
needs to expand.

Simply updating the changed input site is incorrect in the one-site case:
the analysis finds full-state error 0.4465 and residual norm 0.5292. Updating
the correct 11-state region gives the same solution as the full system,
within numerical precision, and a global residual below `1e-12`.

The script uses analytically known supports to verify these results; it is
not an implemented efficient support-discovery algorithm. For future
nonlinear models, a local Hessian alone is not a global certificate unless
the relevant convexity and stability conditions are established.

## 4. Is this an advantage over an encoder-decoder?

Against recomputing an entire feed-forward network on every frame, retained
state and local updates can save work. But the fair comparison is an
encoder-decoder with cached activations, sparse connections, a surface skip
connection, and propagation of changes rather than absolute activations.

For the present linear model, that cached network evaluates the three exact
increment formulas above. It produces precisely the same new surface and
internal states as the equilibrium loop. There is no established advantage
of iterative settling over that direct incremental evaluation; iterations
could make the loop more expensive.

The shared surface can be a natural interface for continuing external and
internal interactions: state remains available, and a new constraint causes
local inconsistency that propagates until resolved. Whether this organization
beats an equally incremental feed-forward or recurrent baseline is an
implementation and workload question, not a consequence of drawing a loop.

## 5. Conditions required for real computational savings

- **Persistent state:** do not reset the module for every observation.
- **Changes are accessible:** a full dense comparison to discover which
  inputs changed is itself linear in input size; event-producing inputs can
  avoid that scan.
- **Local dependencies:** sparse input does not guarantee sparse response
  when broad features or constraints connect the entire surface.
- **An incremental scheduler:** a dense optimizer or dense matrix solve still
  does dense work, even if most final states barely change.
- **Accurate residual accounting:** many individually small changes can
  accumulate. Thresholding must track the remaining inconsistency rather
  than silently discarding each small event.
- **Incremental output:** emitting a fresh dense output array every time
  incurs a full-array cost even if inference only updates a few entries.

The equilibrium may also need updates when the input stays fixed but coarse
context, another module, or an internal parameter changes. Reusing input
alone is not a sufficient stopping condition.

## 6. Verification and proposed comparison

Run

```
.venv/bin/python -m experiments.incremental_surface_analysis
```

The output is [incremental-surface-analysis.json](../reports/incremental-surface-analysis.json).
For each example it compares a full solve, the cached encoder/decoder
increment formula, and a solve restricted to the analytically affected
states. All agree within `1e-12`; restricted solutions are checked against
the global residual. It also verifies the error bound for an insufficient
one-site update. No running experiment output is accessed.

A useful next computational experiment should compare:

1. a full recomputation baseline;
2. a cached encoder-decoder using exact change propagation;
3. a persistent equilibrium module with local residual-driven updates.

Use identical input/context streams, target outputs, and error tolerances.
Include fine redistributions, one-site changes, broad changes, and nearly
global changes. Measure actual arithmetic or edge updates, residual-check
cost, latency, and memory; do not substitute changed-state counts for work.

The analytical result is that **sparse incremental computation is possible
in the current hierarchy**. Its source is preserved state and structured
dependencies. The shared-surface loop is one way to organize those properties,
but it has not yet been shown to outperform a comparably incremental network.
