# Stored-pattern hierarchy v1 results

This innermost-only coupling run was already executing when the user clarified
that all corresponding layers must couple with increasing strength inward.
It is retained as a control, not presented as the requested final architecture.
See [the graded study](standing-pattern-hierarchy-v2-contract.md).

Two 8→4→2→1 modules learned local quadratic preferred-detail functions from nine
paired signals. Eight held-out midpoint signals entered only the input outer
layer. Output was the continuous state of the distinct output module's outer
layer, without an inner label or external learned decoder. Grouping, polynomial
family, alignment and scalar generative task were supplied.

| Model | Output RMSE at step 64 | Step 1024 | Step 2048 |
| --- | ---: | ---: | ---: |
| Learned quadratic, frozen | 0.48900 | 0.10037 | 0.02745 |
| Learned quadratic, adapting | 0.48901 | 0.10367 | 0.03404 |
| Learned affine, frozen | 0.47348 | 0.15932 | 0.13761 |
| Zero preferred detail | 0.51202 | 0.26789 | 0.25804 |
| Uncoupled quadratic | 0.50574 | 0.50574 | 0.50574 |

Quadratic output started at RMSE 0.50574. All eight coupled quadratic queries
still had free gradient norm above 1e-5 at step 2048: useful improvement preceded
settling. This is not proof of convergence. Online parameter adaptation slightly
worsened task error. Exact direct reconstruction from the input mean gave RMSE
1.33e-16 with quadratic preferences, 0.13620 with affine preferences and 0.25748
without detail. The direct solution makes no iterative-computation advantage claim
possible here; the experiment isolates representational and dynamical behavior.

Gradient error was 1.35e-10 and coefficient fitting error 3.33e-16. Independent
module defaults are stationary with no evidence. The functions are identifiable
in this deliberately realizable data family; this is interpolation, not discovery
of grouping, arbitrary patterns or useful continuous learning from scratch.

The run took 18.04 seconds including checks, 28 fits, all probes and repeated
benchmarks. Full phase timings, benchmark spreads, counts, hashes and memory are
in [runtime JSON](../reports/standing-pattern-hierarchy-v1-runtime.json);
[scientific JSON](../reports/standing-pattern-hierarchy-v1-results.json) preserves
every outer readout. Costs and limitations are specified in the
[prospective contract](standing-pattern-hierarchy-v1-contract.md).
