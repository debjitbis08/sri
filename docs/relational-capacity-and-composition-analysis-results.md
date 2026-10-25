# Relational capacity and composition: analytical results

2026-10-25. Read the [main derivations](relational-capacity-and-composition-analysis.md)
and the [supplementary weak-coupling bound](relational-composition-weak-coupling-bound.md).
No capacity fitting, learning run, or experimental architecture selection occurred.

## Findings

- Rank-two terminal retention has a unique masked equilibrium with the current
  positive cycles and injective radial maps, despite removing the terminal penalty.
- A separate output module coupled through complete terminal ports also admits
  a sufficient positive-definiteness proof. Fixed-route transfer rank is bounded
  by cross-module coupling rank. This makes the communication interface explicit.
- Fixed quadratic compositions remain linear and cannot pass the task's known
  affine bound. Extra dimensions and interaction must be tested separately from
  the source of nonlinear relational behavior.
- Passive unobserved helpers cannot enlarge the zero-penalty subspace of an
  unchanged existing state. This does not rule out useful changed correlations,
  newly exposed outputs, or adaptive inference.
- A later analytical extension bounds output error under weak terminal coupling,
  even with input-dependent routes. Thus composition does not automatically
  evade the isolated-module obstruction. See its exact assumptions and threshold.
- Uniform relative-rotation transport around four parts is inconsistent for
  relations 1 and 3. Correct boundary transport restores a two-dimensional
  consistent family. This is a proposed-connection/task mismatch, not evidence
  against circular organization.
- Vector equivariance requires compatible lateral connections, radial maps,
  observation operators and local frames. The present scalar cycle alone does
  not provide it.

## Numerical checks

[Runner](../experiments/relational_composition_analysis.py),
[scientific receipt](../reports/relational-composition-analysis-v1.json),
[runtime](../reports/relational-composition-analysis-v1-runtime.json),
[repeat receipt](../reports/relational-composition-analysis-v1-repeat/relational-composition-analysis-v1.json),
[repeat runtime](../reports/relational-composition-analysis-v1-repeat/relational-composition-analysis-v1-runtime.json).

All finite checks passed in both executions. Scientific JSON was byte-identical
(SHA-256 `9c84aafca96f0836854dad5825680cb9ec95b0c321a32f140f6936f304e3b8c9`).
Two map seeds times two retained ranks gave four coupled fixtures, all SPD.
Coupled condition numbers were about 1,742–2,678; this is a fixture observation,
not a uniform conditioning theorem. Maximum full/Schur discrepancy was
1.34e-15 and maximum relative solve residual 3.06e-15. Complete terminal ports
gave transfer rank eight in all fixtures; no task-completion score was evaluated.

Transport-cycle checks gave nullity zero for uniform cycles at relations 1/3,
nullity two at relation 2, and nullity two for all corrected boundary transports.
The three open-chain relation spaces jointly span six dimensions, while their
positive-penalty conjunction has zero nullity. An explicit invalid mixture is
saved. The scalar-cycle rotation commutator norm was 2.5053, versus zero for the
vector-channel cycle fixture. Passive-helper elimination and frame-change
identities passed to numerical precision.

The supplementary weak-coupling theorem was derived after these checks. It is
supported by its explicit algebra; it was not included in the numerical receipt.

## Computation and reproducibility

First execution: 0.1884 wall seconds, 0.1838 process CPU seconds, peak RSS
58,716 KiB. Repeat: 0.1915 wall seconds, 0.1869 CPU seconds, peak RSS 58,996 KiB.
Combined executions cost 0.3799 wall seconds and 0.3707 CPU seconds; phase times
are components, not additional totals. Timing covers imports and provenance,
excluding final runtime serialization. There were no fits or tuning candidates.

Each execution performed 13 symmetric eigenvalue decompositions, 15 dense
solves, and 23 SVDs, with dimensions in the receipt. Fixture initialization
additionally used 24 QR decompositions and eight terminal-vector normalizations
through the existing random-parameter helper; these are included in runtime
but not in the runner's decomposition counter. Largest single declared Hessian
storage was 204,800 bytes. Native LAPACK workspace and FLOPs were not measured.
No optimizer, replay or learned persistent state was retained.

Reproduce in a fresh output directory:

```
.venv/bin/python -m experiments.relational_composition_analysis --output-dir /tmp/sri-relational-check
```

The runner refuses overwrites. All source hashes, Python/NumPy/SciPy versions,
BLAS configuration, single-thread settings, revision and dirty paths are recorded.
The environment emitted NumPy's optional-PyYAML formatting warning; it did not
affect computation or checks. This is not an inference benchmark.

## Next decision

The minimal rank-two control is analytically ready for a separately frozen
capacity contract. A two-module terminal-coupled candidate has a certified
solvable interface, but its routes and coupling budget still need specification.
Use the new weak-coupling bound to exclude futile coefficient ranges, not to
select a presumed successful coefficient. A designated output preserves the
repository's evidence/output separation; summed outputs are a separate control.

Relational/geometric arms need an explicit mechanism for representing different
relations without forcing incompatible constraints simultaneously. A shared
transient relation state is one candidate, but would require nonlinear inference
analysis. Supplying the correct relation or its analytic extractor is a privileged
reference, not a learned-abstraction result. Integrate the composition/task audits
before freezing those arms; do not launch the entire proposed factorial yet.
