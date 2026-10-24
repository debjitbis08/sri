# Analysis 1: when adapting locality is worth its cost

**Status:** analytical development study, 2026-10-24. No new confirmation
data or learned superiority claim. Motivated by benchmark v1's rotated-task
failure. Numerical witnesses use explicit constructed matrices, not selected
benchmark worlds. Verification: `python -m experiments.adaptation_analysis`.

## Exact approximation cost

Let clean observations have covariance `S >= 0` in d coordinates. A code has
k orthonormal rows and projector P; its clean reconstruction error is
`tr(S)-tr(PS)`. For a partition G into k disjoint supports, with one unit
vector per support, the best retained variance is exactly

```
R_G = sum_{g in G} lambda_max(S[g,g]).
R_global = sum_{i=1}^k lambda_i(S).
excess_local_error = R_global - R_G >= 0.
```

Proof: disjoint supports make rows orthogonal automatically, and each block
independently maximizes a Rayleigh quotient. The unrestricted maximum is
the top-k eigenvalue sum. No optimizer or extra examples can remove this
gap within the fixed class. This generalizes the preflight's projector
calculation to anisotropic signal covariance, including correlated factors.

For unit signal eigenvalues, normalized loss is `1-R_G/k`. With d=32,
k=4, and four constant eight-site templates shifted four sites, each old
block contains equal-energy pieces of two independent templates. Its top
eigenvalue is 1/2: fixed-local loss is exactly 1/2. Selecting the correct
partition phase restores loss zero with the same deployed code size.

This correction has a boundary. Take the first four columns of a normalized
32-dimensional Sylvester Hadamard matrix. Each contiguous eight-site window
has restricted covariance eigenvalues 1/4 (four times). Every circular phase
of the eight-site partition has loss 3/4. A global rank-four code has loss
zero. Searching translations cannot solve arbitrary mixing.

## A bounded way to learn the partition

First try a declared finite bank of K partitions, not unrestricted learned
routing. Fit a code within each partition on training pairs; select with
independent validation pairs. Score `z_g = h1^T P_g h2`, whose expectation
is `tr(P_g S)` when view noises are independent of the signal and each other.
This uses only paired observations. Never select by the ground-truth floor.

For zero-mean jointly Gaussian views with marginal covariance T and
cross-covariance S, fixed symmetric P gives the exact identity

```
Var(z_g) = tr(P_g T P_g T) + tr(P_g S P_g S).
```

Condition on the fitted candidates. With n_v independent validation pairs,
Chebyshev and a union bound give simultaneous score error at most epsilon
with probability at least `1 - K Vmax/(n_v epsilon^2)`, clipped at zero.
On that event, selected retained variance is at least the best *fitted*
candidate's variance minus `2 epsilon`. This is a loose sufficient bound,
not a practical sample-size recommendation. It neither eliminates fitting
error nor establishes the benchmark's non-Gaussian conditioned-coefficient
variance formula. For that benchmark use empirical validation uncertainty
or derive its fourth moments separately. Zero population gap between phases
means exact phase identification is unnecessary and may be impossible.

An adaptive method should therefore be evaluated at equal total independent
pairs, including validation. An oracle best-partition result is a ceiling,
not a learner. Compare to global/local ridge CCA selected with the same
budget and to fixed locality. Phase adaptation addresses boundary placement;
it does not yet learn arbitrary feature memberships.

## Computation and alternatives

Let block size b=d/k. The following assumes batch covariance construction
and full dense eigendecomposition per block; CCA changes constants and
requires additional covariance matrices and factorizations of the same order.

| Method | Construction / fitting time | Working statistics | Deployed projection |
| --- | --- | --- | --- |
| Fixed partition | O(n d b + d b²) | O(d b) | O(d) time/weights |
| K partitions, no reuse | O(K n d b + K d b²) | O(K d b), or O(d b) sequentially with rereading | O(d) after one is selected |
| Full global covariance | O(n d² + d³) | O(d²) | O(d k) time/weights |
| Rank-k streaming subspace update | O(d k + d k²) per observation with QR | O(d k) | O(d k) |

The streaming row is a computational alternative, not a convergence claim.
Candidate validation adds O(K n_v d) projection work. Storing all fitted
local codes costs O(Kd), separate from covariance workspace. Input storage
or rereading must be counted. Shared covariance entries can reduce bank cost;
report the implemented reuse rather than crediting it in advance.

If an overlapping dictionary uses r filters each supported on b sites,
projection costs O(rb), but rank and orthogonality no longer come for free.
Forming its Gram matrix can cost O(d r²), with O(r³) dense factorization;
orthogonalization can destroy sparsity. A learned dense rotation adds O(d²)
parameters/application cost and can erase the reason for locality. These
are reasons to test bounded phase adaptation first, not proofs against more
general adaptation.

## Decision

The first experiment should ask whether **data-selected partition phase
recovers shifted-task performance while retaining the aligned low-data
advantage after paying for selection**. Freeze the phase bank and validation
allocation before confirmation. Include arbitrary dense rotations as an
explicit expected failure and global CCA as the escape route. Report oracle
floors, learned excess error, quality versus total pairs, and quality versus
compute separately. Only proceed to overlapping or adaptive-rank groups if
the residual failure can be attributed to their representational limits.

This experiment tests useful adaptation of the locality prior. It cannot
establish an advantage of reciprocal dynamics.

## Verification receipt

[Deterministic report](../reports/adaptation-analysis.json) and
[runtime receipt](../reports/adaptation-analysis-runtime.json) accompany both
analyses. The runner checks 96 eight-dimensional block eigendecompositions:
the shifted witness has fixed loss 0.5 and best-phase loss zero (up to
roundoff); the dense witness has loss 0.75 for every phase. An independent
256-term Gaussian fourth-moment expansion agrees with the score-variance
trace formula. These checks verify examples and identities, not finite-data
success of phase selection. The runtime receipt records actual elapsed time,
CPU time, RSS, environment, work counters and source hashes.
