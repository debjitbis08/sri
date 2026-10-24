# Single-module learning: analytical preflight specification

**Status:** development preflight specified before generating its finite-sample
results, 2026-10-24. Authorized scope is analytical equivalences and spectral
estimators only. No neural suite, PyTorch dependency, retained-context learning
rule, or nonlinear module is implemented here. Results inform candidate
selection and cannot be called confirmatory evidence for that selected candidate.

## 1. Questions

1. Are the learned linear hierarchy and its matched feed-forward/compact
   controls functionally and gradient-equivalent?
2. How quickly do a global spectral estimator and a locality-constrained
   spectral estimator learn stable structure from related observations?
3. What sample-efficiency benefit does locality supply when correct, and
   what irreducible cost does it impose when wrong?

## 2. Exact equivalences

Use 32 surface, eight intermediate, four core coordinates. Each row of `A`
is a unit vector supported on its disjoint four-site group; each row of `B`
is a unit vector supported on its two corresponding intermediate sites.
Then `C=BA` has orthonormal rows, one per disjoint eight-site group.
Conversely every such compact `C` factors into these `A,B`: normalize its
two four-site segments into `A`, and put their lengths into `B`. Zero-length
segments receive an arbitrary fixed unit row and zero coefficient.

For the previously declared quadratic energy, compare its full spatial
equilibrium with the explicit feed-forward formulas for surface, middle,
and core states. Check arbitrary learned restrictions, several positive
couplings, no context and fixed context. Check finite-difference gradients
of the same core paired loss through both realizations. This is a numerical
check of an algebraic identity, not a training experiment.

The compact local class has `4*(8-1)=28` continuous degrees of freedom.
The hierarchy has `8*(4-1)+4*(2-1)=28`; signs are discrete symmetries.
The unrestricted four-dimensional subspace of R^32 has `4*(32-4)=112`
degrees of freedom. These counts do not themselves prove a sample bound.
They identify the source of a plausible locality advantage.

## 3. Generating tasks and paired observations

Use 64 independent task worlds and eight independent training streams per
world. World index `w=0..63`; stream index `r=0..7`. Dimensions are `d=32`,
`m=4`. Each world has an aligned orthonormal `U` with one column supported
on each eight-site block. Generate each eight-vector by drawing three
standard-normal coefficients in the orthonormal DCT-II modes 0,1,2, then
normalize. The DCT basis is a generator choice, not given to estimators.
This creates smooth within-block templates with unknown shapes and signs.
It is a tailored synthetic development task, not a general learning benchmark.

Derive three versions of each world:

- **Aligned:** the original four supports.
- **Shifted:** circularly translate the entire 32-vector by four sites.
- **Dense:** apply a random orthogonal 32x32 matrix from QR decomposition,
  with diagonal signs of R made positive. This is a nonlocal stable-subspace
  stress test, not a claim that dense rotated patterns are spatially broad.

For each training stream, generate 256 independent rows of `c~N(0,I_4)`
and two independent arrays `e1,e2~N(0,I_32)`. Observations are

```
h1 = U c + sqrt(nu) e1
h2 = U c + sqrt(nu) e2,        nu in {0.25,1}.
```

Both views share only `c`. Their known population mean is zero; do not
estimate/subtract a sample mean. Use nested pair budgets
`N in {4,8,16,32,64,128,256}`. Rotate both signal and nuisance for shifted
and dense worlds. This preserves global statistical difficulty exactly
while changing alignment with the imposed local supports. Use the same
underlying draws at both nuisance strengths.

Generators: `default_rng([20261101,0,w])` draws the four three-component
template vectors, then the dense 32x32 rotation seed matrix; QR requires no
further randomness. `default_rng([20261101,1,w,r])` draws c, e1, e2 in that
order. Gate inputs use a separate `[20261101,2]` generator. These seeds
are development seeds; future neural confirmation must use new tasks/data.

## 4. Estimators and population references

Estimate the symmetric cross-covariance

```
M_N = (H1^T H2 + H2^T H1)/(2N).
```

- **Global spectral:** top four algebraic eigenvectors of `M_N`, not the
  largest absolute eigenvalues.
- **Local spectral:** top eigenvector of each fixed diagonal 8x8 block,
  giving four disjoint normalized directions.
- **Fixed averaging:** normalized constant vector in each eight-site block;
  it sees no data and has no fit.
- **Population global:** true subspace `range(U)`; unattainable ground-truth
  reference, not an estimator receiving privileged training information.
- **Population local:** top eigenvector in each block of `S=UU^T`; the
  optimal member of the declared local class. Its nonzero error is the
  locality approximation floor.

The spectral objectives maximize `trace(V^T M_N V)` over their respective
orthonormal classes. Embeddings cannot collapse to zero. Eigenvector signs
do not affect the projector or any metric. Population boundary ties do
not affect the optimal value; sampled task worlds almost surely avoid them.

## 5. Metrics and uncertainty

Primary preflight metric is normalized subspace loss

```
L = 1 - ||V^T U||_F²/4 = ||VV^T-UU^T||_F²/8.
```

This lies in [0,1], measures clean signal power lost on independent new
coefficient combinations, and uses the ground truth only for evaluation.
For an independent test observation with isotropic nuisance variance
`nu_test`, orthogonal reconstruction has exact relative MSE `L+nu_test`.
Report predictions for `nu_test=nu` and `4nu` without drawing test data.
This deliberately does not treat subspace loss as a universal metric for
future nonlinear models.

Report mean loss, its range across independent task-world means, and
mean +/- 1.96 standard errors across the 64 world means (unclipped normal
approximation, not a guaranteed finite-sample confidence bound). Average
the eight streams within a world first. Budgets, models, and alignments are
paired; compute uncertainty on paired differences using the same grouping.
Do not treat every budget or every stream within a world as a new task.

Report first budget with mean loss <=0.10 as a descriptive threshold, or
`not reached by 256`; do not interpolate it into a proved sample-complexity
ratio. Also report excess over the population local floor. Store world
means, not only pooled summaries, so uncertainty can be recomputed.

For an exact finite-sample covariance reference, let `S=UU^T`, `T=S+nu I`.
Gaussian fourth moments give

```
E||M_N-S||_F² = [(tr T)² + tr(T²) + (tr S)² + tr(S²)]/(2N).
```

For d=32,m=4 this is `(20+132nu+528nu²)/N`. In the aligned local class,
the sum of errors of its four retained covariance blocks is
`(8+36nu+144nu²)/N`. Report the empirical covariance errors against these
references. These are covariance moment identities, not equalities for
subspace error; converting them to eigenspace bounds also needs eigenvalue
separation. Global population gap is one; local gaps are reported by regime.

## 6. Correctness gates and reproducibility

Before the finite-sample run, require:

- equivalence of hierarchy, compact factorization, and explicit equilibrium
  within `1e-9`, including zero-segment factorization;
- finite-difference paired-loss gradients through explicit and spatial
  equilibrium maps agreeing within `1e-6`;
- orthonormal generative columns, preserved covariance under each rotation,
  and global loss invariant under joint rotation within `1e-9`;
- noiseless local recovery from one nondegenerate paired example, and
  noiseless global recovery from four independent spanning examples;
- local population eigenvectors attaining the sum-of-block-leading-eigenvalue
  optimum, and every local estimate's population loss above that floor;
- a full-budget estimator unaffected by unrelated future draws; proper
  use of the declared prefixes and no use of U in fitting;
- deterministic canonical JSON, pinned environment, one BLAS thread,
  recorded source/specification hashes, and byte-identical regeneration.

The finite covariance moments are reported as diagnostics rather than a
fragile Monte Carlo pass/fail gate. Runtime is separate from deterministic
results. Existing experiments and their artifacts are not modified.

## 7. Decision rule after preflight

Use the numbers to recommend a candidate, but do not start the neural suite:

- If the maps and their exact gradients coincide, do not nominate the
  unchanged linear equilibrium model as architecturally distinct from its
  matched feed-forward counterpart.
- If locality reduces finite-pair error when aligned but introduces an
  irreducible misaligned floor, attribute that tradeoff to the locality prior,
  not circularity or depth. Carry the matched local compact model forward
  as a mandatory control.
- A new retained-context/local learning rule or nonlinear energy requires a
  separately specified mechanism and analysis. Do not claim it is distinct
  merely because it has a different implementation: the FF baseline receives
  the same information and may reproduce the same rule.
- If the spectral estimator already solves the development problem well,
  use it as a strong reference rather than designing a weak comparison.

The resulting recommendation must state which proposed learning advantage
is supported, which remains untested, and which additional design choice
is needed before any neural implementation.
