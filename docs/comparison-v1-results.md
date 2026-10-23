# Comparative modeling v1 — results and grades

**Status: complete within the [declared contract](comparison-v1-contract.md).**
Four representatives were implemented and evaluated in 50 parameter/input cases.
This is not an exhaustive comparison of mathematical structures or a selected
SRI architecture. All four omit learning and fail to establish an autonomous,
selective structural mediator. No historical certificate was modified.

The [machine-readable certificate](../reports/model-comparison-v1.json) records
configurations, controls, counterexamples, grades, exact errors, and source hashes.
The [separate runtime receipt](../reports/model-comparison-v1-runtime.json) records
elapsed time and peak Python allocation for this instrumented certificate run.

## Evidence profile

“Recovery” below means the declared driven recovery test, not robustness under
all schedules. S and C have explicit simultaneous dynamics and certified limits;
A and R use a specified joint reconstruction procedure. Their costs and native
disturbances are not equivalent. “Mixed” means results vary within the bounds.

| Candidate | Distinct driven outputs | Context changes response | Driven recovery | After withdrawal | Native approximation |
| --- | --- | --- | --- | --- | --- |
| S: grounded spectral operator | Pass, all 5 couplings | Pass for positive context coupling | Pass, contraction certificate | All modes decay to zero | Residual and context-omission bounds pass |
| A: affine F3 system | Pass, both contexts | Pass | Pass, all 81 starts per case | Free-coordinate retention; damage to that coordinate is not repaired | Ordered weak coupling unspecified |
| R: graded parity relation | Mixed: 7 of 8 evidence-pair settings have unique distinct outputs | Changes unique outputs below g=1; ambiguous at g=1; unchanged above | Unique minima recover; competing minima fail the all-source test | Selected minima persist by explicit retention; recovery fails for some responses | Tiny omitted preference can change the output bit |
| C: cellular Hodge operator | Mixed: 9 of 10 attachment/weight settings distinguish inputs | Positive-weight attachment change affects response | Pass, clamped-input contraction | Harmonic response persists for constant attachment; disk with positive weight erases it | Driven residual bounds pass; omission can change the infinite-time limit |

These are common questions, not a common inference benchmark. Only declared
input labels and observation interfaces are shared. Analog amplitudes, bits,
field values, and storage precision have different meanings. The detailed
profiles leave unsupported topology, nesting, and learning claims unresolved
or not claimed; no overall numerical score is assigned.

## S: correction without post-input memory

With coordinate order `(I,K,O,S)`, exact elimination gives

```
K = e*(1+g)/(4+6*g)
I = (e+K)/2
O = K/2
S = g*K/(1+g).
```

Every declared g separates e=-1 from e=+1. Positive coupling to the retained
context satellite changes O. Removing all core pair terms gives O=0.
This is coupled linear reconstruction, not nonadditive structural mediation.

The quadratic matrix Q is the identity plus a nonnegative weighted graph
Laplacian. Hence `z^T Q z >= ||z||_2^2`. In the declared grid, `I-Q/16` has
nonnegative entries and row sum `15/16`, so

`||error_t||_infinity <= (15/16)^t ||error_0||_infinity`.

All 80 single-coordinate ±1 trials satisfy this exact bound and the declared
128-step tolerance. On withdrawal b=0, the unique equilibrium is zero; no input
distinction persists. Eigenspace diagnostics verify the spectral interpretation,
but exact rational solves and bounds certify the responses.

## A: persistence is not reconstruction

The unique driven solution over F3 is

`(I,K,O,S) = (e, 2e+t, -e-t, t) mod 3`.

All 324 starts (81 for each of four cases) reconstruct it. Every driven
single-coordinate replacement is corrected by the declared projection.
The constraint filtration has direction-space dimensions 2,1,0.

After withdrawal, O becomes a free coordinate under the specified pivot order.
The selector keeps its previous value. This preserves the response, but changing
O changes the remembered value permanently. Each released response has explicit
counterexamples. A different pivot/selection policy could yield different memory
behavior; persistence is not a theorem about F3 itself. Field coefficients do
not supply a graded sensitivity order.

## R: higher arity, ties, and weak decisive preferences

For context t=0, the unique minimum has K=0 and O=I. For t=1, the two possibilities
have costs g (K=0, O=1-I) and 1 (K=1, O=I). Thus g=1 is an exact selection boundary,
not a numerical artifact. At that value both outputs are admissible minima.

Checking only the default selected state would hide a recovery failure. The
certificate disturbs **every minimum**, including the alternative minimum;
some disturbances return to a different output through the lexicographic rule.
Keeping the current minimum after withdrawal preserves states but does not
guarantee reconstruction of every retained response.

All four three-coordinate projections of the parity relation are unrestricted.
Recombining them admits 16 tuples instead of 8. This is an exact witness that
the specified four-way condition cannot be replaced by its lower-order
projections on the same variables.

Scale costs to `K/1024 + 2*(O xor I)/1024`. With I=0 and S=1, omitting the second
term changes O by 1 despite its weight being only 1/512. The objective perturbation
is small, but categorical response error is maximal. A useful output guarantee
therefore needs additional information such as a decision margin; weight magnitude
alone is insufficient. The constructor inspects all 16 tuples, openly charging
the global enumeration to this tiny control.

## C: a real topological effect, with an exposed limitation

Both cellular structures have three vertices, three oriented edges, and one
2-cell. One attaches the 2-cell constantly at a vertex; the other fills the
triangle. Their graph skeletons and cell counts agree. With J the all-ones 3×3
matrix, the operators are

```
constant attachment: L = 3*Identity - J
disk attachment:     L = 3*Identity + (g-1)*J.
```

Both chain complexes satisfy d1*d2=0. Their first Betti numbers are 1 and 0;
their second Betti numbers are 1 and 0. Positive-weight kernel dimensions agree
with the first Betti numbers. At g=0 the disk's face term is ignored, so its
operator has a spurious harmonic direction relative to the full topology.
This is why an operator kernel is not an unconditional hole-count certificate.

Clamping I=e gives O=K=e for constant attachment, and
`O=K=e*(1-g)/(1+2*g)` for the disk. At g=1 the disk's output is zero for both
inputs, even though internal evidence differs. More coupling does not uniformly
improve distinguishability.

On withdrawal, constant attachment preserves the mean and damps the zero-sum
component; a positive-weight disk damps both. A disturbance of one edge by delta
changes the retained mean by delta/3. Therefore the persistent harmonic response
does not recover its original value. All 120 driven disturbance trials satisfy
their contraction bounds, but driven correction must not be transferred to
released memory.

Omitting an arbitrarily small positive face weight changes the infinite-time
limit from zero to the conserved mean. This is not a failure of the finite-time
driven error certificate; the limits and claim differ. No ordinary consumer-CPU
advantage or semantic benefit follows merely from this topological effect.

## Approximation certificates and resource limits

For each strictly diagonally dominant response matrix Q with margin
`delta=min_i(Q_ii-sum_{j!=i}|Q_ij|)>0`, the exact error of an approximation a obeys

`||a-z_exact||_infinity <= ||Q*a-b||_infinity/delta`.

Proof: choose a maximum-magnitude component of the error and apply the triangle
inequality to that row. S has delta=1; C's driven two-coordinate matrices have
positive delta throughout the grid. All tested omission and spectral-cutoff
approximations satisfy the bound. Numerical spectral approximations are converted
to rational coordinates before this posterior check. This is not a guarantee
that an approximation can be chosen cheaply without inspecting omitted effects.

The report counts elimination arithmetic and tuple inspections in their own units.
The runtime receipt includes exact fractions, checking, report construction, and
allocation instrumentation; it is not a fair inference-speed leaderboard.
Fraction bit lengths grow during iteration. No bounds independent of model size,
precision, active structure, or number of continuation steps are claimed.

## Reproduce and research decision

```
python -m experiments.model_comparison
python -m unittest discover -s tests -v
```

The runner writes only the two new comparison reports. Exact results are checked
against closed forms and independent tiny oracles. Runtime measurements vary;
the semantic certificate regenerates identically in the recorded environment.
Floating eigendiagnostics have explicit tolerance/status and are not proof data.
All 59 repository tests passed in the recorded run, including the earlier frozen
certificates. Passing implementation checks does not turn the modeling failures
in the table into successful hypotheses.

No candidate currently establishes recoverable **distinct post-input responses**
through autonomous selective core mediation. The results nevertheless distinguish
mechanisms: contraction repairs and forgets; a free coordinate retains and drifts;
graded minimization has ties and discontinuous readouts; topology supplies
invariants whose values can still be corrupted.

Keep these four as comparison baselines. An additional structure is not required
to complete this bounded test, and no larger search was used to rescue a failure.
A justified next candidate would need a specific mechanism for reconstructing
which persistent collective mode was intended after disturbance, while preserving
context and a core contribution beyond a generic solver. The comparison does not
yet select that mechanism or settle what an SRI mode or layer should be.
