# After the reset controls: make the code easy to read again

**Literature follow-up:** [forget-and-relearn and iterated learning](forget-relearn-iterated-learning-notes.md)
connects this hypothesis to Zhou et al. (2022), Kirby, Ren et al. (2020), and
Li and Bowling (2019). These are established explanations and controls that
were missing from the initial discussion; their competing interpretations
require measuring immediate forgetting separately from recovery.

2026-10-30. Interpretation of [reset results](fan-in-pilot-v1-reset-results.md)
and the implemented reset/carry controls. No new measurements or training.

## Revision of the hypothesis

My earlier emphasis on P attrition was not supported as stated. P-only RigL
does not recover the main cross-sense gain; g-only gives more of that gain,
and static removal of units does not reproduce it. The stronger claim that
gradient-dependent survival selects shared features is also unnecessary for
substantial transfer: random rewiring with zero initialization achieves .74
English comparison, .80 Devanagari comparison and .98 Chinese comparison.

The most informative observation is strong transfer with only .11/.12 of live
g/P connections on their most-used coordinate. Concentrated wiring is not
necessary for substantial transfer. Yet collection variance is .98 on one
principal direction. Thus low-dimensional *content* and concentrated *wiring*
are different properties, and the former survives removal of the latter.

## A candidate explanation: repeated readout relearning

**Hypothesis:** the encoders are pressured to present quantity in a form that
many changing sparse readers can learn to use. A stable specialized decoder
can compensate for an awkward code. Random rewiring and resets repeatedly
remove that compensation. The network can respond by making the code itself
easier to interpret. Rewiring stops at update 4,500, so final success can also
include fitting a fixed reader to a representation shaped earlier by resets.

This does not require the random choice to recognize what is meaningful. The
matching, comparison and prediction objectives still supply the meaningful
signal. Randomness changes which solutions are convenient to retain and relearn.
It does not create quantity supervision by itself.

For example, a distributed scalar code can be

    c(n) = b + v n.

This is one-dimensional variation even if v has nonzero entries in all sixteen
coordinates. A reader of any coordinate j with v_j≠0 can recover n through a
scale and offset. It need not share the same coordinate as another reader.
With noise, tiny loadings are not useful; near-rank-one covariance alone does
not establish that arbitrary coordinates carry robust quantity information.

### A simple mathematical prediction

For a scalar target n and coordinate c_j with nonzero variance, the best affine
predictor of n from c_j has mean-square error

    min_(a,b) E[(n-a c_j-b)²] = Var(n)(1-Corr(n,c_j)²).

**Derivation:** center both variables; differentiation gives
a=Cov(n,c_j)/Var(c_j). Substitution gives the expression. If j is sampled
uniformly and a separate fitted affine readout is allowed for each coordinate,
the mean error is Var(n)[1-mean_j Corr(n,c_j)²]. Constant coordinates supply
no information and have baseline error Var(n).

This idealized reference favors quantity information accessible through many
coordinates, not just one high-variance direction. It does not prove that the
actual pairwise readers optimize this loss or refit optimally after a reset.
It yields a testable prediction: random-rewire codes should support quantity
readout from more individual coordinates, or support faster recovery of new
sparse readers, than random-fixed codes.

## What the existing controls do and do not establish

- Carrying weights is worse than RigL on the two primary measures, supporting
  the importance of its reset operation. But carry also changes future switch
  counts (about twice as many) and unit death. It is not a fixed-trajectory
  intervention on weight erasure alone.
- Random rewire gives substantial transfer without gradient-based coordinate
  selection. It is not established equivalent to RigL on both primary measures.
  For taps→withheld Chinese, its paired difference is −.15 with interval
  [−.36,+.06]; this does not bound RigL's possible advantage to "a little".
- Static random unit removal is insufficient. This does not prove that selective
  unit loss during learning is irrelevant, or that it cannot interact with
  resets. "Dead units alone are insufficient" is supported; "not a cause" is
  stronger than this control establishes.
- The tested random intervention combines rewiring, zeroing and Adam-moment
  resets. It does not isolate resetting weights on unchanged connections.
- Restricting either reader improves parts of the outcome; the result does not
  support a uniquely privileged P mechanism. P-only's .97 first-PC variance
  and weak cross-sense matching also show that low code rank alone is insufficient.

## Useful next discriminator

Before another large training sweep, freeze encoders and projections from each
arm and train fresh identical sparse readers with a counted, matched budget.
Evaluate recovery curves and transfer, not only their final fit. This directly
tests whether reset training made representations easier for a new reader to use.
Use fresh reader seeds and report all fits/selection costs under a new contract.
A cheaper first diagnostic is collection-trained single-coordinate count probes
applied unchanged to forms and taps, reporting the distribution across all
coordinates rather than selecting the best one. This checks cross-sense
accessibility instead of merely explaining collection variance.

For an intervention on the cause, compare reset-in-place against random
rewire-and-reset at the same specified event rate. That distinguishes loss of
reader memory from changing which features are read. Neither proposed test has
been run here.

Relation to the limited-memory proposal: the results strengthen the case for
disrupting specialized reader solutions, but not the claim that uniform decay
is equivalent to abrupt resets. The relevant question becomes whether its
non-decaying pathway and encoders learn information that remains easy to read
when the temporary reader solution is removed.

All new mathematical claims above are direct algebra under explicit assumptions;
the neural explanation remains a hypothesis. This note adds no novel-method or
external-literature claim. No numerical experiment or new computation receipt
is needed for the symbolic derivation; any follow-up probe or refit requires
its own measured computation and preregistered comparisons.
