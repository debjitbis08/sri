# Counting specialization and generalization guarantees

2026-11-01. Discussion and symbolic analysis, not a frozen contract or new run.
Follows [code space versus readers](code-space-reader-analysis.md).

## Audit and claim boundary

The current generic sequence architecture uses a learned real 16-dimensional
code, learned relation-conditioned MLP and learned two-history predictor
(`src/generic_net.py`). The vector space itself does not encode arithmetic.
Earlier heads did supply strong task-compatible structure: distance matching,
affine succession and scalar linear comparison. The [generic-head pilot](generic-relations-v1-results.md)
found taught collection comparison .96 -> .98 while untaught Chinese-form
comparison fell .88 -> .51. This is evidence for the earlier heads' contribution,
not a causal result about current sparse-reader training.

Current counting sequences, same/next/more questions, collection prediction
targets, nine-count candidate sets and repeated design decisions on this task
remain specialized evidence and selection. Different scripts are different
observations of the same latent quantity, not independent relational task families.
The [arbitrary pairing study](arbitrary-pairings-v1-results.md) already shows a
tradeoff: sparse48 recall .34 versus dense .53 at block 4. It still maps symbols
to counts and is not a test of general structured reasoning outside quantity.

The hypothesis class is the set of complete encoder-reader functions (with
allowed masks and memory state), not R^16 alone. The optimizer, resets, rewiring,
losses and curriculum select among those functions. Dense networks of matching
width can represent fixed sparse subnetworks by zeroing weights, but that
containment does not imply they learn the same solution from finite data.

## A falsifiable empirical question

Test the narrower claim: a frozen sparse-reader training recipe improves held-out
relational learning beyond tasks reducible to one ordered scalar. Specify whether
the target is a new example, a new rendering, a new combination, a new value range,
a new task learned from scratch, or few-shot transfer of an existing representation.
These require different tests and should not share one generalization score.

Keep counting as reference. Include cyclic relations (no globally consistent
strict total order), two independently varying factors with held-out combinations,
and branching graph relations. Use perceptual presentations with matched nuisance
variation and comparable evidence budgets. A second ordered magnitude alone is
too close to counting to address the concern. Arbitrary associations are a
specificity/storage control, not a task with inferable unseen random labels.

Cross at least unconstrained vector codes and a learned low-rank bottleneck with
dense, static sparse and disrupted sparse readers. Whole-operand dense disruption
is a competing explanation for earlier sparsity results. Retain task-appropriate
or privileged-latent baselines to check whether each fixture is learnable. Match
examples and report cost frontiers separately; width, parameters and FLOPs are
different controls. Evaluate fresh training and frozen-code transfer separately.

Freeze choices using development tasks; hold entire task instances/families and
their generators' seeds out of architecture selection. Report family-specific
effects and intervals over independent tasks as well as training seeds. A failure
with successful controls falsifies the specified broad advantage, not every
possible sparse architecture. Success supports only the sampled task environment.

## Useful theory, with assumptions

**[derived] A finite-class bound.** Suppose H is fixed before observing n iid
training examples and loss lies in [0,1]. Hoeffding's inequality and a union bound
give, with probability at least 1-delta, simultaneously for every h in H:

    R(h) <= R_hat(h) + sqrt((ln |H| + ln(1/delta))/(2n)).

For a predeclared encoding of at most B bits, |H| <= 2^B, so replace ln |H|
by B ln 2. Encoding must include masks, quantized weights, encoders, readers and
retained state used for prediction; variable-length codes need an appropriate
coding/union-bound treatment. Adaptive selection among encodings also counts.
For a layer with h rows each choosing k of d inputs, possible masks alone cost
h log2 binom(d,k) bits before weights. This is an upper-bound framework, not a
computed certificate for our model. Quantizing requires reevaluating its loss.
An uninformative bound does not establish failure. Repeated exposures or correlated
pairs from one episode cannot simply be counted as independent n.

**[abstract] Capacity and algorithm-sensitive bounds.** VC/Rademacher theory
controls estimation error for a specified loss class and sampling distribution.
PAC-Bayes can use a learned distribution over weights, with a valid prior and
KL penalty, to certify its same-distribution risk. A stochastic-network guarantee
is not automatically a guarantee for its deterministic mean weights. Neither
method removes approximation error or certifies an arbitrary distribution shift.
The full trainable encoder must be included, or learned on independent data,
before a frozen-feature reader bound is used as a whole-system guarantee.

**[derived] A direct non-identifiability test.** If two hypotheses in the actual
class agree on all observable training evidence but disagree on a target query,
that evidence does not logically determine its answer. Constructing such a pair
disproves a guarantee across both possible worlds. It does not show failure under
a specified probabilistic prior favoring one world. A practical version is to
search for equal-training-fit models with divergent held-out answers; failure to
find one is not a proof of uniqueness.

**[derived] Representation sufficiency.** If c(x)=c(x') but a required query q
has different answers for x and x', no deterministic reader of (c,q) can answer
both correctly. This proves a restricted expressivity failure. Near collisions
require noise/precision/reader-regularity assumptions. Low covariance rank alone
does not prove this failure: even a scalar real code can index a finite set for
a sufficiently expressive decoder. For a continuous scalar bottleneck on an
open two-dimensional state domain, stronger topological obstructions can apply;
those assumptions do not follow from our finite counting fixture.

**[derived] Structural proofs.** A shared scalar score with comparison defined by
s(a)>s(b) guarantees transitivity of that relation. It cannot express a strict
three-cycle a>b, b>c, c>a. The old linear-order head has this limitation; a generic
pairwise MLP need not. Such proofs establish a specific algebraic property or
incompatibility, not correctness of learned sensory grounding.

## Literature record

Search date 2026-11-01. Queries included Bartlett/Mendelson Rademacher risk bounds,
Baxter inductive bias learning, Dziugaite/Roy nonvacuous PAC-Bayes bounds, Ben-David
different-domain theory, and Zhang random-label generalization. Evidence below
is **[abstract]** (primary landing pages/abstracts/search summaries), except the
explicitly derived arguments above; no full-paper theorem audit was performed.

- [Bartlett and Mendelson (2002)](https://www.jmlr.org/papers/v3/bartlett02a.html):
  Rademacher/Gaussian complexity risk bounds. Different from our evolving
  representation and dependent curriculum. Implication: specify the full loss
  class and independent sampling unit before attempting a numerical bound.
- [Dziugaite and Roy (2017)](https://proceedings.mlr.press/r15/dziugaite17a.html):
  computed nonvacuous bounds for large stochastic neural networks via PAC-Bayes
  optimization. Different tasks and model distribution. Implication: certificates
  are possible in some settings; our weights are not certified by that result.
- [Baxter (2000)](https://arxiv.org/abs/1106.0245): learning a shared inductive bias
  across an environment of related tasks can improve generalization. The 2011
  arXiv posting is of the earlier work. Different assumptions from a single
  counting fixture. Implication **[analogy]**: whole-task holdouts are necessary
  evidence for our proposed general-purpose bias.
- [Ben-David et al. (2010)](https://escholarship.org/uc/item/2nv1j9sc): domain
  adaptation bounds relate source error, distribution divergence and joint
  source/target achievable error. Different formal setting from task-family
  transfer. Implication **[analogy]**: similar latent distributions alone need not
  imply compatible labels or transferable relations.
- [Zhang et al. (2017)](https://arxiv.org/abs/1611.03530): studied deep networks
  fitting random labels. Different networks/data, not a capacity measurement of
  ours. Implication **[analogy]**: parameter count and training fit alone are weak
  explanations of the selected solution's transfer.

## Before any experiment

No numerical computation was performed. A future contract must define the task
environment, endpoints, controls, selection budget and failure criteria, with a
full prior-work review and [computation reporting](computation-reporting.md).
For F families, T instances, S seeds, C code choices and R reader choices, base
fit count is F*T*S*C*R before probes, tuning and rejected candidates. First-layer
cost is O(hd) dense or O(hk) genuinely sparse per example; dense hidden layers
still contribute O(h^2), and masked dense kernels may save no computation.
Count all encoders, cached codes, optimizer/memory state, independent episodes,
exposures and selection fits. Record phase timings and peak storage separately
from deterministic scientific results. Prior measured costs are context at their
reported scope; missing historical phases must be marked unavailable.
