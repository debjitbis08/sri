# Fan-in transfer and limited memory: analytical review

**Scope update:** the proposal was replaced in revision `217701a` by decaying
dense reader units. External-store criticisms below describe the superseded
draft. See [the revised analysis](decaying-readers-theory.md) for the current
proposal, including a correction: fixed75's 16 residual units were randomly
sparse, not dense.

2026-10-30. Analysis, not a new frozen contract. Latest completed artifacts available
at review: `fixed`, seeds 230–239. The timing arms are implemented, but no timing
result/comparison artifact is present. The limited-memory study is a draft.

Evidence below distinguishes observed results, deductions from code or stated
assumptions, and hypotheses. No new network training was run. Historical results
and the planned proposal are not rewritten by this review.

## What the latest result changes

See [fixed results](fan-in-pilot-v1-fixed-results.md), following
[RigL at one dimension](fan-in-pilot-v1-rigl1-results.md).

| Measure | Dense | Random sparse | RigL | Fixed | Fixed 75% |
| --- | --- | --- | --- | --- | --- |
| taps → withheld Chinese | .13 | .12 | .66 | .76 | .31 |
| English comparison | .49 | .56 | .75 | .52 | .51 |
| Chinese comparison | .57 | .64 | .86 | .99 | .77 |
| Devanagari comparison | .64 | .68 | .81 | .66 | .67 |
| Chinese paired, own block | .64 | .62 | .38 | .56 | .61 |
| Collection variance on first PC | .88 | .91 | .99 | .81 | .83 |

**Observed:** RigL's main transfer effects and taught-fit costs replicate. A fixed
scalar input to the readers is sufficient for strong Chinese transfer, but not for
the broader RigL outcome. Fixed75 does not approximate RigL despite its superficially
similar concentration. This is a result about learning with a constraint, not proof
that the constrained model represents functions unavailable to the dense model.

Two interpretations in the results need narrower wording:

- The predeclared label "learned choice matters" passed, but the comparison changes
  the whole wiring trajectory, coordinate, residual wiring, optimizer resets and
  dead units. It does not identify coordinate choice as the cause.
- Dead units are **not necessary for Chinese transfer**: fixed has none. They remain
  a possible contributor to RigL's English/Devanagari advantage. The existing
  comparison does not exclude that contribution.

## 1. The constraint is on what readers can distinguish

**Derived from `src/generic_net.py` and the masks.** In the fully fixed arm,

    q(x) = code(x)[d]
    g(a,b,r) = G(q(a), q(b), r)
    P(a,b) = F(q(a), q(b)) ∈ R^16.

Therefore the direct derivative of g with respect to either operand's code is
parallel to coordinate d. P's input derivatives also have support only there.
Items with equal q cannot be distinguished by these readers, holding the other
operand and relation fixed. End-to-end training must put useful distinctions in q.

This does **not** prove that q is count, monotonic in count, or information-poor in
bits. Nine arbitrary identities fit in nine scalar values, and nonlinear readers
can learn a lookup over those values. A scalar bottleneck is an inductive bias,
not a theorem of abstraction. The cross-sense and ordering tests supply the
empirical evidence for quantity structure.

The encoder still emits 16 coordinates. P emits a full 16-dimensional prediction,
scored by squared distance against full collection codes. Consequently off-axis
coordinates still receive target-side gradients. Reader concentration does not
force the entire code onto the coordinate axis.

This resolves an apparent numerical puzzle: RigL has 99% of collection variance on
one principal direction, but only 48% on its reader coordinate. A near-rank-one
code can take the form c(n) ≈ v q(n)+b, where v has many nonzero components.
Reading one component can recover that scalar even though the line is oblique.
For exact rank one, axis variance fraction equals squared cosine with the PC;
.69² ≈ .48 is consistent with the reported means, though averages do not obey
this identity exactly. Fixed's 81% first-PC variance shows that fixed reader
inputs do not by themselves enforce rank-one output geometry.

## 2. Why the k=1 RigL dynamics are unusually strong

**Derived from `experiments/fan_in_pilot.py:rigl_step`.** The number dropped per
unit per operand is `max(1, round(alpha*k))`. At k=1 it is always one, including
when the cosine alpha approaches zero. Every 100 updates through update 4,500,
each operand connection is reconsidered. The old connection is eligible for
immediate regrowth; a replacement starts at zero and has its Adam moments reset.

Thus this regime is repeated gradient-based coordinate selection. Magnitude
ranking cannot discriminate between the one existing edge and any other active
edge in that operand: there is only one. The cosine schedule does not gradually
reduce the fraction reconsidered at k=1. Concentration may emerge over time, but
that trajectory has not been measured.

For a first-layer weight, its dense gradient on a batch is

    grad(W_ij) = Σ_b δ_bi x_bj,

where δ is the backpropagated preactivation error. Selection finds coordinates
correlated with current errors, not necessarily the coordinates carrying count
most faithfully. It is also sensitive to coordinate scale. On a single example,
all nonzero-error units rank coordinates by |x_j|; on a batch the error-weighted
correlations differ between units. This supplies a possible explanation for
concentration without assuming that each unit independently discovers count.

**Hypothesis:** useful quantity signal, activation scale and shared loss gradients
favor a coordinate; more readers select it; their subsequent gradients strengthen
its usefulness to the encoders. Prediction and comparison share the code and can
reinforce the same coordinate. RigL's g and P choosing the same dimension in all
ten seeds is consistent with this feedback, but final masks cannot establish its
direction or distinguish useful-signal selection from scale-driven selection.

**Implication for the timing run:** its gradual schedule is an imposed continuation
schedule, not a reconstruction of measured RigL concentration. In particular,
integrating cosine alpha is not integrating k=1's actual replacement budget.
A negative timing result would reject that schedule's sufficiency, not timing in
general. Even a successful arm would show sufficiency rather than prove RigL
followed the same mechanism.

## 3. Why 75% concentration can fail

**Derived:** mask share is not functional influence. With 48 constrained units
and 16 unconstrained units, the next layer can suppress the former and amplify
the latter. No stated bound prevents the free path from dominating. Random
residual connections can preserve script-specific solutions.

**Hypothesis:** fixed75 leaves a cheap route to taught accuracy, so the encoder
need not reorganize representations into a common quantity variable. Its better
retention and taught fit, alongside weaker transfer, fit that explanation.
RigL's remaining connections are selected by current gradients, not a permanent
random bypass, and many of its P units become inactive.

A decisive inexpensive diagnostic on saved checkpoints is to ablate modal-axis
and nonmodal units separately and measure score changes. Also measure gradients
and activations: connection counts alone do not establish reliance. This would
be a new diagnostic, not a result obtained in this review.

## 4. Alignment, ordering and exact matching are different

Consider a scalar model for sense/script s:

    q_s(n) = a_s n + b_s + ε_s(n).

**Derived in this illustrative model:** positive a_s can preserve within-script
ordering even with a large offset b_s. Cross-sense matching also needs compatible
scales and offsets; exact nearest-count matching is sensitive to noise relative
to adjacent-count spacing. For example q(n)=n+20 has perfect order but only 1/9
nearest-count accuracy against candidates 1,…,9. The numerical check records
this counterexample; it is not a fit to the learned comparator.

This makes excellent comparison with mediocre exact matching unsurprising.
It does not make the observed taught-fit cost inevitable: nine exact quantities
and their same/next/more relations can all be represented on one scalar. The
loss of matching accuracy may therefore be an optimization, calibration or
interference cost of this training procedure, rather than insufficient scalar
capacity for the intended task.
Moreover, the generic g is neither constrained to be symmetric nor to implement
transitive equality. Strong taps↔forms matching need not imply equally strong
form↔collection matching. Measures at a script's own block and at the final block
also differ in training time and should not be treated as simultaneous tests.

The reported off-axis statistic measures only a **mean displacement**, normalized
by collection variation along the axis. Small displacement does not establish
count-by-count alignment: errors can cancel, and an increase in its denominator
can reduce the statistic. Fit slope, offset, count-conditioned residuals and
within-count variation separately before crediting semantic alignment to it.

**Unresolved:** why fixed is so good specifically on Chinese. Curriculum position,
visual features and optimization are confounded. The first three Chinese numerals
being stroke-like is a candidate cue, not an explanation of .99 ordering over all
nine counts. Reverse script order, permute glyph-to-count mappings consistently,
or inspect per-count trajectories to distinguish these explanations. A given
coordinate index has no intrinsic meaning across counterfactual training runs;
using RigL's final index in another arm is an oracle index, not transplantation
of its learned quantity feature.

## 5. What a decaying memory actually changes

Review target: [limited-memory proposal](limited-memory-proposal.md).

**Derived:** the proposed retrieval answer has no training loss, does not gate
network training, and does not modify the code. Given the same replay batches,
enabling or disabling retrieval leaves network learning unchanged. Retrieval can
improve combined answers but cannot itself cause consolidation in this design.
The causal intervention on learning is the distribution of replay exposures.

Writing the replay distribution as Q_t and new experience as D_t gives an
idealized batch objective

    L_t = .75 E_Dt[ell] + .25 E_Qt[ell] + context losses.

The actual implementation averages several item-level losses separately, so
these are episode proportions, not exact coefficients on all gradient terms.
Recency changes which earlier losses remain represented. It supplies no explicit
penalty for retaining arbitrary details in network weights and no loss for
future forgetting. Shared structure can survive if current experience continues
to train it; removing old evidence can equally destroy useful structure.

There are two separate retention obligations:

    input glyph → quantity coordinate → reusable relations.

Context-only training can maintain the latter while the former drifts. Since
collections and forms share the image encoder, context training can actively
change old form codes even after forms disappear. At first order,

    Δq(old form) ≈ -η ∇θ q(old form) · ∇θ L_context.

Neither the sign nor size is controlled by the existence of a count axis. A
stable quantity concept cannot reconstruct a forgotten arbitrary glyph mapping.
The proposed prediction that an axis protects English therefore remains a
hypothesis; the fan-in results' worse English matching are countervailing evidence.

The 2,000-update delay is active training on a changed distribution, not passive
decay or a demonstration of grokking. A no-update checkpoint is the zero-cost
reference for weight retention. Improvement during context training would need
to be distinguished from preserving already learned structure.

## 6. Exact consequences of the proposed store

Assume one **fresh** episode inserted each update, no reinsertion of replayed
episodes, no strength refresh, identical initial strength, and a full store.
Then strength is strictly ordered by age: dropping the weakest is exactly FIFO.
K determines the retention horizon; H changes sampling but not eviction.

For ages a=0,…,K−1, let r=2^(−1/H). Then

    p(a) = (1-r) r^a / (1-r^K).

At s updates after a script ends, provided all entries at the boundary came from
that script, its replay share for 0≤s≤K is

    M(s) = (r^s-r^K)/(1-r^K).

For K=1500, H=750 and four replay draws/update:

| Updates after boundary | Old-script share | Old episodes/update |
| --- | --- | --- |
| 0 | 1.000 | 4.000 |
| 500 | about .507 | about 2.026 |
| 1000 | about .196 | about .783 |
| 1500 | 0 | 0 |

Thus English receives no replay throughout Devanagari under these assumptions.
The persistent reference instead gives English two of the four replay slots on
average during Devanagari. Equal total replay does not equal matched old-task
exposure. At stationarity, the expected lifetime replay count of each inserted
episode is four (sum four times its normalized age weights). Changing H
redistributes those opportunities in time; it does not increase that mean.

## 7. Resolve these design issues before freezing

1. **No-refresh versus writing from the batch.** A quarter of the proposed mixed
   batch is replay. Randomly writing from that batch at strength one can reinsert
   old episodes, implicitly refreshing them. Write only genuinely new episodes,
   or retain their original timestamp and explicitly define deduplication.
   Otherwise the 1,500-update disappearance claim and the equations above fail.
2. **Withheld-label leakage.** Existing episode tuples contain the true count even
   at withheld steps (`e=-1` suppresses their collection). A retrieval index must
   include count answers only for directly observed form–collection pairs.
   Indexing every stored form with its tuple's count would directly label the
   withheld forms. Raw proximity thresholds cannot fix that leak.
3. **Delay protocol.** "Context only, no forms" conflicts with the prediction that
   the persistent buffer keeps rehearsing English during the delay. Specify
   whether this means no *new* forms, with replay continuing in all replay arms,
   or absolutely no forms, with replay disabled in all arms. In the latter case
   retaining the buffer changes no delay gradients. With no fresh writes,
   exponential fading alone never empties a finite store: normalized sampling
   probabilities are unchanged when all strengths shrink equally. Continued
   context writes or an explicit expiry threshold are needed for eviction.
4. **Separate recency from capacity.** The cheapest additional control is the
   same online FIFO store with uniform replay. It distinguishes age weighting
   from the storage horizon. A reservoir store at equal capacity tests historical
   coverage. The historical buffer also differs in current-script inclusion and
   script balancing. Six arms as drafted measure a package, not fading alone.
5. **Interpret the interaction cautiously.** A smaller absolute retention loss
   under fan-in may reflect its lower starting accuracy or a floor. Report each
   arm's acquisition and subsequent change, alongside absolute final accuracy,
   probe ordering, calibration and taught fit. A positive interaction alone is
   insufficient to establish protection by structure.
6. **Stage 2 is not yet a pure-memory control.** An unused script mapped by a fixed
   random permutation to counts is still a dictionary into the same quantity
   concept. Once its mappings are learned, all number relations still transfer.
   Independent arbitrary associations across multiple relations, with matched
   exposure and entropy, would better test absence of reusable shared structure.
7. **Threshold and compute accounting.** Calibrating tau on same-form neighbours
   estimates acceptance of true matches, not false-match rejection. Measure
   different-count false acceptance and coverage too, on permitted training
   data. Calibration uses labelled information and computation even with a fixed
   rule; record it. Each stored episode has multiple form steps, so nearest-neighbour
   cost is O(K L p), not O(K p), unless only one form is indexed per episode;
   L is maximum steps and p=784. Index bytes do not include the backing rendered
   data that keeps raw episodes retrievable. Both must be accounted for.

Keep the RigL schedule explicitly tied to the original 6,000-update learning
phase if that is intended; extending total steps to 8,000 must not silently
change when rewiring stops. Define the eight transfer measures at the new
evaluation times rather than reusing "own block" names for post-delay values.

## Practical reading and next discriminators

**Current best hypothesis:** restricting readers encourages a shared quantity
interface, while adaptive connectivity changes how encoders reach that interface.
The strongest evidence is replicated transfer with code-probe improvement;
the unproven part is why RigL's trajectory aligns more scripts than fixed wiring.

First read the timing results when available, with the k=1 scheduling caveat.
Before another full run, log concentration versus time, coordinate RMS and
gradient correlations; inspect functional reliance on residual units using
saved checkpoints. For memory, resolve write provenance, permissible retrieval
labels and the delay protocol, then use a matched uniform-FIFO control. These
checks are cheaper and more identifiable than interpreting six arms as proof
that fading creates concepts.

Prediction, not a result: persistent replay should usually protect arbitrary
form mappings best; decaying replay may preserve quantity operations while
losing their older input mappings. An actual transfer gain is possible if old
replay interferes with shared structure, but that is what must be measured.

## Prior work and reading scope

Sources opened 2026-10-30, by direct URL; no novelty search.

- [Evci et al., RigL](https://proceedings.mlr.press/v119/evci20a.html)
  **[abstract]**: sparse training with magnitude/gradient topology updates.
  Its published accuracy/compute findings do not establish cross-sense transfer
  or the special per-unit k=1 behavior derived above.
- [Kumaran, Hassabis and McClelland 2016](https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf)
  **[read: abstract, introduction and schema-consistency discussion, manuscript
  pp. 11–13]**: complementary systems support experience learning and structured
  knowledge; replay can reweight experience; integration depends on compatibility
  with existing structure. This motivates separating acquisition, replay and
  integration. It does not predict that recency eviction improves this network.
- [Chaudhry et al. 2019](https://arxiv.org/abs/1902.10486)
  **[abstract]**: joint training on current examples and tiny episodic memories
  can be effective. This supports replay as a strong baseline, not the inference
  that a recency-only store preserves arbitrarily old tasks.
- [Khandelwal et al. 2020](https://arxiv.org/abs/1911.00172)
  **[abstract]**: nearest-neighbour retrieval augments a language model without
  additional training. Here keys are pixels and retrieval does not influence
  optimization; its usefulness must be separated from representation learning.
- [Power et al. 2022](https://arxiv.org/abs/2201.02177)
  **[abstract]**: delayed generalization during continued algorithmic-task training.
  The proposed delay removes form experience and changes the training objective;
  grokking is an analogy, not a prediction for that intervention.

## Reproduction and computation

Run `.venv/bin/python -m experiments.fan_in_memory_analysis`. It aggregates the
frozen fixed comparison, checks the k=1 drop counts, verifies the FIFO formula
against direct geometric sums and checks the scalar-ordering counterexample.
It does not validate the hypothesized neural learning dynamics.

Deterministic output: `reports/fan-in-memory-analysis.json`. Separate whole-script
timing, CPU, RSS, source hashes and environment:
`reports/fan-in-memory-analysis-runtime.json`. No fits, training, random trials
or selection/tuning. Let A be parsed artifact size, M the number of measures,
C arms, S seeds: time O(A+MCS+K), storage O(A+MCS+K) for this implementation
(fixed four geometric-sum checks). Provenance hashing adds linear file-byte
work. This is an analytical receipt, not an inference benchmark.

For a future store implementation, episode indices cost O(KL), indexed pixels
O(KLp) if materialized, naive sampling O(K) per update, and brute-force retrieval
O(QLKp) for Q queries. Backing data, model weights, optimizer state, calibration
and rejected runs must be included separately. Existing pilot runtime receipts
do not measure the proposed memory overhead; no speed claim is made here.
