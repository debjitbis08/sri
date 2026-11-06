# Code space versus sparse readers

2026-11-01. Research discussion; no new experiment, frozen contract or measurements.

## What the evidence separates

The [erasure pilot](fan-in-pilot-v1-erasure-results.md) supports disruption of
sparse readers under the tested architecture, losses and schedules. It does not
establish that sparsity is necessary for learning patterns in general. Static
sparse readers do not reproduce the disrupted readers' transfer. Dense resets
match the per-connection rate, not the loss of an entire operand input per unit.
A whole-operand reset in dense units remains an important competing control.

There is already evidence that the learned representation changes independently
of the original reader: a collection-trained affine count probe on each code
coordinate transfers to Chinese forms with mean order accuracy .59 for dense,
.61 for static random sparse readers, and .90 for random rewire. These are
post hoc diagnostics, not a causal percentage decomposition. See the
[coordinate analysis](count-coordinate-analysis.md). Collection count is spread
over most coordinates even in dense arms; cross-sense agreement is the difference.

The code family is currently a shared learned 16-dimensional real vector. Its
learned organization and the reader are coupled by training. The working
hypothesis is reader restriction/disruption -> encoder pressure -> more broadly
accessible cross-sense structure. The losses supply the relation information.
Reader and code effects can interact, so there need not be one additive share
attributable to each.

## Cheapest discriminating comparisons

1. Freeze encoders and projections from dense, static sparse and disrupted sparse
   training. Fit fresh dense and sparse readers on each, with paired data and
   reader seeds. Preserve the original withholding rules; report transfer and
   learning curves, taught fit and retention separately. Do not transplant old
   reader weights between independently trained latent bases. A dense reader
   succeeding on disrupted-training codes would show sparsity need not persist
   at deployment. It would not show dense training can discover those codes.
2. Repeat on common orthogonal transforms of the frozen codes, applied to all
   senses. Include random rotations and a collection-training-only PCA basis;
   no whitening. Orthogonal transforms preserve distances, inner products and
   information but change coordinate accessibility. A dense first layer can
   absorb the inverse rotation exactly; a sparse mask generally cannot. Rotate
   predictor outputs and candidate targets consistently when testing P. An
   untouched reader after rotation is not a valid accessibility comparison.
3. If needed, train a factorial comparison of code families with dense/static
   sparse/disrupted sparse readers. Include whole-operand dense resets and a
   temporary input masking baseline to distinguish sparsity from disruption
   severity. These are distinct interventions, not equivalent algorithms.

For frozen-code accuracy Y(C,R), report representation contrasts at fixed R,
reader contrasts at fixed C, and their difference-of-differences interaction.
These describe accessibility under the specified refitting procedure; they do
not identify a natural causal mediation percentage for joint training.

## Alternative code families worth distinguishing

| Family | Question and limitation |
| --- | --- |
| Orthogonally rotated existing codes | Same information and geometry, different accessibility to coordinate-sparse readers; first diagnostic. |
| Learned low-rank bottleneck, expanded back to 16 coordinates | Does a dimensionality constraint reproduce the pressure? Include rank 1 and a modest higher rank; a scalar can discard non-count structure. Learn it from the same losses, without supplying count labels. |
| Unit-normalized continuous codes | Does direction-based geometry suffice? Removes norm information; it changes both capacity and distance geometry, so is not merely a coordinate change. |
| Sparse nonnegative or top-k activity codes | Tests sparse representations rather than sparse connections. A reader can miss active coordinates; overlap and support coverage matter. |
| Discrete learned prototypes / vector quantization | Tests categorical reuse. Distinct prototypes alone impose neither order nor successor; quantization also changes optimization. |
| Fixed nonlinear random features of a learned code | Tests whether a broader feature basis makes simple readout sufficient; extra dimensions, feature construction and potential periodic aliasing must be counted. |

Structured product codes can follow if the task contains multiple independently
varying factors. Scalar, ordinal or thermometer codes assigned using the true
count would supply the sought structure; use them only as explicitly privileged
diagnostic controls, not as evidence of learning that structure.

## Prior work and evidence levels

Search date 2026-11-01. Queries: "site.jmlr.org dropout prevent neural networks
overfitting 2014 co adaptation", "site.arxiv.org disentanglement impossible
without inductive biases Locatello 2019", and "site.papers.nips.cc random
features large scale kernel machines Rahimi Recht 2007". Primary abstract pages
were opened; full papers were not reviewed for this note.

- **[abstract]** [Srivastava et al. (2014)](https://www.jmlr.org/beta/papers/v15/srivastava14a.html)
  report dropout's reduction of co-adaptation and supervised generalization
  improvements. **Differs:** temporary masking is not persistent weight/moment
  erasure or rewiring. **[analogy]** Input masking is a cheaper competing control
  for the project's reader-dependence hypothesis, not an explanation established
  by that paper.
- **[abstract]** [Locatello et al. (2019)](https://proceedings.mlr.press/v97/locatello19a.html)
  establish limits on unsupervised disentanglement without inductive biases and
  report no general downstream sample-efficiency gain from disentanglement.
  **Differs:** our relational supervision and multimodal sequences provide
  additional biases. **[analogy]** Naming a coordinate space does not establish
  that its axes will acquire the desired meanings.
- **[abstract]** [Rahimi and Recht (2007)](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
  approximate shift-invariant kernels using randomized features and linear
  learning. **Differs:** no joint multimodal representation training or sparse
  reader disruption. **[analogy]** Fixed feature expansion is an established
  baseline for changing readout expressivity, not evidence it will align senses.
- **[derived]** For orthogonal Q, ||Qa-Qb||^2=||a-b||^2 and a dense readout
  Wc equals (WQ^T)(Qc). A general WQ^T need not retain W's sparsity. This proves
  a distinction between information and coordinate accessibility, not an
  optimization outcome. A fixed rotation during joint training can be absorbed
  by an unconstrained learned projection; frozen-code rotation is more diagnostic.

## Computation requirements for a future contract

No numerical analysis was run for this note. Before execution, specify seeds,
data splits, candidate transforms, reader widths, supports, fit budgets and
selection rules under [computation reporting](computation-reporting.md).
For N cached d-dimensional codes, storage is O(Nd), a dense rotation costs
O(Nd^2) with O(d^2) transform storage, and covariance plus full PCA costs
O(Nd^2+d^3). A width-h first layer costs O(hd) per dense example versus O(hk)
for k inputs per unit with an actual sparse implementation; subsequent dense
layers still cost O(h^2). Dense masked implementations need not realize sparse
operation savings. Multiply by exposures, fits, seeds and all tried transforms;
count encoder extraction, selection, probes, optimizer state and cache storage.
Report equal-example quality and quality versus total measured cost separately,
with deterministic results and nondeterministic runtime receipts in separate
files. Historical costs may be reused only at their documented scope; missing
phases must be marked unavailable.
