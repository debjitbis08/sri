# Equality judgment versus code convergence

Discussion 2026-11-05. Analysis of the frozen [v2 results](equality-relations-v2.md)
and their implementation; no new experiment contract or training run.

## What should converge?

**[derived]** Same count is an equivalence of numerical meaning, not identity of
the whole object. English “three”, a digit and three dots may retain different
form information while supporting the same numerical judgments. A sufficient
representation could be `(count information, form information)`, with numerical
operators ignoring the latter. Equal full vectors are sufficient for identical
outputs of a deterministic reader in a fixed context, but not necessary.
Approximate vector proximity alone does not guarantee identical decisions near
a reader's decision boundary.

**[derived: source inspection]** `src/equality_net_v2.py` trains balanced binary
cross-entropy on `S(G([c_a,c_b]), op)` (or concatenated object codes on the count
base). This penalizes wrong pair judgments, not distance between equal objects.
Gradients reach the encoder, but the pair network and reader can also absorb the
learning. There is no implication that successful equality classification makes
the full object codes coincide. Nor does fitting the observed pairs guarantee
equivalence properties on unseen pairs.

The useful target is correct equality and numerical substitutability on new
items, orders, scripts and after later learning. Code alignment is a diagnostic
and a possible mechanism. It is not the primary success criterion. Properties
should remain learned from examples under the program's existing constraint.

## What v2 establishes and leaves open

- **[read: local results]** On the count base Chinese nearest identity is
  0.64–0.71 while equality identity is 0.33–0.37. Available identity information
  is not being used effectively by this equality reader. Representation and
  reader failures must be separated before prescribing stronger alignment.
- **[read: local results]** Chinese label replay reaches 0.83 identity on the
  two-space base versus 0.79 for relational replay. Retention is an additional
  problem; teaching the current script equality does not establish protection
  of an absent old script.
- **[derived: design inspection]** The control already teaches forms numerical
  differences to collections in both orders. It is not devoid of same-count
  information. Similar final scores show no demonstrated incremental benefit
  from replacing label replay with cross-script operator pairs, not that those
  pairs caused no learning. These arms also change the replay objective; they
  are not a pure add-equality ablation. Cross-script teaching combines `eq` and
  `more`, so it does not isolate equality's causal contribution.
- **[derived: measurement inspection]** `zh×dev` pair accuracy tests transfer,
  not the implication `eq(a,b) and eq(b,c) => eq(a,c)` on triples. Symmetry
  agreement should be separated by positive and negative pairs because most
  all-pairs examples have unequal counts.
- **[derived: measurement inspection]** “Judged equal” is the argmax collection
  item even if every equality logit is negative. Substitution is balanced
  agreement with that reference's predicted outputs. It supports behavioral
  consistency with a selected reference, not a proof of Leibniz substitution
  for accepted equalities, all operators or all contexts. Shared reader errors
  remain possible. Unknown Thai still receives a forced choice.
- **[derived]** Strict `more` must be false in both directions on equal counts,
  but full-vector coincidence is not necessary for that. A reader can ignore
  form differences or learn a finite region for ties. The current measurements
  do not distinguish tie-boundary errors from representation errors.

## Cheapest useful next diagnosis

**[proposal, untested]** Separate equality classification, candidate ranking,
numerical substitution and retention. Report positive/negative equality recall,
same-count versus adjacent-count errors, candidate score margins, and correct
identification after each block. Evaluate substitution for accepted equalities
and against true task answers, alongside the existing forced-choice diagnostic.
Split taught and withheld counts. Test actual triple violations separately.

**[proposal, untested]** On frozen representations compare the current equality
reader with nearest prototypes and the existing difference reader's zero class.
The latter is a diagnostic using already learned numerical information, not a
replacement definition of equality. A small retrained reader would separate
reader learning from encoder learning more directly, but costs an additional
fit and needs held-out evaluation. Compare base/normalization choices explicitly:
the current count versus two-ctx comparison changes both the pair pathway and
context normalization.

No weights were retained in v2, so diagnostics requiring logits, embeddings or
reader refits need a reproducible rerun with snapshots. Existing aggregate
reports cannot supply them. Freeze any such extension only after declaring its
complexity, rerun/selection costs and separate runtime reporting under
[computation reporting](computation-reporting.md).

## Prior work and alternatives

Search 2026-11-05: `site.proceedings.mlr.press supervised contrastive learning
Khosla 2020`; opened the primary abstract of
[Khosla et al., Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362).

- **[abstract]** Supervised contrastive learning explicitly attracts same-class
  embeddings and separates different classes; the paper reports image
  classification results. This differs from binary classification through an
  unrestricted pair reader, and from continual cross-script numerical learning.
- **[analogy / untested]** Such an objective is an established alignment control
  if geometric convergence itself is the hypothesis. It introduces geometric
  bias, so success would not demonstrate that an unconstrained equality operator
  independently discovered that geometry. Applying it to full object codes may
  suppress form distinctions needed elsewhere. It is not yet the recommended
  next full run; first locate the failure with the cheaper reader diagnostics.
