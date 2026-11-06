# Core generality, context and sensory features

2026-11-01. Consolidated record of the research discussion. **Exploration, not
a selected architecture, frozen contract, or new experimental result.** The
user is exploring what is possible; the proposed studies below are not an
instruction to launch runs. README and frozen historical artifacts are unchanged.

## Working position

**Priority clarification:** [CPU efficiency](cpu-efficiency-design-constraint.md)
is a fundamental architectural constraint alongside capability. Context,
perception and memory cannot be added without accounting for their execution
and learning costs. No CPU-efficiency advantage has been established.

The learned real-vector code space plus sparse readers is a plausible general
computational component, with no demonstrated restriction to counting or a
single ordering. This is a representability claim with capacity assumptions,
not evidence that the current training procedure learns grammar or geometry.

Access to sufficient context and working state is a major known limitation of
the current two-item predictor. Attention and memory are possible remedies;
neither particular mechanism is mandatory. The user also mentioned "atlas" as
a possible memory mechanism; no specific implementation or capability was
defined or evaluated in this discussion. Supplying context does not by itself
prove that the core can learn to select and use it.

Sensory feature extraction is another major component. Treat existing feature
extractors as candidates to reuse and evaluate, rather than assume they preserve
everything a relational task needs. The research separates:

1. **Perception:** retain the sensory distinctions relevant to the question.
2. **Context and memory:** expose relevant history and current working state.
3. **Code and readers:** combine available information into predictions/actions.

These are functional obligations, not a requirement for three separate modules.
Known information/capacity limits can be derived; experiments must establish
learnability, robustness, efficiency and transfer. Memory and attention are not
claimed to resolve every limitation.

## Evidence and theory already recorded

| Topic | Current position | Detailed record |
| --- | --- | --- |
| Code versus reader contribution | Sparse disruption changes cross-sense code accessibility; no causal percentage split. Static sparsity alone does not reproduce the gain. | [Code-space analysis](code-space-reader-analysis.md) |
| Counting specialization | The vector space is not explicitly numerical; earlier heads and the task/selection process supply counting-compatible biases. Different scripts are not independent task families. | [Specialization and generalization](counting-specialization-generalization.md) |
| Exact reader properties | First layer sparse, second layer dense; cyclic relations are representable; blind directions and finite piecewise-affine complexity can be derived. | [Hypothesis properties](sparse-reader-hypothesis-properties.md) |
| Grammar | Missing prefix information is a hard obstruction. Contextual inputs and adequate state make grammar processing plausible, not demonstrated. | [Grammar notes](grammar-learning-extension-notes.md) |

The current fixed-width network is not an arbitrary-accuracy universal
approximator. The architectural family is more general: **[derived, using the
standard universal-approximation result as familiar background]** on a compact
domain, suitable masks allow 32 first-layer units to pass all coordinates of
two 16-dimensional codes, shifted into the positive ReLU region. Each unit can
set the weight from its other operand to zero. A subsequently widened dense
ReLU hidden layer can approximate continuous functions of those coordinates.
This fits the first-layer budget of sparse48, but requires suitable masks and
potentially more than its present 64 second-layer units. The argument is for
a fixed relation; it is not a proof for every fitted mask or an optimizer.

The distinction throughout is **representable versus reliably learned**.
Dense containment does not imply equal finite-data learning behavior. A grammar
or geometric task may be expressible while the training procedure learns a
shortcut. Conversely, successful attention/context encoders may perform much
of the reasoning themselves; controls are needed before crediting the core.

## What the core cannot do without changing the conditions

These limitations are **[derived]**, with proofs and qualifications in the linked
notes:

- Recover a distinction when two required cases have identical reader inputs.
- Use information annihilated by a fixed reader's first-layer mapping.
- Realize arbitrary functional complexity with its present fixed width/depth.
- Compute every continuous numerical operation exactly: its raw output is
  piecewise affine in the code inputs, so exact multiplication over an open
  continuous domain is excluded for that reader, though bounded approximation
  and finite-set solutions are possible.
- Track unbounded exact structure with a fixed finite number of reliable state
  configurations. Growing memory and repeated computation alter that premise.
- Guarantee a correct unseen rule when the available evidence admits conflicting
  explanations and no further assumption distinguishes them.

These are not claims that a whole system with additional encoders, memory and
iteration has exactly the same limits as a single reader call.

## Geometric shape relations

**[derived capability assessment, not tested]** Suitable codes and sufficient
reader capacity can support position comparisons, size comparisons, orientation
relations and approximate geometric compatibility. Touching, overlap,
containment, symmetry and congruence require representations carrying the
relevant shape detail and, where appropriate, part correspondences. This is
not a claim that all these relations fit the present width for arbitrary shapes.

| Relation | Necessary information to make the judgment possible |
| --- | --- |
| Left/right or above/below | Position in compatible reference frames |
| Larger/smaller | The relevant measure of extent or size |
| Parallel/perpendicular | Orientation or directional features |
| Touching/overlapping/containing | Position and sufficient boundary information |
| Same shape after rotation/scaling | Shape structure and treatment of transformations |
| Symmetry/congruence | Enough geometry to establish appropriate correspondences |

The first-layer fan-in is per unit, not a limit that the entire reader can use
only one geometric property. Its dense second layer combines features.

Invariance is not automatic. Joint translation preserves overlap, and joint
rotation preserves relative angles, but image-frame "left of" changes under
rotation. Specify which transformations should preserve the answer rather
than enforcing all invariances indiscriminately. Exact boundary contact or
equality needs explicit numerical/perceptual assumptions; practical approximate
tasks should declare tolerances. A fixed finite-precision code cannot preserve
unlimited arbitrary boundary detail for every possible query.

Geometry is useful conceptually because many relations can be tested with both
objects present at once, without long-term memory. Independently varying shape,
position and orientation would challenge the core to retain multiple factors,
rather than one common count-like coordinate.

## Sensory feature extraction

**[derived task-sufficiency argument]** Recognizing an object category does not
imply preserving the information needed for a relation. Two encodings can both
identify the correct object but differ in whether they retain exact location,
orientation, boundary shape, order or part structure. Information discarded
upstream cannot be recovered by the reader merely by increasing its capacity.

Invariance can help recognition while hurting other questions. Removing
orientation may help recognize a rotated object, but prevents reporting its
orientation unless that information is retained elsewhere. A useful design
may retain both invariant identity and transformation-dependent attributes;
this is a candidate organization, not a mandatory named-slot representation.

The user's suggestion that feature extraction might be solved is recorded as
an exploration question. No survey in this discussion establishes that a
particular off-the-shelf sensory model is sufficient. The practical position is
to reuse strong candidates where appropriate and assess their information
content against the intended relation family.

## Proposed ways to distinguish causes, not scheduled experiments

- **Reader versus representation:** freeze existing codes and fit fresh dense
  and sparse readers; test orthogonal changes of basis without losing information.
- **Task specificity:** evaluate cyclic, multi-factor and branching relations,
  holding complete tasks out of architecture selection.
- **Context:** add a contextual state and compare dense/static-sparse/disrupted
  variants on controlled formal-language tasks before language-scale claims.
- **Perception versus core:** start with explicit geometric attributes, then
  frozen sensory features, then joint perception/core training. Match interfaces
  and report any privileged ground-truth attributes as extra information. This
  sequence helps locate failures, but differing supervision/capacity must not be
  mistaken for a pure causal ablation.
- **Use rather than recognition alone:** assess interpretation, composition or
  execution, alongside relational classification and prediction.

All future studies need their own prior-work review, declared controls and
selection rules under [literature review](literature-review.md), and theoretical
plus measured cost reporting under [computation reporting](computation-reporting.md).
No new empirical measurements, benchmark claims or timing results are added here.
External literature and its reading scope are recorded in the linked topic
notes and [prior work](prior-work.md); the geometry/perception discussion above
is an argument about required information, not a new literature finding.
