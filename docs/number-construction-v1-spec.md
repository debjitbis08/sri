# Number construction v1: counting episodes and confidence-controlled relational revision

Status: **draft, 2026-10-28 (revision 3: answer semantics and withholding decided). Nothing
implemented; not frozen.** Follows the [number systems v1
pilot](number-systems-v1-pilot-results.md) and the [meta v1
pilot](number-systems-meta-v1-pilot-results.md). Choices marked **Proposed** await the
researchers' decision.

## Where this comes from

The pilots point to two separate problems (the researchers' framing, 2026-10-28):

1. **The system does not group the writing systems.** Representations of the same number
   stay apart, and learning a new one disturbs the old ones through the shared
   satellite. Standard tools are expected to help (alignment, per-system adapters). That
   is a separate, measured component, not designed here. At least one condition runs
   without it, so it cannot hide problem 2.
2. **Nothing pressures the system to use the core relations.** The label is reached
   directly. Successor and order are side tasks, so the cheapest solution is a lookup,
   and unlabelled numbers snap to known labels.

Two proposals from the researchers, refined by the external review:

- **Teaching.** Children meet counts in order, in one language at a time, built up on
  real collections. Count-list recitation and understanding of cardinality are separable
  (Sarnecka & Carey 2008; [prior work](prior-work.md#number-core-network)), so order
  alone is not the target. Succession must be tied to something observable.
- **Confidence.** Familiar inputs get an immediate answer. When that answer is
  unreliable *and* relations can help, the system should revise it using the learned
  relations. Low confidence does not always mean reasoning will help: an unreadable
  symbol needs another look, while an unfamiliar symbol inside a known count sequence can
  be recovered through relations.

The parts below test distinct capabilities, each separately:
- grounded counting (A);
- whether existing relations carry usable information (B1);
- whether relying on relations improves them (B2);
- whether confidence makes revision economical (the gating comparisons in B1 and B2).

## Answer semantics (shared by all parts)

**Decided 2026-10-28:**
- **Primary answer: collection matching.** The question is "Which collection has the
  quantity represented by this form?"
  - Candidates are collection images of 1–9 objects in fresh random arrangements, in
    shuffled candidate positions.
  - The choice uses the judge's *same* score on core codes.
  - Visual shortcuts are controlled: candidates use the fixed-total-area rendering, so
    ink area does not track count, and arrangements come from evaluation pools never used
    in training.
  - This tests whether the form has acquired a quantity meaning, without making any
    language (Arabic digits included) the privileged output.
- **Secondary, reverse-direction test:** given a collection, select the lesson-language
  form, among that language's nine forms, by the same score. It is not primary, because
  it mixes quantity understanding with learning the output vocabulary.
- **Immediate and revised answers use the same candidates for each item,** so they are
  directly comparable.
- **B1 on existing checkpoints:** collection matching with dots candidates, since dots
  is a source in those runs. The old fixed-code label readout is reported alongside as a
  diagnostic of those models.

## Information available during revision (shared)

Three levels, always reported separately:

| Level | Context items | Their meanings |
| --- | --- | --- |
| **inferred (main)** | observed inputs (neighbouring forms in the sequence, or grounded collections) | inferred by the system from those inputs; never supplied |
| **oracle (upper bound)** | as above | correct core codes supplied; privileged, reported only as a bound |
| **none** | no context | revision has nothing to revise against; expected to do nothing |

"Grounded" means the context item is a collection, whose count the system must itself
recognize. Collections are not oracle labels.

## Withholding policy (shared)

**Decided 2026-10-28:**
- **What is withheld.** For every language after the first, **two nonadjacent interior
  counts** (from 2–8, each with directly paired counts on both sides) are **never
  directly paired** with a collection.
  - They are drawn by a recorded seed and balanced across languages.
  - Boundary counts (1 and 9) are reserved for a separate extrapolation test.
- **What training shows at a withheld step.** The form and the counting transition
  (add one / remove one) are shown, and the collection is hidden. Excluded in training
  for that form:
  - direct collection matching;
  - answer feedback;
  - any equivalent supervision, such as a *same* pair with a collection.
- **Varied sequences.** Starting points and lengths vary, with descending sequences and
  intermediate starts included, so a form's position in a fixed lesson cannot become its
  meaning.
- **Naming.** The sequences deliberately give these counts indirect grounding. They are
  called "never directly paired", not "unseen" or "ungrounded".
- **Evaluation of each withheld form, in two conditions:**
  1. **Isolated:** has teaching established a meaning that recognition retrieves? Strong
     isolated performance shows successful acquisition.
  2. **With grounded relational context** (inferred information level): can revision
     recover the meaning when immediate recognition fails? Gains over isolated show
     useful relational inference.

  Neither condition forces deliberation when the answer is already known.
- **B1 on existing checkpoints:** the withheld counts are the existing unlabelled
  numbers. Those runs did not follow this policy (no sequences), so B1 tests only the
  context condition's mechanism, not acquisition.

## Part A: counting episodes

**Episode.** A collection is built and changed step by step. At each step the learner
sees the collection and that step's form in the lesson's language:
1. start with one object; add one at a time, with varied objects and arrangements;
2. operations: ask for the quantity, remove one object, resume counting from an
   intermediate point;
3. later, a second language through the same quantities and operations.

The language is supplied as context at first. A later test checks whether the system
identifies the context itself.

**Persistent state.** An episode is temporal inference only if activity carries from one
step to the next. With persistent state, step t's core is solved with the coupling
μ‖c_t − S(c_{t−1})‖², with c_{t−1} the previous step's inferred state held fixed (see
Computation). For *remove one*, the coupling is μ‖S(c_t) − c_{t−1}‖². The episode loss
sums the steps.

**Arms (counting order, with language blocks kept fixed):**

| Arm | Within a language block | State between steps |
| --- | --- | --- |
| A-episode | episode order | persistent (coupling to the previous step) |
| A-order | episode order | none (ordinary per-item solves; order changes only the optimization sequence) |
| A-shuffled | the same items and operations, shuffled within the block | none |

The operations are supervised as relation pairs in A-order and A-shuffled, so all three
arms see the same grounded evidence.

**Language scheduling, tested separately:** A-episode with languages blocked against
interleaved.

**Forgetting:** grouping by language does not by itself preserve earlier languages.
**Proposed:** measure forgetting in the main arms, and add problem 1's per-language
adapters as one extra arm, not by default.

**Measures:**
- cardinality by collection matching, including withheld counts;
- succession on observed changes (add one / remove one);
- examples needed and retention when a new language is taught through the same
  quantities.

## Part B: relational revision with confidence control

The current solver cannot "think longer": it already finds the exact equilibrium, and the
successor plays no part in label inference. Revision is an explicitly defined operation.

**Revision operation (proposed main form): relational context as energy.** For item i
with context:

> E_i(state | input) + μ ‖c_i − S(c_prev)‖² + μ ‖c_next − S(c_i)‖²

where c_prev and c_next are the context items' core-inner states. **Proposed for v1:**
- the context states are computed by their own ordinary solves (inferred level) and then
  held **fixed**;
- the revised state of item i is one solve of the item's system with the added terms;
- the answer is read from the revised state.

Jointly inferring a whole sequence is a later variant (see Computation). An alternative
operation, anchor counting (the k minimizing ‖c_i − S^k(c_anchor)‖), is reported as a
secondary variant.

**B1: do the existing relations carry usable information?** The network is frozen: no
parameter changes except the controller. Checkpoints: number systems v1 C14 and I14, and
the meta pilot arms.
- The successor S was trained, so revision may correct snapped labels when context is
  available.
- I14 (private orders) and a **perturbed successor** (S replaced by an affine map of
  matched norm, or by S from another run), with perception held fixed, test whether
  *correct* relations drive any improvement.

**B2: does supervising revised answers improve relational competence?** The errors of
revised answers are back-propagated into S, the judge, the core translations and the
readout. **Proposed:** the satellites stay as in ordinary training. Compared with ordinary
training at equal total compute. B2 is the "pressure" experiment. B1 only says whether the
pressure has something to work with.

**Gate (inference).** It uses only signals available *before* revision:
- answer margin;
- SRI equilibrium energy (an untested confidence feature, see below);
- whether context is available;
- an estimate of input familiarity (for example distance to known representations).

Disagreement between the immediate and revised answers needs the revision, so it is
**not** a gate signal. It is the controller's training target and a diagnostic.

**Controller training.** On training data, compute both answers and label each item as
"revision helped / hurt / no change". The controller (**proposed:** logistic regression on
the gate signals) predicts the expected error reduction. Training data must include:
- confidently wrong immediate answers;
- items where revision damages a correct answer.

**Decision rule.** Revise when the predicted error reduction exceeds c × (revision cost).
c converts computation into loss units. **Proposed:** report the whole
accuracy-versus-computation curve over thresholds, plus one pre-registered c, fixed before
the confirmation run on development data. Controller and context-processing costs are
charged.

**Selection comparisons (B1 and B2), at matched average budgets where applicable:**

| Selector | Role |
| --- | --- |
| immediate only | no revision |
| always revise | is revision itself useful? |
| random revision at the gated arm's average rate | does gating choose better than chance? |
| confidence-gated | the proposal |
| oracle selector (revise exactly when it helps) | upper bound |

**Tasks with relational context:**
- withheld or unfamiliar forms inside known count sequences;
- ambiguous forms whose position relative to grounded items favours one reading;
- isolated unreadable inputs, where revision should not help and the gate should not
  choose it.

**Measures:**
- accuracy;
- solves and encodings per answer;
- corrected initial mistakes against damaged correct answers;
- calibration of the gate's prediction (expected calibration error, reliability curves;
  Guo et al. 2017).

**SRI equilibrium energy as a confidence feature.** Liu et al. (2020) derive their energy
score from classifier logits (their equation 4). Our minimum quadratic activity energy is
a different quantity, so it is an **untested** feature here. Its value as a predictor of
errors is measured (AUROC) against the margin, with controls:
- feature magnitude: the energy scales with the norm of the clamped input, so it is
  normalized by that norm and compared with the norm alone;
- medium and representation: reported per medium, and with the representation as a
  covariate.

## Computation

Notation: n = unclamped state size for one example (with sizes 64/32/16, n = 3 × 112 − 64
= 272), d = 16 (core inner), L = sequence length, N = items.

- **Ordinary solve:**
  - one Cholesky factorization of the n × n system per medium and parameter state,
    O(n³);
  - O(n²) per item for the triangular solves.
- **Revision with fixed context (v1):**
  - the added terms change the Hessian only in the core-inner d × d block: + μ I for the
    previous-item term, + μ SᵀS for the next-item term;
  - context enters only the right-hand side, so there is one extra factorization per
    medium, μ and context pattern (previous only, next only, both). It is reused across
    items: O(n³) per pattern;
  - per revised item: O(n²), plus the context items' ordinary solves (encoding and
    O(n²) each, often already computed in a sequence);
  - storage O(n²) per pattern.
- **Joint sequence inference (later variant):** a block-tridiagonal system of size L·n.
  Block Cholesky costs O(L n³) time and O(L n²) storage per distinct parameter state, with
  no reuse across sequences of different context patterns.
- **Persistent state in Part A:** as fixed-context revision, once per step. Training
  backpropagates through the step's solve, with the previous state held fixed (no
  gradient through time) in v1.
- **Controller:**
  - O(features) per decision;
  - training cost is one immediate plus one revised answer per training item, about 2×
    the ordinary inference cost on the controller's training set.
- **Calibration and selection:** threshold and c selection, any temperature fitting, and
  hyperparameter search for μ, on development data only. All counted as selection cost.
- **Measured:**
  - wall and CPU time per phase (context encoding, revision, controller training,
    calibration, selection);
  - solves and encodings per answer;
  - peak RSS;
  - following [computation reporting](computation-reporting.md), with timing kept apart
    from the deterministic results.

## Prior work to review before freezing

Named from memory unless marked:
- Sarnecka & Carey 2008, reviewed at abstract level;
- curriculum learning (Bengio et al. 2009); blocked versus interleaved practice in human
  category learning;
- calibration (Guo et al. 2017);
- energy-based detection of unfamiliar inputs (Liu et al. 2020): equation 4 checked via
  the review, a different energy from ours;
- adaptive computation (ACT, Graves 2016; PonderNet, Banino et al. 2021), early-exit
  networks, selective prediction and learning to defer;
- iterative refinement and constraint-based energy reasoning;
- amortized inference, for a later "practise until instinctive" step;
- dual-process accounts (Kahneman), as motivation only.

## Order of work

**Proposed:**
1. Settle the answer semantics, the information levels and the withholding policy above.
2. Run **B1** on existing checkpoints, with all five selectors, the inferred and oracle
   levels, and the perturbed-successor control. It is cheap and says whether the
   relations already carry correcting information.
3. Then Part A and B2.

## B1 protocol (fixed before running, 2026-10-28)

- **Checkpoints:**
  - number systems v1 C14 and I14 at step 3,000;
  - meta pilot arms ordinary, ordinary + dev, meta and meta-I.
  
  All are frozen.
- **Items:** every source representation's evaluation images, 40 per number:
  - half A (20) trains the controller and selects μ and the threshold;
  - half B (20) is reported.
  
  Main summaries exclude dots, the candidate system itself. Withheld means numbers
  unlabelled in that representation.
- **Candidates:** for each item, 9 fresh fixed-area dots images (counts 1–9) in shuffled
  order. They are rendered from their own streams, so they are identical across
  checkpoints.
  - Immediate answer: the candidate with the highest *same* score against the item's
    core-inner code.
  - Revised answer: the same, with the revised code against the same candidates.
- **Context.** The item sits between counts n − 1 and n + 1, one-sided at 1 and 9:
  - **collections (main):** fresh fixed-area dots images of n ± 1, with states inferred
    by their own solves;
  - **forms (secondary):** other evaluation images of the same representation at n ± 1;
  - **oracle bound:** the mean core-inner code of that count's dots images;
  - interior counts 2–8 are the main summary; 1 and 9 are reported separately.
- **Revision:** one solve of the item's system with
  ½μ‖c − S(c_prev)‖² + ½μ‖c_next − S(c)‖² added, context held fixed.
  - μ ∈ {0.1, 0.3, 1, 3, 10, 30, 100}, selected on half A by always-revise accuracy over
    all items.
  - Perturbed successor: A′ = RA, t′ = Rt, with R a random orthogonal matrix (recorded
    seed). Perception is unchanged.
- **Gate features (before revision):**
  - matching margin (best minus second-best *same* score) and best score;
  - equilibrium energy divided by ‖f‖²;
  - ‖f‖;
  - whether context is one-sided;
  - medium.
- **Controller:** multinomial logistic regression over help / hurt / no change on half A.
  Expected gain = p_help − p_hurt.
- **Threshold:** chosen on half A to maximize accuracy − c × revision rate, with c = 0.02
  accuracy per revised item, pre-registered here.
- **Selectors on half B:**
  - immediate;
  - always revise;
  - random at the gated rate (exact expectation);
  - gated;
  - oracle.
- **Also reported:**
  - the gated accuracy–rate curve over thresholds;
  - controller calibration (ECE of p_help, 10 bins);
  - AUROC for predicting immediate errors from energy/‖f‖², from ‖f‖ alone and from the
    margin, per medium;
  - the old label readout, immediate and revised, as a diagnostic.
- **Not in B1:** unreadable-input tests, joint sequence inference, and anchor counting.
- **Added before the full run, after the smoke run (2026-10-28): swapped-item control.**
  With grounded neighbours n ± 1, the context alone determines the answer, so revision
  could ignore the item. For each item row, the item is replaced by images of another
  number n′ of the same representation (recorded stream), keeping the context and the
  true successor.
  - Reported per μ: the share of revised answers that follow the context (n) and the
    share that follow the item (n′).
  - Integration = revised accuracy with the true item minus the share following the
    context with the swapped item. Near zero means the item contributes nothing beyond
    its position.
  - In the smoke run (4 items per number), revised answers followed the context about
    as often with swapped items as with true ones.
