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

## Part C: curiosity, asking questions to learn (idea, 2026-10-29)

*Researchers' idea after B1.* B1 showed that the core can know a quantity exists
("between 3 and 5 there is a count, 4") without knowing how it is read or written in a
given language. A system that knows *what* it does not know can ask a targeted question.

**Two kinds of structured gap:**

| Gap | What the system knows | What it lacks | Question |
| --- | --- | --- | --- |
| writing (missing form) | the quantity's place among grounded counts | the form in this language | "How do you write the one after three?" |
| reading (missing meaning) | a symbol, and a relational hypothesis for it ("it should be four") | confirmation | "Is this four?" (yes/no) |

**Why the core matters here.**
- Without relations, a learner can only say "I don't know". With them, it can name the
  gap and propose a hypothesis, so one answer resolves it.
- The core's job becomes turning ignorance into a specific question.

**Confidence's role.** Ask when recognition is unsure *and* relational inference is
sure. That is sharper than "low confidence → think harder". A confident relational
hypothesis that recognition cannot confirm is a precise trigger.

**Learning from answers (fast mapping).** The answer is retained: the form joins the
system's knowledge, so next time it is recognized immediately. This is the "deliberate
answers become instinct" loop. It requires learning from the inference, which B1 (frozen
models) did not test.

**Core as teacher.** A related step, without an external answer: the core's own confident
inferences become training targets for the symbols, as in B2. It is checked by isolated
recognition afterwards. Part C adds the option of asking when the core alone is not
enough.

**Measure: answers needed to learn a new language,** with retention, then isolated
recognition. This is the learning-cost criterion from the [meta study](number-systems-meta-v1-spec.md).

| Learner | Questions |
| --- | --- |
| random | answers at random items |
| uncertainty-based (classic active learning) | asks where recognition is least certain, without relations |
| relation-guided (the proposal) | asks about detected gaps, with hypotheses |

If the relation-guided learner needs fewer answers at equal retention, the core pays for
itself in learning cost.

**Sketch.** Built on Part A's grounded counting:
1. Teach one language by counting episodes.
2. Introduce a new language with some forms missing or unreadable.
3. Let the system detect gaps, ask, and learn.
4. Count answers; measure retention and isolated recognition.

**Prior work to check** (from memory, not verified):
- active learning and query selection;
- Loewenstein's information-gap theory of curiosity (1994);
- intrinsic motivation and curiosity-driven learning (Oudeyer & Kaplan; Schmidhuber);
- children's question asking (Chouinard 2007);
- fast mapping (Carey & Bartlett 1978).

Active learning is established. Using a learned relational core to *generate* targeted
questions and hypotheses may not be, but that needs checking.

**Status:** idea only. Depends on Part A, and on a confidence signal that predicts when
recognition is wrong.

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
4. **Updated 2026-10-29 after B1 and B1-ambiguous.** B1 ran:
   - inference from relations works;
   - combining the item with the context adds little beyond filtering;
   - nothing is learned from the inference.

   Next comes **Part A**, designed so that confidence and learning from relational
   inference (core as teacher, then Part C) can be tested on its outputs.

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

## B1-ambiguous protocol (fixed before running, 2026-10-28)

B1 showed that with both neighbours grounded, the context alone determines the answer,
and revision ignored the item. Here the context is ambiguous and the item informative,
so only combining them can succeed.

- **Task: one adjacent grounded neighbour, side unknown.** Each item (form of count n)
  comes with one fixed-area dots image of count m, where m = n − 1 or n + 1 (recorded
  stream).
  - The system is told only that m is adjacent.
  - Only m ∈ 2–8 is used, so both hypotheses n = m ± 1 are valid counts.
  - The context alone gives at most 0.5.
- **Same checkpoints, item sampling, halves (A for selection, B reported) and candidates**
  as B1, with fresh streams. Dots excluded from the main summary.
- **Answers compared on half B:**
  1. **immediate:** 9-way collection matching from the item alone;
  2. **context only:** 0.5 by construction (both hypotheses equally valid);
  3. **restricted recognition:** the item's matching scores, restricted to the two
     context-consistent candidates (m − 1, m + 1). This is filtering with no relational
     inference, and the baseline revision must beat to show the successor adds anything;
  4. **energy hypothesis selection:**
     - revise under both hypotheses, h = +1 adding ½μ‖c − S(c_m)‖² and h = −1 adding
       ½μ‖c_m − S(c)‖²;
     - pick the hypothesis with the lower total energy at its revised state (item energy
       plus relation term);
     - answer by 9-way matching of the chosen revised state;
  5. **oracle side:** revise with the true side (B1's propagation), as a reference.
- **Controls:**
  - private-order checkpoints and the perturbed successor (as in B1);
  - **swapped item:** the item replaced by another number n′ of the same representation,
    keeping m. Reported: the share of answers on the true side, the other side, and n′.
    If the item is ignored, the true and other sides split about 50/50.
- **μ** ∈ {0.1, 0.3, 1, 3, 10, 30, 100}, selected on half A by energy-selection accuracy.
- **Groups:** all, withheld, labelled, image, sequence.
- **Integration** = energy-selection accuracy minus max(restricted recognition, 0.5). It
  is positive only if the relations add information beyond filtering by the context.
- **Not here:** gating (B1 showed nothing to select until revision is selectively
  useful), distance-two and noisy context (later variants).
- **Added before the full run, after the smoke run:** energy side choice. This is the
  share of items where the lower-energy hypothesis is the true side, read as the count
  m ± 1, before 9-way matching of the revised state. It separates hypothesis selection
  from final matching. Selection of μ is unchanged.

## Part A protocol (pilot, fixed before running, 2026-10-29)

Decisions taken autonomously while the researchers were away, as authorized. Each is
recorded here with its reason. This is a pilot; nothing is frozen.

**Languages and media (image only in v1):**
- L1 = english (words), taught first, with every count directly paired with collections.
- L2 = chinese and L3 = devanagari_digits, taught next, each with **two withheld counts**
  under the withholding policy.
- **Transfer language:** thai_digits, used only in the learning-cost test after training.
- *Reasons:* symbolic systems avoid the stroke-count shortcut of tally, Roman and
  Egyptian. Leaving out the sequence medium (Morse, Hand) keeps v1 to one satellite. The
  language is implicit in the form (no language token), so the "identify the context"
  test does not apply in v1.

**Withheld counts:**
- Drawn per seed from 2–8: two nonadjacent counts for L2 and two for L3, all four
  distinct (balanced across languages).
- Recorded in the results.

**Collections:**
- Pre-rendered incremental dot episodes: positions persist as counts grow from 1 to 9,
  so "add one" literally adds a dot and "remove one" removes the last.
- The radius is fixed within an episode and varies across episodes.
- Evaluation candidates are fresh fixed-area dots, as in B1.

**Episodes:**
- Start 1–8, length 3–9 within 1–9; 30% descending ("remove one").
- Each step shows the count's collection and its form, from the language's training
  pool. At withheld steps the collection is hidden: no matching or pairing loss for that
  form.
- Training is blocked by language: L1, then L2, then L3, with an equal number of updates
  per block. Collections appear in every lesson.

**Losses** (one image satellite, shared core; the output module is present but has no
loss, because no language is a label space):
1. **Matching:** cross-entropy over 9 candidate collections, using the judge's *same*
   score between the form's core-inner code and each candidate.
   - Candidates are one collection per count, shared within a batch, with the step's own
     collection at its count.
   - Only at steps whose collection is visible.
2. **Succession:** binary *next* judgments on consecutive steps, for forms (including
   withheld forms) and for visible collections. Descending steps are reversed pairs.
   Negatives are batch pairs whose counts do not differ by +1.

**Arms** (same evidence: the same episodes and the same pairs):

| Arm | Batches | State between steps | Extra |
| --- | --- | --- | --- |
| A-episode | whole episodes | persistent: each form solved with ½μ‖c − S(c_prev)‖² (remove one: ½μ‖S(c) − c_prev‖²), c_prev = previous form state, detached; μ = 1 | losses on episode states |
| A-order | whole episodes | none | — |
| A-shuffled | the same steps and pairs, mixed across episodes | none | — |
| A-episode+teach | as A-episode | persistent | **core as teacher:** each form's isolated core-inner state is pulled toward its episode state (detached), weight 1 |
| A-episode-interleaved | as A-episode, languages interleaved | persistent | the language-scheduling test |

- *Why μ = 1:* in B1, μ ≥ 1 made the context dominate. For A-episode that is the
  intended "carry the count forward" behaviour.
- *Why the teach arm:* B1 showed relational inference works but leaves no trace. This
  arm tests whether learning from it gives withheld forms their meaning in isolation.
  That is the fast-mapping step, and the most direct test of the core as teacher.

**Budget:**
- 3 blocks × 1,000 updates; batch 16 episodes; Adam 1e-3; seed 100.
- Sizes 64/32/16, as before.

**Evaluation** (evaluation pools, after training; also after each block for retention):
1. **Isolated collection matching** for every form of L1–L3 against 9 fresh fixed-area
   dots candidates, split into paired and withheld forms, per language. The acquisition
   test.
2. **Relational context for withheld forms:**
   - two-sided grounded context (B1 revision; answer and swapped-item control);
   - one-sided context with side unknown (B1-ambiguous: restricted recognition, energy
     side choice).
3. **Succession on held-out images:** *next* accuracy for collections and forms,
   consecutive against non-consecutive.
4. **Reverse direction:** for a collection, choose the form among the language's nine.
5. **Retention:** L1 matching after the L2 and L3 blocks.
6. **Confidence hooks.** Per-item signals are logged: margin, best score, energy/‖f‖²,
   ‖f‖ and agreement between isolated matching and relational inference. Reported:
   - AUROC for predicting isolated errors;
   - AUROC for separating withheld from paired forms;
   - the accuracy of items where agreement holds.
7. **Learning cost (transfer):** thai_digits taught by counting episodes with every
   count paired, 4 examples per count, Adam 1e-3 on the whole model. Recorded:
   - isolated matching at 0, 25, 50, 100 and 200 updates;
   - retention of L1–L3.

**Expected readings:**
- **Acquisition:** high isolated matching on withheld forms in any arm means relational
  grounding through succession works (as in the cross-relation pilot).
- **Curriculum:** differences between A-episode, A-order and A-shuffled show whether
  coherent order matters beyond the same evidence.
- **Learning from inference:** A-episode+teach above A-episode on withheld forms, in
  isolation.
- **Scheduling:** blocked against interleaved, in forgetting.

**Not in this pilot:**
- per-language adapters (problem 1);
- the sequence medium;
- boundary extrapolation;
- a language-identification test;
- gating (needs the confidence analysis first);
- Part C;
- multiple seeds. A three-seed rerun follows if the pilot runs cleanly.

### Part A pilot 2 amendments (2026-10-29, after pilot 1, before running)

Pilot 1 is reported in [Part A results](number-construction-v1-part-a-results.md).
Changes:

1. **Persistent state is coupled to the grounded previous step.**
   - Each form is coupled to the successor of the previous step's **collection** state
     when that collection is visible. Only at a step after a hidden collection is it
     coupled to the previous form's episode state.
   - Withheld counts are nonadjacent, so chains are at most two links long. In pilot 1,
     coupling to the previous coupled form compounded along whole episodes and diverged.
   - It is also the intended reading: "the count continues from what you saw".
2. **Gradient-norm clipping at 5** in every arm, a uniform safety change.
3. **Training collections:** half of the incremental episodes keep the total area fixed.
   The radius shrinks as dots are added (R/√k, at least 0.9), with positions kept and
   spacing set at insertion. So ink amount does not give the count in training either,
   matching the fixed-area test candidates.
4. **2,000 updates per block** (pilot 1: 1,000 was too short for English and Devanagari).
5. **The full evaluation runs at the end of every block,** not only at the end.
   - **Primary acquisition measure:** isolated matching of each language's withheld
     against paired forms **at the end of its own block**, before later blocks can cause
     forgetting.
   - The end-of-training evaluation measures retention.

Mode `pilot2`, so pilot 1's artifacts are kept.
