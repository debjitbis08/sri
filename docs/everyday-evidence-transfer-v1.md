# Everyday evidence transfer v1

**Contract frozen 2026-11-07 before training.** User direction: this is a learner for real-world data, including
mathematics, not a mathematical proof engine. User authorized design and execution. This study asks whether an
example-based component can use the existing learner's knowledge to acquire and apply a new property, revise an
explicitly taught exception, and preserve earlier knowledge. No rule strings, proof labels or transition supervision.

## Scope and reuse

Reuse the everyday-world animal facts and the four frozen `amortized-introduction-v1/virtual` readers (450–453).
Add one generic support-conditioned classification component. The readers, introduction encoder, walker and critic
are not retrained or replaced. New properties live in explicit example memory. This tests episodic learning, not
consolidation into shared weights or autonomous retrieval. The fixture is small and curated: 34 animals, not field
data. Animal names label existing entities; the new component never receives their text or a task identity.

## Task: learn a new grouping from examples

The fixture has 31 attributes: six classes, eleven body parts, five abilities, four habitats, five foods. A new
property is a fresh binary grouping of animals, demonstrated by labelled examples. Some groupings depend on one
attribute, others on combinations (e.g. swimming or climbing). Their names and generating definitions are hidden.
These are experimental groupings/preferences, not claims that an invented biological category exists.

Training entities: all animals except dog, whale, sparrow, penguin, salmon, bee, turtle and frog. Those eight are
held out from **all adapter-training contexts and targets**; the old reader may know them. This measures using old
knowledge about entities absent from the new component's practice, not learning entirely unknown entities.

Data generator, used only for labels: single attribute, intersection, union, exclusive-or of two attributes.
Keep concepts with at least six positives and six negatives among the 26 training animals, and at least two of
each among the eight held-out animals. Deduplicate **extensions and complements** before splitting, so alternative
descriptions cannot leak the same grouping across train/test. Assign a stable representative in order single,
intersection, union, exclusive-or. Fixed split seed 2041: 70% of non-exclusive-or concepts train, remainder novel
combination test. Exclusive-or concepts not extension-equivalent to earlier forms are a held-out structure test.
Report bank sizes and any shortfalls; never change quotas based on model performance. Every episode randomly
reverses category labels, so entity identity or zero-context guessing cannot identify the target.

### Practice and evaluation

- 2,000 Adam updates, lr 0.001; 32 episodes per update; no weight decay or tuning. Hidden size 64, decoder 128.
- Uniformly sample training concepts; context size 0, 4 or 8, with equal probability. For nonzero sizes sample equal
  positives/negatives without replacement. Four disjoint target animals, two of each class. Use only training entities.
- Context labels have independent 10% corruption during practice; targets use the underlying concept. This models
  imperfect observations. No label-noise indicator is supplied.
- In 25% of episodes, add an explicit exception: a separate animal, absent from context and ordinary targets, is
  given the opposite of its group label. Add that observation and ask about that animal. Its loss is included with
  ordinary target losses. This teaches specific evidence use, without requiring unseen arbitrary exceptions to be guessed.
- Checkpoints 500/1000/2000 are evaluated but never selected. The 2,000-step checkpoint is primary.
- Test up to 100 novel-combination concepts and 100 held-out-structure concepts, four contexts each, both polarities
  by deterministic alternating assignment. All ordinary questions concern the eight held-out animals. Supports
  come from training animals. Sizes 0, 4, 8 use nested balanced supports; report per-concept balanced accuracy,
  Brier score and confident-error rate (wrong with probability ≥0.9), averaged across contexts and seeds.

Test conditions at context size 8:

1. Clean examples; shuffled labels (same label counts); all labels reversed, with reversed query truth.
2. Corrupt 25% of labels (two observations); separately remove half of the supplied factual-profile entries with
   an explicit missing mask. This is missing source evidence, not a claim the predicate is false.
3. **Explicit exception:** choose one held-out query animal, append its opposite group label as a specific
   observation. Measure accuracy/probability on that animal before/after, and accuracy change on the other seven.
   This is a deliberate taught-query overlap, reported separately from generalization.
4. **Conflicting observations:** append both labels for that same animal. Report its confidence/entropy; there is
   no uniquely correct target label or imposed 0.5 answer under the noisy-observation model.

The source-profile masks and corruption draws are paired across neural arms. Predictions on old stored-reader
queries and previously stored new contexts are checked before/after new learning. Freezing and separate context
memory make retention structural; a pass is an integrity check, not evidence for solving catastrophic forgetting.

## Components and arms

Inputs have the same dimension in every neural arm: a normalized frozen 32-dimensional entity code, 31 positive
factual supports, and 31 observed-entry masks (94 total). `learned` profiles use the frozen logic route's support
for each (animal, relation, attribute) fact; no introduced-property truth is queried. `oracle` uses fixture 0/1 facts.
`codes` zeroes profile values and masks. With missing entries, zero the value and mask. All profiles are prepared
before training and kept frozen. Applicability is not needed for these known in-domain animal attributes.

- `codes`: entity codes alone, with zero padding for equal architecture/parameter count.
- `learned`: entity codes plus the existing reader's factual profile.
- `oracle`: same learner with correct source facts, an explicitly privileged diagnostic.

All arms use the same deterministic attention-conditioned classifier: input projection 94→64→64; element encoder
of [projected entity, ±label], 65→64→64; mean pooling over context and four-head cross-attention from each projected
query to context projections with encoded labelled examples as values; decoder [query, mean, attention] 192→128→1.
Sigmoid gives category probability. Empty context uses zero summaries. Padding is masked. This is a deterministic
CNP/attention-inspired classifier, not a full latent Attentive Neural Process. No module knows the generator's forms,
the relevant attributes, the concept index, or any mathematical rule. Standard modules stay unchanged across tasks.

**Cheaper baseline:** per-episode ridge least-squares classifier on the same normalized learned input vectors plus
an intercept, targets ±1, fixed λ=0.1, K×K dual solve. Map its score by `(score+1)/2`, clipped to [0,1]. No meta-training,
hyperparameter selection or task definitions. Empty context predicts 0.5. Report its Brier score as a heuristic
probability mapping, not guaranteed calibration. All test conditions apply to it too.

## Criteria and interpretation

Primary claim (novel combinations, learned arm): at eight clean examples, mean balanced accuracy ≥0.75, improvement
over zero examples and shuffled labels each ≥0.10. Report all seeds and the held-out-structure test separately;
the latter cannot be hidden by pooling with easier concepts.

Knowledge-use claim additionally needs ≥0.05 gain over `codes`. If ridge matches or beats the neural component,
prefer the simpler mechanism unless another measured capability justifies the extra training. Oracle gains expose
source-knowledge limits but do not alone prove that errors are exclusively in the reader.

Exception criterion: after explicit teaching, accuracy ≥0.80 on exceptions and ordinary-query balanced-accuracy
drop ≤0.05. No criterion for guessing an unobserved arbitrary exception. Counterfactual label reversal must yield
≥0.75 BA to support evidence-following. Confidence with missing/noisy/conflicting evidence is diagnostic, not a
claim of calibrated epistemic uncertainty. Report gains and false alarms together.

This is statistical learning and transfer in a controlled domain. It does not establish general reasoning,
natural-language understanding, unsupervised concept discovery, richer relation arities or lifelong consolidation.

## Prior work reviewed before freezing

2026-11-07: opened the primary abstracts of [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html),
[Matching Networks](https://arxiv.org/abs/1606.04080), and [Attentive Neural Processes](https://arxiv.org/abs/1901.05761).
Findings are recorded in [prior work](prior-work.md).

- **[abstract]** CNPs learn prediction from small context sets across tasks. This is the established method family;
  the present contribution is an internal test of its connection to the existing factual learner.
- **[abstract]** ANPs identify underfitting at observed context points and introduce query-dependent attention.
  **[hypothesis]** This may help explicit exception uptake here; the published result is not a guarantee for these data.
- **[abstract]** Matching Networks map a labelled support set and query to an answer without task-time fine-tuning.
  This motivates explicit example memory; our network is not their full architecture.
- **[derived]** Few examples may fit several concepts; correct outcomes need not be uniquely determined. We measure
  predictive performance, not proof correctness. Masked evidence must not silently become a negative fact.
- **[read: local results]** Amortized introduction showed copy transfer but poor inverse generalization; everyday
  v3 showed threshold shifts masquerading as gains. Here supports, counterfactual labels and BA/Brier/false alarms
  test evidence use. Frozen old knowledge prevents changing the source while diagnosing application.
- Known failures: overfitting task/entity identities (hold out both); extension leakage (deduplicate complements);
  support underfitting (attention and exception read); tiny correlated fixture (explicit limitation); confidence
  without information (zero/noisy/missing/conflicting reads). Cheap ridge is included. No novelty claim.

## Computation contract

N=34 entities, F=31 attributes, d=94 input, h=64, K≤9 practice context, Q≤5 practice targets, batch B=32,
T=2000 updates. Conflict test K=10. Sources are prepared in O(NF) neural-reader evaluations and O(NFd_reader²)
arithmetic for the frozen MLP, O(NF+Nd) stored floats; reader weights are retained separately.
Task construction enumerates O(F²N) labels, deduplicates O(F²N) byte keys, stores O(MN) int8 labels.
Support/query sampling O(BN) per update. Model training O(TB[(K+Q)dh+(K+Q)h²+KQh]); storage O(P+B(K+Q)h+BKQ)
float32 activations plus task data. Parameters P, gradients P, Adam moments 2P floats plus step counters.
Retained new-concept memory is K entity IDs and labels, O(K), referencing shared frozen profiles; counts/bytes
reported separately from model parameters and serialized checkpoints. No proof search, graph closure or beam.
Ridge construction/solve O(K²d+K³), prediction O(QKd), workspace O(Kd+K²). Count every per-context fit and its cost.

Four seeds × three arms =12 fits; estimates 1–3 CPU min per fit, 5–15 minutes wall on three single-thread workers,
<1 GiB per process. Smoke: all arms, seed450, 50 updates, separate files; mechanical checks only, no tuning.
Measure construction, sampling/training, checkpoint/final neural evaluation, ridge fits, retention checks and save
times (wall/process CPU). Record counts, shapes, source/checkpoint/contract hashes, revision/dirty status,
software/BLAS/thread/hardware metadata, parameters/buffers/optimizer/memory bytes, process RSS (not whole-job peak).
Scientific JSON excludes timing; runtime JSON reports aggregate worker CPU and smoke/verification/analysis separately.
Report quality versus total training/evaluation costs at equal data; no inference-latency or scaling claim.
Unavailable quantities are null with reasons. Every seed and final checkpoint is retained; no selection budget.

Runner: `experiments/everyday_evidence_transfer.py`; reports `reports/everyday-evidence-transfer-v1*.json`.
Results go in `docs/everyday-evidence-transfer-v1-results.md` so this frozen contract keeps its hash.

    EV_SMOKE=1 EV_WORKERS=3 .venv/bin/python -m experiments.everyday_evidence_transfer
    EV_WORKERS=3 .venv/bin/python -m experiments.everyday_evidence_transfer
