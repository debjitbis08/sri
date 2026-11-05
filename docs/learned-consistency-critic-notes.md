# Learned consistency critic

Discussion 2026-11-05. User proposal; no frozen contract or implementation.

The proposed critic learns dependencies among operator outputs on collections,
flags unusual combinations online, and supplies an offline repair objective.
Readers and critic remain fixed during repair; the encoder changes subject to
preserving accepted evidence. This supplies a route for dependencies learned
in one domain to constrain another domain through shared operator outputs.

## Mechanism and qualifications

- **[derived]** Positive examples plus an unconstrained scalar energy minimized
  only on those examples admit constant energy. Specify a normalized density,
  contrastive objective, or masked conditional prediction. A small model that
  predicts each operator output from the others is a concrete option. Generic
  corruption does not require coding symmetry, but some corruptions are valid
  combinations; they cannot all be called logical contradictions.
- **[derived]** Statistical rarity and logical contradiction differ. Collection
  confidence, count frequencies and finite-range arithmetic can become shortcuts.
  Learned dependencies need not be universal laws. Masked prediction also learns
  conditional frequencies; it does not eliminate this problem.
- **[read: local results]** The premise that all collection outputs are about
  99% correct is not established by v2: collection `more` tie consistency is
  about 0.89–0.92. Critic targets include source errors, not just valid examples.
- **[derived]** Same-pair outputs can teach local compatibility such as equality
  versus strict order. To learn substitution, the critic must also see contexts
  such as `eq(a,b), R(a,m), R(b,m)` and the reversed argument positions. An
  equality-only triple learns dependencies within equality, not substitution
  into another operator. These are input relationships, not hand-specified
  rules; their validity must still be learned.
- **[derived]** Freezing readers does not freeze their outputs when their shared
  encoder changes. Accepted anchors require explicit output preservation on
  retained examples, not merely a confidence mask or stopped gradient. Protection
  remains limited to the examples constrained. Conflicting anchors should allow
  abstention. Trust-region or small-update limits alone do not ensure correctness.
- **[derived]** Repair needs confidence supported by retained teaching evidence
  and stability, not simply a large logit. Consistent wrong outputs remain
  unidentifiable to the critic. Preserving independent unequal evidence reduces
  collapse but does not rule out coherent wrong identities.
- **[derived]** The monitor is cheap given its complete inputs, but evaluating
  every operator in both orders is additional work when the original query
  requests only one. Cost grows with the operator inventory and any triple or
  context probes. Partial monitoring or deferred full evaluation are options.
- **[derived]** Chinese > Devanagari > collections in energy is neither necessary
  nor sufficient: confidence shift can produce this order without detecting
  contradictions, and coherent forgetting can leave Chinese low-energy. The
  proposed ordering is suggestive, not a decisive prerequisite.

## Prior work

Search 2026-11-05: `site.proceedings.mlr.press structured prediction energy
networks Belanger McCallum 2016`; `site.proceedings.mlr.press Guo calibration
modern neural networks 2017`. Primary abstract/search excerpts inspected.

- **[abstract]** [Belanger and McCallum, Structured Prediction Energy
  Networks](https://proceedings.mlr.press/v48/belanger16.html) learn dependencies
  among output labels and optimize predictions through the learned energy.
  Here the proposed repair optimizes encoder parameters during consolidation,
  which introduces retention and shared-parameter effects absent from merely
  optimizing one output vector. This is a related mechanism, not an exact
  implementation of SPEN or evidence that this repair will work.
- **[abstract]** [Guo et al., On Calibration of Modern Neural
  Networks](https://proceedings.mlr.press/v70/guo17a.html) finds neural confidence
  poorly calibrated and studies calibration methods. This supports caution about
  confidence anchors; it does not establish a solution under script shift.
- **[proposal / untested]** Masked operator prediction is a simpler alternative
  to an unspecified scalar energy. Conditioning on accepted evidence, predicting
  missing outputs, and consolidating those predictions may transfer dependencies.
  It still needs safeguards against source errors and distribution shift.

## Response to the review

Discussion 2026-11-05, after the review above.

- **Accepted.** Masked conditional prediction replaces the unconstrained energy.
  It is also purely neural and has no built-in relation form.
- **Accepted; [read: local results].** Collection `more` irreflexivity is
  0.89–0.92 at block 3 (`docs/equality-relations-v2.md`). The critic trains on
  teaching labels where they exist, which they do for collections, and not on
  reader outputs, so it does not learn source errors.
- **Accepted.** Substitution needs context inputs such as
  `eq(a,b), R(a,m), R(b,m)` in both argument orders. Single-pair inputs teach
  only local compatibility, such as `eq` against strict `more`.
- **Accepted.** Repair needs an explicit output-preservation loss on retained
  anchors (distillation-style), plus abstention when anchors conflict.
- **Accepted as limits.** Rarity is not contradiction, but a learned critic
  learns only dependencies by design. Mitigations: keep confidences out of the
  critic's input, and hold out count ranges so frequency shortcuts show.
  Consistent wrong outputs remain undetectable; Thai, which was never taught,
  gives no signal.
- **Not accepted as stated; [read: local results].** The kind-level energy
  ordering is not the main evidence. `docs/equality-diagnostics-v2.md` tests the
  signal pair by pair:
  - On the two-space base, false acceptances conflict 3–7 times as often as true
    ones.
  - Flagged pairs are false 61–75% of the time, against a base rate of 23–47%.
  - On the count base the signal is uninformative (0.77 against 0.78).

  The ordering is backing evidence.
- **Proposed staging:**
  1. A single-pair masked critic trained on teaching labels, with no repair. It
     should match the diagnostics' conflict precision.
  2. Add the context inputs and test substitution.
  3. Add repair with output preservation and abstention.

## Asking for examples when confused

User question 2026-11-05: can the conflict the critic detects let the system ask
for examples?

- **[proposal / untested]** Conflict is the trigger for a query, as in active
  learning. Unlike query-by-committee, the disagreement is among one system's
  operator outputs, not among models. A conflict names the pair and operators
  involved. The masked critic can also point to the least-supported output, so
  the question is specific: "is this form equal to this collection?" or "what
  count is this form?".
- **[read: local results]** Flagged-pair precision (61–75% against 23–47%) means
  asking about flagged pairs should find errors about twice as often as random
  asking, on the two-space base for taught scripts.
- **[derived]** Two triggers are needed:
  - **Conflict:** detects errors among taught kinds.
  - **Unfamiliarity:** a kind with no taught relations, such as Thai, gives no
    conflict signal and must trigger asking for another reason.
- **[derived]** Asking versus repair:
  - Conflict with supporting anchors: repair offline under output preservation.
  - Conflict with no support, or with contradicting anchors: abstain and ask.
- **[derived] Failure modes:**
  - The count base, where the signal carries no information.
  - Sampling bias from teaching only on queried examples. Mix in random
    examples, and read A after B.
  - Conflicts caused by source errors in taught relations, which send the
    question to the wrong item.
- **[proposal / untested] First test,** on the v2 snapshots and with no critic
  trained:
  1. Use the pair-level conflict count as the selection score.
  2. With a fixed teaching budget k, compare random, lowest-margin and
     conflict-flagged selection.
  3. Measure the gain on Chinese, then read A after B.

  Selection is nearly free; retraining per query batch is the measured cost.

## Learned probe selection

User direction 2026-11-05: the critic must not check every dependency. A learned
filter chooses a small set to check.

- **[derived]** The critic holds no stored rules. What is selected is a set of
  **probes**: operator outputs such as `R(a,m)` and `R(b,m)` for a context item
  `m`. The masked critic judges consistency only from the selected probes. The
  set is variable, so the critic should take a set of probe tokens, for example
  through attention.
- **[proposal / untested] Selection makes two choices:**
  - **The operator.** The inventory grows with every taught relation.
  - **The context item.** This is the expensive choice: n items per operator. A
    cosine first stage draws candidates from memory near `a` and `b`, and a
    learned score ranks them.

  The score is query–key matching:
  - The query is the relation code plus the codes of the queried items.
  - The key is the probe's operator code plus its item code.
  - Keys and values use learned projections, with no fixed addresses.

  Online, only the top k probes are evaluated.
- **[proposal / untested] Training signal: distil the full check.**
  1. Offline at consolidation, evaluate every probe on a sample of queries.
  2. Record which probes carried the conflict, or most changed the masked
     prediction.
  3. Train the filter to predict those probes from the query alone.

  The full check is paid for only on the consolidation sample.
- **[derived] Scaling:** online cost changes from O(|R| · n) probe evaluations
  per query to O(k), plus the candidate stage. Full-check cost is O(|R| · n) per
  sampled query at consolidation, and it grows with the operator inventory.
- **[derived] Failure modes:**
  - **No exploration.** An unselected probe never shows whether it was useful.
    Keep a fraction of random probes, and give newly taught operators an
    exploration period, since they have no history.
  - **Frequency shortcut.** The filter may learn what conflicts often rather
    than what decides this query. Report recall on rare error types separately.
  - **Inherited blind spots.** Selection can only lose information relative to
    the full check. What the full check misses (consistent wrong outputs,
    untaught kinds) the filter misses too. Its value is cost.
- **[derived] Substitution test.** If equality is substitutability, then:
  - For `eq(a,b)` queries, the useful probes are `R(a,m)` against `R(b,m)` across
    operators.
  - For `more(a,b)` queries, the useful probe is mainly `eq` on the same pair.

  A filter that finds this difference without being told has learned what
  equality is for. This is read from the selection pattern, not from accuracy.
- **[proposal / untested] First run.** Three operators (`eq`, `more`, `sum10`),
  both orders, and collection items as contexts keep the full check affordable,
  so it is complete ground truth. Measures:
  - recall of the full check's flagged errors against k;
  - the same for random k probes;
  - the selection pattern for each query operator.
- **To search before a contract (not yet checked):**
  - disagreement-based active learning (Settles' survey; Seung, Opper and
    Sompolinsky on query-by-committee) and sampling bias;
  - self-training and pseudo-labelling confirmation bias, for repair towards
    critic predictions;
  - Learning without Forgetting / distillation, for output preservation;
  - active feature acquisition;
  - top-k mixture-of-experts routing (Shazeer et al. 2017);
  - distilling a full model into a sparse one.

Any implementation or run requires a separate contract and declared training,
monitoring, replay and repair costs under the computation reporting policy.
