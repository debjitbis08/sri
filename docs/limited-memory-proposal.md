# Limited, decaying memory as pressure to consolidate

Draft, 2026-10-30, revised four times the same day. Not frozen.

**Revisions.**
- The first draft modelled memory as an external store of training episodes. An
  external review found four problems in it. The researchers then set the model of
  memory: **memory is the plain, dense network**, not a separate structure.
- The second draft put memory in decaying dense reader units. A mathematical review,
  [decaying readers theory](decaying-readers-theory.md), found two factual errors about
  the baseline and several gaps in measurement. This version makes those corrections
  (see "Review points"). The researchers also clarified internal replay: it **replays the
  dense memory part, so that concepts form quicker** in the sparse part.
- The [timing run](fan-in-pilot-v1-timing-results.md) has finished. Neither the
  coordinate nor gradual timing reproduces RigL, so that draft's structure part used
  **RigL** wiring (superseded in the fourth draft by random rewiring).
- The [reset](fan-in-pilot-v1-reset-results.md) and [erasure](fan-in-pilot-v1-erasure-results.md)
  runs and the [count-coordinate analysis](count-coordinate-analysis.md) have since
  finished. This version adds what they show: disrupting sparse connections, not
  moving them to a better place, is what helps; dense readers gain nothing from it; the
  count is spread over most code coordinates, so the right transfer measure is the
  average coordinate. It adds a reset arm beside decay, an average-coordinate measure,
  checkpoints at each phase, and the prior work that was missing (see "Review points").
- A second external review and the researchers' decisions: **random rewiring is the
  default wiring for the structure part, RigL optional**; the study is meant to run on
  the GPU if a real timing and a determinism check allow it. The review's six points
  are addressed (see "Review points"): a checkpoint-fork experiment for handoff,
  a fully specified reset control, a corrected replay limit, a corrected decay bound,
  a frozen selection and confirmation protocol with full costs, and a corrected claim
  about dense units.

## Question

Can one network answer from memory at first and form transferable structure later, if
its memory is limited and fades?

The researchers' view:
- **Memory is the dense network.** Without sparse reads, a network fits taught
  pairings, arbitrary ones included. It needs no separate structure.
- **Memory is limited and decaying, and holds recent training more strongly.** To keep
  something long term, the learner has to form structure that can be stored.
- **The learner must stay able to learn again,** both new and old material, by training
  again.
- **Not everything is a concept.** Some things, such as a route, are pure memory.
- **Replay is external or internal.** External replay takes help from an oracle, the
  training data. Internal replay randomly replays the learned state of the dense memory
  part, so that concepts form quicker.
- **Start with the readers.** Whether memory is needed elsewhere has to be discovered.

## What is already known

- **Dense readers fit taught questions better; readers on one shared coordinate
  transfer better on some measures.** RigL at one dimension per operand gave large
  untaught transfer on all three scripts, at a cost to taught matching and retention
  ([fixed-coordinate results](fan-in-pilot-v1-fixed-results.md)). This is not a clean
  split into memory and concepts: a reader on one scalar can still store arbitrary
  finite identities.
- **Units on other coordinates undo the concentration.** In `read-q16-fixed75`, all 64
  units read one dimension per operand: 48 on the shared coordinate, 16 on their random
  coordinates. Transfer fell back to near the random arm. **None of those units was
  dense**, so this run did not test a dense memory part. A dense part beside a sparse
  one is new, and its effect could go either way: it adds a bypass, but also broad
  gradient access.
- **No unit-preserving arm has reproduced RigL's English and Devanagari gains**
  ([timing results](fan-in-pilot-v1-timing-results.md)). RigL's dead units, partial
  concentration and repeated rewiring were candidates; the later runs narrowed them.
- **Disrupting sparse connections is what helps** ([erasure results](fan-in-pilot-v1-erasure-results.md)).
  Random rewiring with restart at zero gets most of RigL's transfer with no
  concentration and no dead units; resetting in place, or moving while keeping weight,
  each beats random wiring; **dense readers reset at the same rate gain nothing**; half
  the rate loses part of the gain and twice the rate adds little. So the structure
  part's wiring is a choice (see "Open choices"), and a dense part is not expected to
  gain from erasure.
- **The count is spread over the code, not on one coordinate**
  ([count-coordinate analysis](count-coordinate-analysis.md)). In every arm, dense
  included, 13–16 of 16 coordinates individually order collections. The arms differ in
  whether the average coordinate's count map transfers to taps and forms (Chinese forms
  0.60 dense, 0.90 random rewire). Readers did not concentrate on the best coordinate
  (0.06–0.09 of reads, chance 0.0625). The reading, untested, is that disrupted sparse
  readers cannot depend on any coordinate, so every coordinate has to carry the count.
  Retention could not be examined from final checkpoints.

## Design

### Readers with a decaying memory part

The first layer of `g` (40 inputs: two 16-dimension codes and an 8-dimension relation
token) and of `P` (32 inputs: two codes) has 64 units each.

- **48 structure units** read one code dimension per operand. **Default wiring: random
  rewiring with restart at zero** (the `read-q16-untied-rewire` arm of the erasure run:
  every 100 updates each unit moves each operand's connection to another coordinate
  with the run's per-layer probability, the new weight and its Adam moments start at
  zero). RigL is optional (see "Selection and confirmation"). `g`'s relation-token
  columns stay fully connected for these units, as before. Their weights do not decay.
  - **Rewiring runs on the structure rows only.** `rewire_step` asserts one connection
    per operand in every row of the columns it handles; it must be given the structure
    rows and skip the dense rows.
  - **Rewiring stops at update 4,500,** as in the fan-in runs. The longer curriculum must
    not stretch it into the delay or relearning.
- **16 memory units** read all of their inputs, as dense units do: both codes, and in
  `g` the relation token. Their parameters decay after every optimizer step:
  - **incoming weights and biases**, token weights included, decay toward their starting
    values: w ← w₀ + (w − w₀) · 2^(−1/H);
  - **outgoing weights,** their 16 columns in the second layer, decay toward zero at the
    same rate. The rest of the second layer and the third layer do not decay.
- **Starting weights.** Every arm's first-layer rows start from the same standard draw,
  as in the fan-in runs. Dense rows keep the standard initialization without rescaling.
  All arms with a memory part start from identical weights, whatever their H.

Why both directions decay:
- **Recent training is held most strongly.** With w = w₀ + u, decay after the update
  gives u_T = r^T u₀ + Σ r^(T−t) Δ_t with r = 2^(−1/H): older increments are weighted
  down exponentially. This is exact for the actual Adam increments Δ_t. It is not a
  half-life for an association or an answer.
- **Outgoing decay removes influence in the end.** On fixed inputs and with fixed
  downstream layers, the review bounds the memory part's contribution by an envelope
  C₁rᵗ + C₂r²ᵗ ([theory](decaying-readers-theory.md), bound 2). It vanishes eventually,
  but not monotonically relative to its initial influence: incoming weights returning
  toward their start can switch silent ReLUs on, so influence can first rise. Incoming
  decay alone would leave random features that the next layer could still use.
- **The conditions do not hold in training.** The encoders and later layers keep
  learning. In particular, the non-decaying layers can scale up to compensate for
  shrinking outgoing columns (ReLU's positive scaling makes this exact). Small memory
  weights therefore do not show forgetting; it must be measured by what the output
  depends on (see "Measures").

**Decay and penalties.** Under plain SGD, this decay equals an L2 penalty toward the
starting weights. Under Adam they differ: decay applied after the step is decoupled from
the adaptive gradient (Loshchilov and Hutter). It is a forgetting rate in that specific
sense, not a different kind of thing from a penalty.

### How decay can hand knowledge to structure

The review solves a two-path model exactly. A slow path s and a fast, decaying path m
predict a target together; training continues on that target. With decay, the only
resting point is the whole target carried by the slow path and nothing by the fast one.
Decay alone moves the answer into structure, with no distillation step, if two
conditions hold:

- **The structure part can represent the target.** If it cannot, decay creates error,
  not consolidation, whatever H is.
- **The target keeps being presented.** During a period without forms, decay erases form
  knowledge rather than consolidating it.

The handoff is slow. Its half-life is about H × (1 + b²/a²), where a and b are how
strongly the slow and fast paths affect the output. With b = 4a, H = 500 gives about
8,500 updates, longer than training. In that example, after 6,000 steps the slow and fast
paths carry 0.42 and 0.57 of the answer, with a nearly perfect combined fit.

Consequences for the design:
- **The main runs cannot identify handoff.** Higher structure-only accuracy with lower
  memory dependence also follows if decay suppresses the dense bypass from the outset
  and structure learns directly. The main runs therefore support only "greater reliance
  on structure"; handoff is tested by the checkpoint fork below.
- **Consolidation happens during learning and rehearsal,** not during the delay. The
  structure part must be measured before forms disappear.
- **A short run with incomplete handoff is not evidence against consolidation.**
- **Internal replay of the dense part** (stage 2) can keep presenting what memory holds
  after the examples are gone. That is its purpose: concepts form quicker in the
  structure part.

### Curriculum

The pilots' D-varied+rehearsal, unchanged, so arms compare with the fan-in runs. Then two
phases, the same in every arm:

- **Delay:** 2,000 updates of context episodes only (taps, collections, comparison). No
  forms and no rehearsal of forms. The delay is not free of learning: context gradients
  and Adam's moments still move the memory units. It measures what survives continued
  training without forms.
- **Relearning:** 200 updates of English language episodes, as in block 1.

The existing rehearsal buffer (external replay) is kept during the Chinese and
Devanagari blocks, in every arm.

### Checkpoint fork for handoff

Handoff means knowledge the memory part held moves to the structure part. To see it,
memory dependence must exist first, then decay must be compared with no decay under
identical continued teaching.

- **Fork point:** the end of block 3 of `mixed` (no decay), per seed. The saved
  checkpoint holds the model, the optimizer state (Adam moments), the rehearsal buffer
  and the random-stream positions, so every continuation starts from the same state and
  sees the same data.
- **Entry rule:** a seed enters the handoff reading if silencing the memory part there
  lowers Chinese matching by more than 0.05 (the audit's margin). Other seeds are
  reported and excluded: there is nothing to hand off.
- **Continuations:** 2,000 updates (one block, declared) of the block-3 curriculum with
  its rehearsal, forms still presented, with H ∈ {∞, 2000, 500, 125} from the fork.
  H = ∞ is the no-decay control.
- **Handoff reading:** against the H = ∞ continuation from the same checkpoint, a
  decaying continuation has higher structure-only accuracy and lower memory dependence,
  paired by seed, margin 0.05.

### Arms

Fresh seeds, ten per arm.

| Arm | First layer of each reader | Half-life H (updates) |
| --- | --- | --- |
| `dense` | 64 dense units | — |
| `sparse64` | 64 rewiring units | — |
| `sparse48` | 48 rewiring units, no memory part | — |
| `mixed` | 48 rewiring + 16 dense | ∞ (no decay; a new baseline) |
| `mixed-h2000` | same | 2,000 (one block) |
| `mixed-h500` | same | 500 |
| `mixed-h125` | same | 125 |
| `mixed-reset` | same | ∞; reset control, defined below |
| `sparse64-rigl` (optional) | 64 RigL units | — |

**Reset control, defined.** Matches `reset_step` (`experiments/fan_in_pilot.py`) applied
to the 16 memory rows only:
- **Parameters:** the 32 code-input weights of each memory unit (both operands) in `g`
  and in `P`: 16 × 32 per layer. Biases, relation-token weights and outgoing weights are
  not touched; decay touches them. The two interventions are therefore not matched in
  what they change, and the contrast is "decay as specified" against "reset as
  specified".
- **Action:** each selected weight restarts at **zero**, not at its starting value (decay
  returns to the start value), and its Adam first and second moments are zeroed.
- **Probability:** each weight, independently, with the layer's `REWIRE_P` (0.357 for
  `g`, 0.230 for `P`), as `dense-inplace` did.
- **Cadence and stop:** every 100 updates, last at update 4,500, then none, so no reset
  occurs in the delay or relearning, while decay continues through them.

- `mixed-reset` compares decay with the established form of disruption. Decay and reset
  both remove memory-unit content, but only reset has been run, and on dense readers it
  gained nothing. The question is whether decay does better on the memory part, not
  whether reset helps the structure part.
- `sparse48` is the right short-H limit: as H shrinks, the memory part goes silent and
  leaves 48 structure units, not 64.
- The three half-lives are declared candidates, not tuning. A claim that rests on one
  successful H must account for all three and be confirmed on fresh seeds.

### Stage 2: internal replay of the dense memory (not in this design)

The researchers' intent: replay the dense part, not the sparse part, so concepts form
quicker.

- **What it does.** Excite the readers with internally generated codes. Take what the
  memory part adds to the answer: a frozen copy's full output minus its output with the
  memory part silenced. Train the structure part to give it. This keeps presenting the
  memory's content after the examples are gone, so the handoff can continue.
- **Its limits** (review, section 5):
  - **The structure part can only learn what its inputs determine.** Under squared
    error, the best it can do is the memory's average answer given what the structure
    part sees as a whole: the coordinates its 48 units jointly read (each reads one per
    operand, so together they may cover many or all of the 16) and the relation token,
    through its units' one-coordinate features. The loss is whatever the memory knows
    that those features cannot express, not everything outside one coordinate. How much
    that is has to be measured; it is not assumed to be large.
  - **Random codes may not look like real ones.** They can train behaviour on inputs the
    network never meets, and miss the learned forms entirely. Codes drawn to match the
    statistics of recent codes would be closer. Keeping those statistics is a small
    summary, not a store of experience.
  - **It does not train the encoders.** Replay through the readers cannot protect how a
    form is encoded.
  - **The memory's "contribution" is defined only relative to the rest.** The shared
    layers are nonlinear, so it is a difference between two outputs, not a separate
    store.
- Without a model of the data, this is pseudorehearsal (replay on synthetic inputs). It
  is deferred until stage 1 shows how decay behaves.

### Later studies

- **Routes.** Arbitrary pairings with no shared structure, to compare recall of
  concept-consistent and arbitrary items after decay.
- **Interference.** After the delay, heavy training on a conflicting mapping, to test
  whether the count axis survives while specifics are overwritten.
- **Asking questions** ([Part C](number-construction-v1-spec.md#part-c-curiosity-asking-questions-to-learn-idea-2026-10-29)).
  The structure part supplies the hypothesis ("this should be four"), the memory part
  takes the answer at once, and repeated questions keep the target presented during
  handoff. Prediction: spaced questions about fading items consolidate better than the
  same number asked together. This needs a confidence measure that this design does not
  provide.

## Measures

- **Over time,** every 500 updates:
  - matching per script, for the current block and earlier ones;
  - comparison on forms; taps → withheld forms.
- **What the answer depends on.** At each evaluation, every matching and comparison
  measure is scored four ways: full network; memory part silenced (outgoing columns
  zeroed); structure part silenced; both silenced. Report:
  - accuracy and score margins for each;
  - the interaction residual, f(s,m) − f(s,0) − f(0,m) + f(0,0). Shared nonlinear layers
    can make both parts needed together, which would otherwise look like neither having
    learned.

  These measure dependence under intervention, not separate stores.
- **Consolidation before forms disappear:** structure-only (memory silenced) accuracy at
  the end of each block, and at the end of block 3.
- **Gain compensation:** norms of the second and third layers, and of the memory
  columns, over time.
- **Decay-only diagnostic:** a copy of the end-of-block-3 checkpoint with every
  parameter frozen except the decay, run for 2,000 steps. It separates pure decay from
  what training during the delay does.
- **Reactivation:** per memory unit, activity on evaluation codes, outgoing-gradient
  magnitude, and time to become active again in relearning. Returning to random weights
  does not guarantee a unit responds again.
- **Relearning:** English matching after 25, 50, 100 and 200 relearning updates, as a
  curve from each arm's own accuracy at the start of relearning. **Scalar:** the
  relearning gain, the trapezoid mean over 0–200 updates of (accuracy − accuracy at 0),
  reported beside the accuracy at 0.
- **At the end of block 3, after the delay, and after relearning:** the sequence pilot's
  full measures, the arrangement probe and the fan-in code-structure diagnostics.
  **Save a checkpoint at each of the three points** (and at the end of block 1); the
  count-coordinate analysis could not study retention because only final checkpoints
  existed.
- **Count on the average coordinate:** the [coordinate probe](count-coordinate-analysis.md)
  at the same points: order accuracy of a collection-fitted count map on taps and each
  script's forms, **averaged over all 16 coordinates** (the transfer measure of
  record), with the best-on-collections coordinate and the linear-direction alignment
  reported beside it. Not computed at a chosen coordinate.
- **Under reader disruption:** at the same points, the eight transfer measures again
  with a random 25% of each reader's first-layer connections zeroed (fresh draws,
  ten per checkpoint). A count that every coordinate carries should survive; a count that
  the readers depend on specific connections for should not. Descriptive; the 25% is
  a declared level, not tuned.

## Predictions

Written before any run.

- **Not monotone in H.** Shorter H removes the memory part's bypass, but also shortens
  the time for handoff. A middle H getting both taught matching and transfer does not
  follow from theory and may not happen.
- **`mixed` (no decay)** fits taught questions at least as well as `dense`. Its transfer
  is unknown: a dense bypass may undo the concentration, as random coordinates did, or
  its broad gradients may help.
- **Structure-only accuracy at the end of block 3 rises as H shortens,** if handoff
  happens within training. Incomplete handoff is likely: the toy model suggests it takes
  longer than training.
- **After the delay,** English matching falls most in short-H arms. What remains is
  carried by the structure part.
- **Average-coordinate transfer** (Chinese forms, taps) falls in `mixed` relative to
  `sparse48`, if the dense part gives readers a path that does not need every coordinate
  (as random coordinates did in `fixed75`). It recovers as H shortens, if decay silences
  the memory part. `mixed-reset` may stay near `mixed`, as dense readers gained nothing from
  reset, but that run reset an entirely dense reader; dense units beside sparse ones are
  untested. If decay does no better than reset on the memory part, the
  decay story has no support beyond reset.
- **Relearning** is faster from structure-rich arms, on the curve measure, for the count
  ordering. A lost link between a specific shape and its count cannot be restored by
  structure without new examples.

## Decision rules

Margins as in the audit: 0.05 for accuracies, 0.03 for AUROCs. Paired 95% t-interval
rule.

- **Contrasts:**
  - each decaying arm against `sparse48`, `sparse64` and `mixed`;
  - `mixed` against `dense` and `sparse64`;
  - the selected decaying arm against `mixed-reset` (confirmation run).

  On: the eight transfer measures after block 3; Chinese paired at its block;
  structure-only accuracy at the end of block 3; English matching after the delay; the
  relearning curve.
- **Readings:**
  - **Memory and structure both:** some H is equivalent to or better than `sparse48` on
    taps → withheld Chinese and English comparison, and better than `sparse48` on Chinese
    paired at its block. Any such H must be confirmed on fresh seeds.
  - **Greater reliance on structure:** structure-only accuracy at the end of block 3 is
    better than in `mixed` for some H, with the memory part's dependence lower. This does
    not show handoff.
  - **Handoff:** only from the checkpoint fork (see above).
  - **Erasure without handoff:** the memory part's dependence falls with no rise in
    structure-only accuracy.
  - **Gain compensation:** memory-column norms fall while dependence on the memory part
    does not.
  - **Decay beyond reset:** the selected H is better than `mixed-reset` on
    average-coordinate transfer (primary endpoint a below) in the confirmation run.
  - **Inconclusive is not equivalent.**

### Selection and confirmation

Frozen here, before any run.

- **Primary endpoints** (end of block 3 unless stated), paired by seed:
  - **(a)** average-coordinate transfer: the mean over taps, Chinese forms and
    Devanagari forms of the order accuracy averaged over the 16 coordinates;
  - **(b)** Chinese paired at its block (taught), the underfit check;
  - **(c)** structure-only Chinese matching accuracy;
  - **(d)** the relearning gain.
- **Selection:** among the three decaying arms, H is chosen by the highest seed-mean of
  (a), provided the arm is not underfit (its (b) is within 0.05 of `sparse48`'s). Ties
  go to the longer H. No other endpoint is used to choose.
- **Confirmation:** the selected H, `mixed`, `mixed-reset` and `sparse48` are rerun on
  ten fresh seeds. The readings above are decided on the confirmation run alone; the
  selection run is descriptive. Intervals are unadjusted paired 95% intervals, as in the
  earlier pilots.
- **RigL versus rewiring:** the default stays unless `sparse64-rigl` differs from
  `sparse64` by more than 0.10 (twice the audit margin) on endpoint (a), with the paired
  interval excluding zero. The earlier reset runs already found rewiring to get most of
  RigL's transfer, so the optional arm is run only if the researchers want the check
  repeated in this design.

## Prior work and expected limits

Search and read date: 2026-10-30. Read today: the abstracts of Chaudhry et al. (2019),
Khandelwal et al. (2020) and Power et al. (2022). The decaying readers review read the
abstracts of Hinton and Plaut (1987), Li et al. (2018) and Loshchilov and Hutter. Kumaran
et al. (2016) was seen through search snippets and summaries; the earlier analysis read
selected sections. Everything else is from memory, marked **[familiar]**, and must be
read before this is frozen.

- **Fast and slow weights.** [Hinton and Plaut 1987](https://www.cnbc.cmu.edu/~plaut/papers/abstracts/HintonPlaut87CogSciConf.fastWeights.html)
  **[abstract]**: slow weights with fast decaying overlays; rehearsal restores blurred
  associations. Ba et al. (2016) use fast weights to attend to the recent past
  **[familiar]**. Here the fast and slow parts also differ in connectivity: the slow part
  is sparse, the fast part dense. The handoff result is **[derived]** in the review.
- **Complementary learning systems.** A fast system for specifics and a slow one for
  shared structure (McClelland, McNaughton and O'Reilly 1995) **[familiar]**. The 2016
  update ([Kumaran, Hassabis and McClelland](https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf))
  adds that knowledge consistent with existing structure is integrated faster. Here both
  systems live in one reader.
- **Decay toward a reference.** [Li et al. 2018](https://proceedings.mlr.press/v80/li18a.html)
  **[abstract]** anchor fine-tuning to pretrained weights. Here the reference is the
  random starting point. [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101)
  **[abstract]** separate L2 penalties from decay applied after the step under adaptive
  optimizers; the decay here is the second kind.
- **Weight decay and late generalization.** Weight decay is credited with grokking
  **[familiar; the abstract of [Power et al. 2022](https://arxiv.org/abs/2201.02177)
  does not mention it]**. Power et al. report generalization well after overfitting
  **[abstract]**.
- **Small replay memories** ([Chaudhry et al. 2019](https://arxiv.org/abs/1902.10486),
  **[abstract]**): external replay of a tiny episodic memory helps and does not hurt
  generalization. The pilots' rehearsal is of this kind and is held constant.
- **Pseudorehearsal and generative replay.** Replay on synthetic inputs is
  pseudorehearsal (Robins 1995) **[familiar]**; replay from a model of the data is
  generative replay (Shin et al. 2017, [continuous-learning notes](continuous-learning-notes.md#two-different-uses-of-internally-generated-experience)).
  Stage 2 is the first unless it models the code distribution. Both fit the
  [foundations'](foundations.md) commitment to replay with no store of experience.

- **Forgetting that helps generalization.** Periodic reinitialization of part of a
  network (Zhou et al. 2022, "fortuitous forgetting") and iterated learning (Ren et al.
  2020) **[familiar]**; notes in [forget-relearn-iterated-learning-notes](forget-relearn-iterated-learning-notes.md)
  and the [after-reset analysis](rigl-after-reset-analysis.md). Shrink-and-perturb
  (Ash and Adams 2020) **[familiar]**: shrink weights toward zero and add noise to keep
  a warm start trainable; the decay here shrinks toward the start point and is applied
  per step.
- **Dropout and DropConnect.** Randomly zeroing units or connections each step
  (Srivastava et al. 2014; Wan et al. 2013) **[familiar]** is the established
  counterpart of the reader disruption here and of the disruption test above. It is not
  an arm: the fan-in results use persistent sparse wiring that changes at a set rate,
  which differs, and the cost of adding it to this design is a separate run.
- **Weight-decay variants as the cheaper baseline.** Plain L2 or decay on the whole
  reader, with no sparse part, is the cheapest alternative to a memory/structure split
  and is covered by `dense` plus the established decay results above. Whether it
  matches `mixed-h*` is not tested; the arms differ in connectivity as well as decay.

**Known failure modes.**
- **Gain compensation** by the non-decaying layers. Measured by dependence and norms.
- **Renewal during the delay** by context gradients and Adam's moments. Separated by the
  decay-only diagnostic.
- **Units that do not reactivate.** Measured directly.
- **Incomplete handoff within training.** Expected from the timescale; not read as
  failure.
- **Dense units undoing sparse disruption (hypothesis).** In `fixed75` the residual
  units were sparse, on random coordinates, and the dense-reset arm reset an entirely
  dense reader. Neither tests 16 dense units beside 48 sparse ones. `mixed` is where that
  is measured; it is the baseline the decaying arms must beat.
- **Silencing is approximate:** it changes the inputs the later layers were trained on.
  Reported as dependence with interaction terms.

**What this adds.** Whether a decaying dense part beside a sparse rewiring part gives taught
answers from memory while keeping the sparse part's transfer, whether it hands knowledge
to structure within training, and how that depends on the half-life. The theory predicts
handoff under conditions the protocol does not guarantee; whether it happens here is
open.

## Review points

**External review of the first draft (external store).**

| Point | Status |
| --- | --- |
| Writing from mixed batches refreshes replayed episodes | No store; does not apply |
| Stored episodes contain counts the learner never saw | No store; does not apply |
| The delay is ambiguous, and fading does not empty a store | Delay defined: no forms and no rehearsal of forms in any arm |
| Add a uniform-replay FIFO control | External replay held constant across arms |

**[Decaying readers theory](decaying-readers-theory.md) (second draft).**

| Point | Change |
| --- | --- |
| `fixed75` had no dense units | Corrected in "What is already known"; `mixed` is a new baseline |
| Short H gives 48 units, not 64 | `sparse48` arm added |
| Decay weights increments; delay is not update-free | Stated; decay-only diagnostic added |
| Decay equals L2 under SGD | Stated under "Decay and penalties" |
| Shared layers can compensate for shrinking columns | Dependence and norm measures added |
| Handoff only while the target is presented, and slowly | New section; structure-only accuracy measured before the delay |
| Ablation is not additive | Interaction residual and both-silenced baseline reported |
| Reactivation not guaranteed | Reactivation measures added; claim softened |
| Relearning measured against end of block 1 mixes retention and reacquisition | Curve from the start of relearning |
| Stage 2 representability and coverage limits | Stated in stage 2 |
| Token weights, RigL row assumption, rewiring stop, starting weights | Specified |
| Half-life grid needs confirmation | Stated in "Arms" and decision rules |

**Reset, erasure and count-coordinate results (third draft).**

| Point | Change |
| --- | --- |
| Dense readers gain nothing from reset at the same rate | `mixed-reset` arm added; "What is already known" and predictions updated |
| The count is on most coordinates, not one | Average-coordinate measure of record; per-unit story dropped |
| Retention could not be studied from final checkpoints | Checkpoints at each phase |
| Disruption helps sparse readers | Reader-disruption test added; structure wiring made an open choice |
| Forgetting, shrink-and-perturb and DropConnect not cited | Added under prior work, tagged **[familiar]** |

**Second external review (fourth draft).**

| Point | Change |
| --- | --- |
| Handoff rule does not identify transfer from memory to structure | Renamed "greater reliance on structure"; checkpoint-fork experiment added |
| Reset control defined two ways | One definition from `reset_step`: parameters, zero restart, Adam moments, probability, cadence, stop; unmatched with decay, stated |
| Replay limit applied to one coordinate | Conditions on what the 48 units jointly read |
| Decay bound stronger than derived | Envelope C₁rᵗ + C₂r²ᵗ; eventual, not monotone |
| Selection, endpoints, curve summary and confirmation unspecified; costs missing | "Selection and confirmation" section; costs and phase table below |
| fixed75 claim contradicts the corrected baseline | Failure mode restated as a hypothesis for `mixed` |

## Computation plan

Follows [computation reporting](computation-reporting.md). Orders are per optimizer
update with batch B unless stated; encoder cost is unchanged from the fan-in runs.

**Declared dimensions.** Code 16; first-layer reader units 64 per reader (48 structure,
16 memory); `g` first layer 40 inputs plus bias, `P` 32 plus bias; second layer 64 × 64;
10 seeds per arm; 8,200 updates per run (6,000 + 2,000 delay + 200 relearning);
rewiring and reset every 100 updates until 4,500; forks 2,000 updates; float32.

| Phase | Time | Storage |
| --- | --- | --- |
| Training update | O(B · (encoder + reader parameters)); readers add O(B · 64 · 41) for `g` and O(B · 64 · 33) for `P` per forward pass, as dense | parameters, Adam moments (2×), rehearsal buffer unchanged |
| Decay | 3,232 multiply-adds per update (`g`: 16 × 41 incoming + 16 × 64 outgoing = 1,680; `P`: 16 × 33 + 16 × 64 = 1,552) | starting values 1,184 floats (4,736 bytes); outgoing targets zero |
| Rewiring (structure rows) | every 100 updates until 4,500: one draw per structure row per operand, O(48) per layer | masks unchanged in size |
| Reset (`mixed-reset`) | same schedule: 16 × 32 draws per layer, 1,024 per event, 45 events | none beyond the moments zeroed in place |
| Evaluation | four passes per measure (full, memory silenced, structure silenced, both) every 500 updates | negligible |
| Coordinate probe | closed-form fit per coordinate, O(16 · N) for N collection codes; 34 s for 80 checkpoints in the analysis | per-run results JSON |
| Disruption test | ten masked evaluations per measure per checkpoint, for each of the eight transfer measures | negligible |
| Decay-only diagnostic | 2,000 decay steps on one frozen copy per run | one extra copy transient |
| Fork | 2,000 updates per continuation | one checkpoint with optimizer state and buffer per seed (about 4.4 MB as float32, estimated at three times the 1.46 MB parameter-only checkpoint) |

**Run accounting.** Updates are counted for every run that is executed, including
selection candidates and confirmation:
- selection run: 80 runs × 8,200 = 656,000 updates (8 arms × 10 seeds; the selection
  candidates are the three decaying arms);
- forks: 4 continuations × 10 seeds × 2,000 = 80,000 updates;
- confirmation: 4 arms × 10 fresh seeds × 8,200 = 328,000 updates;
- optional `sparse64-rigl`: 10 runs × 8,200 = 82,000 updates, plus RigL's gradient
  collection every 100 updates;
- total without the optional arm: 1,064,000 updates.

**Estimates (not measurements).** From the earlier CPU runs, about 0.08 s per update per
worker (480 s per 6,000-update run), so with 6 workers: selection run about 146
minutes, forks about 18, confirmation about 73: **about 4 hours on CPU**. An earlier
draft's 115 minutes scaled the run count but not the run length and was wrong. The
GPU figure is unknown until measured.

**Checkpoints.** Four per run (ends of block 1, block 3, delay, relearning), parameters
only, about 1.5 MB each by the erasure run's size (117 MB for 80): about 480 MB untracked
for 80 runs, plus the 40 fork checkpoints (about 180 MB).

**Selection and tuning cost.** Not zero. Selection ranks the three half-lives on the
selection run (all three fully executed, counted above); confirmation reruns four arms on
fresh seeds (counted above). The 48/16 split, the delay length, the relearning length,
the fork length and the 25% disruption level are fixed and not tuned.

**Device.** The GTX 1650 (4 GB) is the intended device if two checks pass; otherwise the
study runs on CPU. The [benchmark](cpu-gpu-benchmark-v1.md) measured about 5.8× six CPU
threads on a simplified step at batch 270 and says the real gain is smaller and
unmeasured.
- **Needed:** device handling in the runner (models, data tensors and encoder inputs on
  the device; rendering on the CPU), `torch.use_deterministic_algorithms(True)` with
  `CUBLAS_WORKSPACE_CONFIG`, and a one-seed real timing against the CPU runtime.
- **Determinism check:** one arm and seed run twice on the GPU must give identical
  results, and a fork continuation must reproduce a straight run. This replaces the
  earlier runs' exact CPU reproduction.
- **Consequence:** GPU results will not equal the CPU fan-in runs, so comparisons with
  them are approximate. Every arm here is rerun, so the study's contrasts are within one
  device. CPU and GPU arms must not be mixed in one comparison.

**Reporting.** Each run writes a separate runtime JSON: command, revision and dirty
status, device, dtype, library versions, threads and workers, wall and process CPU
seconds by phase (training blocks, delay, relearning, evaluation, probes, forks,
confirmation), peak RSS and GPU peak memory, updates executed. Timing stays out of the
results JSON.

## Open choices for the researchers

- Settled: random rewiring is the default structure wiring, RigL optional;
  GPU if the two checks above pass.
- The split: 48 structure and 16 memory units.
- The half-lives: 2,000, 500 and 125 updates.
- Whether to keep the existing rehearsal buffer (external replay) for comparability with
  the fan-in runs, or drop it so that only the network's own memory holds old blocks.
- Whether to run the optional `sparse64-rigl` arm, and the threshold (0.10) for a
  "large margin".
- The declared protocol values in "Selection and confirmation" and the fork (endpoints,
  selection metric, relearning gain, fork length 2,000, entry margin 0.05, disruption
  level 25%).
