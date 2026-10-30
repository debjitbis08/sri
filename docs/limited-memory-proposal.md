# Limited, decaying memory as pressure to consolidate

Draft, 2026-10-30, revised twice the same day. Not frozen.

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
  coordinate nor gradual timing reproduces RigL, so the structure part uses **RigL**
  wiring.

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
  concentration and repeated rewiring are untested candidates.

## Design

### Readers with a decaying memory part

The first layer of `g` (40 inputs: two 16-dimension codes and an 8-dimension relation
token) and of `P` (32 inputs: two codes) has 64 units each.

- **48 structure units** read one code dimension per operand, with RigL wiring as in the
  fan-in runs. `g`'s relation-token columns stay fully connected for these units, as
  before. Their weights do not decay.
  - **RigL runs on the structure rows only.** The current helper assumes the same number
    of connections in every row (`rigl_step` reads it from row 0); it must take the
    structure rows as a group and skip the dense rows.
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
- **Outgoing decay removes influence.** On fixed inputs and with fixed downstream
  layers, the memory part's contribution shrinks at least as fast as r^t, even though
  incoming weights return to nonzero random values. Incoming decay alone would leave
  random features that the next layer could still use.
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

### Arms

Fresh seeds, ten per arm.

| Arm | First layer of each reader | Half-life H (updates) |
| --- | --- | --- |
| `dense` | 64 dense units | — |
| `sparse64` | 64 RigL units | — |
| `sparse48` | 48 RigL units, no memory part | — |
| `mixed` | 48 RigL + 16 dense | ∞ (no decay; a new baseline) |
| `mixed-h2000` | same | 2,000 (one block) |
| `mixed-h500` | same | 500 |
| `mixed-h125` | same | 125 |

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
    error, the best it can do is the memory's average answer given the one coordinate it
    reads. Anything the memory knows that depends on other coordinates is lost.
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
  curve from each arm's own accuracy at the start of relearning.
- **At the end of block 3, after the delay, and after relearning:** the sequence pilot's
  full measures, the arrangement probe and the fan-in code-structure diagnostics.

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
- **Relearning** is faster from structure-rich arms, on the curve measure, for the count
  ordering. A lost link between a specific shape and its count cannot be restored by
  structure without new examples.

## Decision rules

Margins as in the audit: 0.05 for accuracies, 0.03 for AUROCs. Paired 95% t-interval
rule.

- **Contrasts:**
  - each decaying arm against `sparse48`, `sparse64` and `mixed`;
  - `mixed` against `dense` and `sparse64`.

  On: the eight transfer measures after block 3; Chinese paired at its block;
  structure-only accuracy at the end of block 3; English matching after the delay; the
  relearning curve.
- **Readings:**
  - **Memory and structure both:** some H is equivalent to or better than `sparse48` on
    taps → withheld Chinese and English comparison, and better than `sparse48` on Chinese
    paired at its block. Any such H must be confirmed on fresh seeds.
  - **Handoff:** structure-only accuracy at the end of block 3 is better than in `mixed`
    for some H, with the memory part's dependence lower.
  - **Erasure without handoff:** the memory part's dependence falls with no rise in
    structure-only accuracy.
  - **Gain compensation:** memory-column norms fall while dependence on the memory part
    does not.
  - **Inconclusive is not equivalent.**

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

**Known failure modes.**
- **Gain compensation** by the non-decaying layers. Measured by dependence and norms.
- **Renewal during the delay** by context gradients and Adam's moments. Separated by the
  decay-only diagnostic.
- **Units that do not reactivate.** Measured directly.
- **Incomplete handoff within training.** Expected from the timescale; not read as
  failure.
- **Silencing is approximate:** it changes the inputs the later layers were trained on.
  Reported as dependence with interaction terms.

**What this adds.** Whether a decaying dense part beside a sparse RigL part gives taught
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

## Computation plan

- **Decay:** in `g`, 16 × 41 incoming (40 weights and a bias) plus 16 × 64 outgoing:
  1,680 parameters. In `P`, 16 × 33 plus 16 × 64: 1,552. 3,232 in total, one
  multiply-add each per update. Saved starting values: 1,184 floats (4,736 bytes);
  outgoing targets are zero.
- **Evaluation:** four passes per measure (full, memory silenced, structure silenced,
  both), every 500 updates.
- **Decay-only diagnostic:** 2,000 steps of decay on one frozen copy per run;
  negligible.
- **Runs:** 70 (7 arms × 10 seeds), each 8,200 updates (6,000 + 2,000 delay + 200
  relearning) instead of 6,000, plus evaluation.
  - On CPU (6 workers × 2 threads), about 480 s per run plus evaluation: about 100
    minutes.
  - The [CPU against GPU benchmark](cpu-gpu-benchmark-v1.md) suggests the GPU could be
    several times faster. It would need device handling, deterministic settings and a
    real-pilot timing first. This study uses fresh seeds and no CPU comparison arm, so it
    could run on the GPU.
- **Selection and tuning:** none. The half-lives, the 48/16 split, the delay and the
  relearning length are fixed here.

## Open choices for the researchers

- The split: 48 structure and 16 memory units.
- The half-lives: 2,000, 500 and 125 updates.
- Whether to keep the existing rehearsal buffer (external replay) for comparability with
  the fan-in runs, or drop it so that only the network's own memory holds old blocks.
- CPU or GPU for this study.
