# Limited, decaying memory as pressure to consolidate

Draft, 2026-10-30, revised the same day. Not frozen. The sparse part of the readers is
chosen after the fan-in
[timing run](fan-in-pressure-proposal.md#timing-or-choice-of-the-coordinate-written-before-running).

**Revision.** The first draft modelled memory as an external store of training episodes
with a pixel lookup. An external review found four problems in it: refresh by writing
replayed episodes, stored counts the learner never saw, an ambiguous delay, and no
uniform-replay control. The researchers then set the model of memory: **memory is the
plain, dense network**, not a separate structure. This version replaces the external
store; three of the review's four points no longer apply (see "Review points").

## Question

Can one network answer from memory at first and form transferable structure later, if
its memory is limited and fades?

The researchers' view:
- **Memory is the dense network.** Without sparse reads, a network fits any taught
  pairing, arbitrary ones included. It needs no separate structure.
- **Memory is limited and decaying, and holds recent training more strongly.** To keep
  something long term, the learner has to form structure that can be stored.
- **The learner must stay able to learn again,** both new and old material, by training
  again.
- **Not everything is a concept.** Some things, such as a route, are pure memory.
- **Replay is external or internal.** External replay takes help from an oracle, the
  training data. Internal replay randomly replays the learned state.
- **Start with the readers.** Whether memory is needed elsewhere has to be discovered.

## What is already known

- **Dense readers memorize; sparse readers transfer.** Readers at one code dimension per
  operand, on one shared coordinate, gave large untaught transfer and lost taught
  matching and retention ([fixed-coordinate results](fan-in-pilot-v1-fixed-results.md)).
- **A plain mix loses the transfer.** With 48 of 64 reader units on the shared coordinate
  and 16 dense units (`read-q16-fixed75`), transfer fell back to near the random arm,
  while taught matching and retention stayed near dense. Memory that can answer removes
  the pressure.

So a plain mix fails, and a limited, decaying memory is what the researchers' view adds.

## Design

### Readers with a decaying memory part

The first layer of `g` and of `P` has 64 units each:

- **48 structure units** read one code dimension per operand, with the wiring the timing
  run supports (RigL, or a coordinate fixed gradually). Their weights do not decay.
- **16 memory units** read all 32 code inputs, like dense units. Their weights decay
  every update:
  - **incoming weights and bias decay toward their starting values:**
    w ← w₀ + (w − w₀) · 2^(−1/H);
  - **outgoing weights** (their columns in the second layer) **decay toward zero** at
    the same rate.

What this does:
- **Recent training is held most strongly.** A change made t updates ago has shrunk by
  2^(−t/H); training renews what is still in use.
- **Forgetting is real.** A memory unit that is not renewed goes back to its random
  starting features, and its influence on the answer goes to zero. Decaying only the
  incoming weights would not be enough: the second layer could keep memorizing through
  random features.
- **The learner can learn again.** A unit that has decayed is a random unit, not a dead
  one: its incoming weights are nonzero, so it responds to inputs and its outgoing weights
  get gradient again. Decaying everything toward zero could leave units with no response
  and no gradient, as RigL's zero-initialized connections did.

The later layers of `g` and `P` (64 → 64 → output) do not decay. They read both kinds of
unit.

### Curriculum

The pilots' D-varied+rehearsal, unchanged, so arms compare with the fan-in runs. Then two
new phases, the same in every arm:

- **Delay:** 2,000 updates of context episodes only (taps, collections, comparison). No
  forms at all, and no rehearsal of forms. Memory units holding form-specific answers fade
  during it.
- **Relearning:** 200 updates of English language episodes, as in block 1. This measures
  whether old material can be learned again, and how fast.

The pilots' existing rehearsal buffer is external replay. It is kept in every arm, as
part of the curriculum, during the Chinese and Devanagari blocks only. Internal replay is
stage 2.

### Arms

Fresh seeds, ten per arm.

| Arm | Readers | Half-life H (updates) |
| --- | --- | --- |
| `dense` | all 64 units dense | — |
| `sparse` | all 64 on the shared coordinate | — |
| `mixed` | 48 sparse + 16 dense | ∞ (no decay; the 75% arm) |
| `mixed-h2000` | same | 2,000 (one block) |
| `mixed-h500` | same | 500 |
| `mixed-h125` | same | 125 |

The half-lives run from one block down to a few hundred updates. As H → 0 the memory
units do nothing and the arm becomes `sparse`.

### Stage 2: internal replay (not in this design)

Excite the readers with random codes, take the memory units' contribution to the answer,
and train the structure units to give it, before it fades. This moves what memory holds
into structure without the training data. Deferred until stage 1 shows how decay alone
behaves.

## Measures

- **Over time,** every 500 updates:
  - matching per script, for the current block and earlier ones;
  - comparison on forms; taps → withheld forms.
- **Who answers.** At each evaluation, every matching and comparison measure is also
  scored with the memory units silenced (outgoing weights set to zero), and with the
  structure units silenced. This is the network's equivalent of "answered from memory"
  against "answered from structure".
- **At the end of block 3, after the delay, and after relearning:** the sequence pilot's
  full measures, the arrangement probe and the fan-in code-structure diagnostics.
- **Relearning speed:** English matching after 50, 100 and 200 relearning updates,
  against the same arm's English matching at the end of block 1.
- **Memory diagnostics:** distance of memory-unit weights from their starting values,
  and dead units per part.

## Predictions

Written before any run.

- **Dose-response in H.** As H shortens, transfer moves from the `mixed` level toward
  `sparse`.
- **A middle H gets both.** At H = 500 or 2,000, transfer is close to `sparse`, and
  taught matching on the current block is clearly above `sparse`, answered mostly by the
  memory units.
- **After the delay,** network-wide English matching is lower with shorter H, but what
  remains is carried by the structure units, and is ordered along the shared coordinate.
- **Relearning is fastest in arms with the most structure.** `sparse` and short-H arms
  relearn English faster than `dense`, because the count axis is kept.
- Low confidence on the middle H: the fixed75 result shows that even a quarter of free
  units can take over.

## Decision rules

Margins as in the audit: 0.05 for accuracies, 0.03 for AUROCs. Paired 95% t-interval
rule.

- **Contrasts:** each decaying arm against `sparse` and against `mixed`, on:
  - the eight transfer measures after block 3;
  - taught matching on the current block (Chinese paired at its block);
  - English matching after the delay, and after relearning.
- **Readings:**
  - **Memory and structure both:** some H is equivalent to or better than `sparse` on
    taps → withheld Chinese and English comparison, and better than `sparse` on Chinese
    paired at its block.
  - **Decay only removes memory:** every H that reaches `sparse` on transfer is also at
    `sparse` on taught matching.
  - **Decay is not enough:** no H improves on `mixed` in transfer.

## Prior work and expected limits

Search and read date: 2026-10-30. Read today: the abstracts of Chaudhry et al. (2019),
Khandelwal et al. (2020) and Power et al. (2022). Kumaran et al. (2016) was seen only
through search snippets and secondary summaries. Everything else is from memory, marked
**[familiar]**, and must be read before this is frozen.

- **Fast and slow weights.** Two sets of weights on different timescales, with the fast
  set decaying, is an old idea: Hinton and Plaut (1987), "Using fast weights to deblur old
  memories", and Ba et al. (2016) on fast weights for attention to the recent past
  **[familiar]**. The design is a version of this, with one difference: the fast and slow
  parts differ in connectivity as well as timescale. The slow part is sparse; the fast
  part is dense.
- **Complementary learning systems.** A fast system for specifics and a slow one for
  shared structure (McClelland, McNaughton and O'Reilly 1995) **[familiar]**. The 2016
  update ([Kumaran, Hassabis and McClelland](https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf),
  **[secondary summaries only]**) reportedly adds that knowledge consistent with existing
  structure is integrated faster. Here both systems live in one reader, as the
  researchers specify.
- **Decay toward the starting point.** Pulling weights toward their initial values is
  known as a regularizer for fine-tuning (L2-SP, Li et al. 2018) **[familiar]**. That is a
  penalty on the loss; the decay here is applied directly each update, independent of the
  loss, so it is a forgetting rate rather than a cost.
- **Weight decay and generalization.** Weight decay is credited with the late
  generalization in grokking **[familiar; the abstract of [Power et al. 2022](https://arxiv.org/abs/2201.02177)
  does not mention it]**. Power et al. report generalization well after overfitting on
  small algorithmic datasets **[abstract]**. Both suggest that how long memorized
  solutions persist shapes what is learned.
- **Small replay memories** ([Chaudhry et al. 2019](https://arxiv.org/abs/1902.10486),
  **[abstract]**): a tiny episodic memory of past examples beats specialized
  continual-learning methods and does not hurt generalization. This is external replay,
  the pilots' existing rehearsal; it is held constant here.
- **Non-parametric memory beside a network** ([Khandelwal et al. 2020](https://arxiv.org/abs/1911.00172),
  **[abstract]**). The first draft's external store followed this pattern. The researchers
  set memory inside the network instead.
- **Generative replay** (Shin et al. 2017, [continuous-learning notes](continuous-learning-notes.md#two-different-uses-of-internally-generated-experience)).
  Stage 2's internal replay belongs to this family. It fits the [foundations'](foundations.md)
  commitment to replay with no store of experience.

**Known failure modes.**
- **The later layers memorize.** If the non-decaying second and third layers can store
  specifics through the structure units, memory never fades. The `sparse` arm's taught
  costs suggest they cannot do much of this.
- **Decay too slow or too fast for the curriculum.** The H range spans one block down to a
  few hundred updates.
- **Units drift away from their starting values in a way decay cannot track.** Adam's
  moments are not reset by decay; the decay is applied after the optimizer step.
- **Silencing measures are approximate.** Silencing one part changes the inputs the later
  layers were trained on. Reported as a diagnostic, not as an exact split.

**What this adds.** Whether a decaying dense part beside a sparse part gives taught
answers from memory while keeping the sparse part's transfer, in one network, and how
that depends on the half-life. Fast weights predict the forgetting. Nothing read predicts
whether the split recovers transfer.

## Review points

The external review was of the first draft.

| Point | Status |
| --- | --- |
| Writing from mixed batches refreshes replayed episodes | No store; does not apply |
| Stored episodes contain counts the learner never saw | No store; does not apply. The network only learns from what it is shown |
| The delay is ambiguous, and fading does not empty a store | Delay now defined: no forms and no rehearsal of forms in any arm. Decay toward the start really removes what is not renewed |
| Add a uniform-replay FIFO control | External replay is held constant across arms, so no replay control is needed here |

## Computation plan

- **Decay:** one multiply-add per decaying weight per update: (32 + 1) × 16 incoming and
  16 × 64 outgoing per reader, about 1,600 per reader. Negligible. Starting values are
  stored once: the same count of extra floats.
- **Silencing measures:** two extra evaluation passes per evaluation point.
- **Runs:** 60 (6 arms × 10 seeds), each 8,200 updates (6,000 + 2,000 delay + 200
  relearning) instead of 6,000, plus evaluation every 500 updates.
  - On CPU (6 workers × 2 threads), about 480 s per run: about 80 minutes.
  - The [CPU against GPU benchmark](cpu-gpu-benchmark-v1.md) suggests the GPU could be
    several times faster. It would need device handling, deterministic settings and a
    real-pilot timing first. This study uses fresh seeds and no CPU comparison arm, so it
    could run on the GPU.
- **Selection and tuning:** none. The half-lives, the 48/16 split, the delay and the
  relearning length are fixed here.

## Open choices for the researchers

- The split: 48 structure and 16 memory units, matching the 75% arm.
- The half-lives: 2,000, 500 and 125 updates.
- Whether the memory units' biases decay toward their starting values (as here) or stay
  free.
- Whether to keep the existing rehearsal buffer (external replay) for comparability with
  the fan-in runs, or drop it so that only the network's own memory holds old blocks.
