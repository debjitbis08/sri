# The moving-representation problem

Discussion 2026-11-05. User problem statement, assessment, and two rounds of review that corrected the drafts
(recorded in the last section). No contract, nothing run. Local evidence is **[read: local results]** from the
documents linked; literature is **[familiar]**, recalled and not re-read; the rest is **[derived]** or a
**[proposal]**.

## The problem, as the user stated it

Senses encode inputs into codes; a relational core learns relations between codes from examples; a memory stores
experiences under learned addresses. Everything downstream is learned in terms of codes, yet the networks that
produce the codes keep learning, and each new thing taught moves the codes of things learned earlier. The same
failure has returned in every part of the system:

- **Memory.** Stored keys stop matching the current code of the same input, so stored associations become
  unreachable although nothing was overwritten ([moving host](shared-core-sparse-memory-v1-moving-host-pilot.md),
  [natural images](natural-memory-v1-pilot.md)).
- **Relations.** After a second script is taught, the first script's numerals drift and lose their alignment to the
  reference collections. Order within the drifted script survives one block back, not two
  ([two-script reads](two-script-reads-v1.md), [relation codes check](relation-codes-check-v1.md)).
- **Self-checking.** A coherently shifted code is wrong consistently: every relation reads the same shifted code, so
  the outputs agree with each other and checks among the system's own outputs cannot see it
  ([consistency critic](consistency-critic-v1.md)).

Measured directly, the movement accumulates; it is not jitter that averaging removes, and a linear map fitted on far
inputs does not predict it ([drift structure](sense-drift-structure-v1.md)). Tried: slowly averaged encoder weights
(do not keep memories reachable); a learned linear transport from old codes to new, and semantic drift compensation
(little gain, below freezing); rebuilding addresses from stored cue traces (works, at a cost in units); a frozen copy
of the encoder per stretch of time (keeps everything readable; cost grows with time, and copies share no frame).

The question as first put: how can a continually learning system keep a stable frame of reference, so that
memories, relations and alignments stay valid, and detect when they have gone stale, without storing raw inputs and
without freezing the encoders?

**Restated after review.** "Without freezing the encoders" conflicts with the earlier decision to accept frozen
senses ([sense integration plan](sense-integration-plan.md)). In the current design the senses are frozen and
memory keys on them. The encoder that must keep learning is the core's object encoder, and that is where the frame
has to hold. So: how does the core keep a stable frame while its object encoder learns new kinds, how are stale
readings of it detected, and what outside evidence resolves the cases that internal checks cannot?

## Where the movement happens in the current design

**[read: local results]**

- **Senses are frozen.** The relation studies run on cached DINOv2-S and TrOCR-S codes
  ([relation codes check](relation-codes-check-v1.md), [shared output contract](shared-output-v1-contract.md)).
- **Memory keyed on frozen sense codes does not drift.** In the kept-centre pilot, 0 of 5,400 query reads changed
  score, unit, value decoding or acceptance when centres opened or the gate refitted
  ([sense stable pilot](sense-stable-v1-pilot.md)). The remaining loss there is interference, 48 of 5,400 reads, and
  it is small.
- **What moves is the core's object encoder**, as new scripts are taught, and with it the reading of stored values
  through the core. In stage 1 of the memory-core pilot, stored taps answers read through the core fell from 0.96 at
  the delay to 0.72 at relearning ([memory-core pilot](sense-memory-core-v1-pilot.md)).

The "moving host" and natural-image memory pilots describe the earlier design, where the sense itself learned.
Their lesson, that keys through a learning encoder go stale, is why the senses were frozen. The problem now sits
in the core.

## Assessment

### Two problems under one name

**Coherent relabelling.** **[derived, qualified by local results]** A shift that moves each item onto another valid
item's place, consistently across readers, leaves every internal relation agreeing with every other, and a check of
the system against itself cannot see it. That is the blind spot. It is not all drift: the readers are fixed
functions, so an arbitrary movement does break agreement, and the local results show it. The critic catches `eq`
errors on the count base and `more` errors on the scripts ([consistency critic](consistency-critic-v1.md)). A partial
shift, where two forms land on one count, breaks one-to-one matching, and a check within the script can detect it
(the collision check, **[proposal]**, from the discussion of why `eq` is hard to check; first recorded here).

**Unconstrained movement of kinds absent from the loss.** **[derived from local results]** An input kind that is not
in the current loss has nothing holding its code in place; its motion is a side effect of updates aimed at other
kinds. How much the present stream says about an absent kind depends on how near the present inputs are to it. In
the drift measurement a map fitted on the same family predicts most of a block's movement (medians 0.52–0.95); after
a real change a map fitted on background glyphs predicts 0.52–0.81 of the cue symbols' movement, and 0.79–0.86 on
the encoder trained from scratch; a map fitted on natural images or noise predicts nothing or adds movement. So
present inputs that resemble the absent kind carry partial information, and far inputs carry none. The correction
for a far kind needs evidence about that kind.

Consequence: a remedy can live in three places.

1. **Stop the movement** of absent kinds. Frozen copies are the maximal-cost form. The cheap end is penalties on the
   encoder, and they are not alike. Weight-importance penalties (EWC, synaptic intelligence) act on weights and can
   pin an absent kind's directions. Output distillation (Learning without Forgetting) preserves outputs on present
   data only, so by the argument above it protects an absent kind only where present data shares its features. Not
   run in this program; the [retention remedies contract](retention-remedies-v1-contract.md) listed them as not
   included.
2. **Keep evidence** about the absent kind. Trace rebuilding and latent replay are this family. They work, at a
   memory cost, and that cost is the price of the information. Under
   [frozen-code replay is rehearsal](sense-memory-replay-v1-design.md) this is allowed but is rehearsal and must be
   matched against a plain buffer.
3. **Make consumers independent of the coordinates that move.** Partly tried: see the next section.

Semantic drift compensation helped a little and stayed below freezing in the natural-image pilot. That fits: its
best case is a stream that still shows the stored kind, and it has nothing to say once the kind is absent.

### What is lost: not resolution in the codes, but the reading against collections

**[read: local results]**

- **Cross-script comparison is graded by distance**: 0.97 for counts three or more apart and 0.67 for adjacent counts,
  while collections stay at 0.99 even when adjacent ([two-script reads](two-script-reads-v1.md)).
- **Order survives one block back and not two.** Chinese, taught in block 2, keeps its rank among its own forms at
  block 3 (ρ 0.87). English, taught in block 1, had ρ 0.94 at block 1 and 0.20 at block 3
  ([relation codes check](relation-codes-check-v1.md), finding 3). So the survival of order is a one-block effect,
  not a property of the system.
- **Resolution is in the codes.** Chinese core codes at block 3 are 0.99 linearly separable by glyph class, given
  Chinese labels ([equality diagnostics v2](equality-diagnostics-v2.md)). What is lost is the reading against
  collections: comparison through the pair features, and alignment to the collection frame. But the same probe
  reaches 0.84–0.87 on Thai, which was never taught as numbers, so the separability comes mostly from the frozen
  senses, not from anything the teaching left.
- **An unsupervised per-kind affine registration is already in the system and half helps.** Context normalization
  standardizes codes within each kind (`context_norm`, `src/relation_codes.py:24`; Webb et al. 2020). The relation
  codes check scores it half: `two-ctx` beats `two` (0.62 against 0.50), `count-ctx` is worse than `count` (0.46
  against 0.69), and plain `count` beats both. Its own reading of what remains: normalization removes a per-script
  shift and scale, and the rest of the bend is not affine.

**[derived]** So the placement-versus-blur question of the earlier draft is mostly answered. With enough labels a
small per-kind map restores adjacent resolution, because the codes still separate the classes. The informative
question is a savings one: how many labelled pairs a small per-kind map needs for Chinese, which was taught and
drifted, against Thai, which was never taught.

- Fewer for Chinese: the old teaching left usable structure (order, one block back).
- The same: nothing of the teaching survives beyond glyph separability from the frozen senses.

The map's input matters. Fitted on the frozen sense codes it stays valid; fitted on the drifted core codes it holds
only while the core holds still, and must be re-fitted after every further block. Both should be run.

### The frame in the core, with its costs

**[derived]** The proposal: keep the core's object encoder slow or fixed, and give each kind a small registration
from its frozen sense codes into it. Relations are learned on the frame. When a kind is taught, the cost is fitting
one registration from a few confirmations, not relearning the relation. Two known costs:

- **It is per-kind parameter isolation, the adapter family,** and it has to know which kind an input is before
  choosing a registration. Routing by frozen sense code is partly tested, not untested. Label-free centres read like
  told ones, and a batch-shift rule opened about one centre per kind on streams where kinds arrive in runs
  ([sense adapt pilot](sense-adapt-v1-pilot.md)). But no growth rule matches one centre per kind on all four stress
  streams: batch-mean rules fail on interleaved and gradual streams, and the rule that handles gradual change opens
  extra centres for overlapping kinds ([sense growth pilot](sense-growth-v1-pilot.md)). That is the known cost of the
  adapter family here, and it is the domain-routing problem raised in the hypothesis that
  [glyphs may need a separate system](general-frozen-encoder.md).
- **A frame fixed early may lack capacity** for distinctions later kinds need, and that would show as a reading
  failure again, now one the codes could not repair.

**[familiar]** The representational-drift literature supports the split between a stable readout and moving
features. Sensory and association cortex drift over days to weeks while behaviour stays stable (Driscoll et al.
2017; Schoonover et al. 2021; Deitch et al. 2021; Marks and Goard 2021), and the drift concentrates in directions
orthogonal to a stable readout (Rule et al. 2020). Ratzon, Derdikman and Barak 2024 model drift as noise walking
along flat directions of the loss: directions the loss constrains are pinned, flat directions wander. **[derived]**
Far absent kinds are flat directions, which is why they wander. The implication is to keep absent kinds in the loss
cheaply, by replay of codes or by a weight-importance penalty, rather than to stop learning.

### Detection, and where its limit is

- **A single unmatched code does not say which it is.** **[derived]** One code that matches nothing well could be
  novel or stale. The detector for one item should report *unmatched*, not *stale*, and outside evidence resolves
  it: a second sense, rehearsal or asking.
- **Drift and novelty separate at the population level.** **[derived]** One unmatched code looks like novelty. Many
  previously well-matched readings losing margin together is drift. This signal is internal and cheap. Since memory
  keys on frozen senses no longer drift, the place to measure it is the core's readings: the margins of `eq`, `more`
  and matching on stored or recurring items, tracked across blocks. The accuracies that motivated the earlier draft
  (stale nearest-neighbour reads of 0.87, 0.37 and 0.10 in the drift measurement) are label-scored, not internal
  margins; a margin collapse visible from inside has not been measured.
- **Detection within a script.** **[proposal]** The collision check, two different forms judged equal to one count,
  is the cheap detector for a partial shift.
- **Stored facts do not move; stored codes do.** **[derived]** If memory stores the outcome of a cross-sense relation
  as a value, that value does not move with the core, and a fresh read that disagrees with it is a staleness signal
  with no raw input. Two qualifiers. Under [values are sensory traces](sense-memory-core-v1-design.md), the stored
  value must be a frozen-sense trace, such as the collection's sense code; a core code stored as the value would move
  with the encoder. And the memory must be reached through an address that did not move, which the frozen-sense keys
  now give.

### Frozen copies and a shared frame

**[derived]** Copies share no frame, and a chain of small maps fitted between consecutive copies at each boundary
would register only the kinds present at that boundary, by the same argument as above. It helps kinds that recur or
have near relatives in the stream and not far absent ones. Distilling copies into one is consolidation, deferred.

## Candidate tests, in order

1. **Savings test on the existing snapshots.** Fit a small map per kind from k labelled pairs per count into the
   core's frame, for Chinese against Thai, from the drifted core codes and from the frozen sense codes separately.
   Read adjacent-count comparison and alignment to collections as k grows. Control: a dense adapter with the same
   parameter count. Fewer labels for Chinese means the teaching left usable structure; the same means only glyph
   separability survives. No training of the core.
2. **EWC against LwF on the core's object encoder**, since that is what moves, with opposite predictions: the
   weight penalty should protect the absent script, output distillation on present data should not, unless the
   present scripts share its features.
3. **The population-level drift signal on the core's readings**: margins of `eq`, `more` and matching on recurring
   items across blocks, many falling together against one unmatched, with the collision check within a script.
   Memory keys are excluded, since they no longer drift.
4. **Latent replay at a matched memory budget**, the evidence-keeping arm, against a plain buffer of the same bytes.

Each candidate must state how it scales with the number of kinds and with time (registration: parameters per kind
plus a router, whose cost the growth pilot measured; penalty: one importance vector; detector: no growth; replay:
bytes per kind).

## Prior work to read before a contract

All **[familiar]**, recalled, not re-read. Read before any contract names them.

- Rule, Loback, Raman, Driscoll, Harvey and O'Leary 2020, *Stable task information from an unstable neural
  population*, eLife.
- Driscoll et al. 2017, *Dynamic reorganization of neuronal activity patterns in parietal cortex*, Cell.
- Schoonover et al. 2021, *Representational drift in primary olfactory cortex*, Nature.
- Deitch, Rubin and Ziv 2021, *Representational drift in the mouse visual cortex*, Current Biology.
- Marks and Goard 2021, *Stimulus-dependent representational drift in primary visual cortex*, Nature Communications.
- Ratzon, Derdikman and Barak 2024, *Representational drift as a result of implicit regularization*, eLife.
- Masset, Qin and Zavatone-Veth 2022, *Drifting neuronal representations: bug or feature?*, Biological Cybernetics.
- Rule and O'Leary 2022, *Self-healing codes*, PNAS (already in [reading notes](natural-memory-prior-work.md)).
- Kirkpatrick et al. 2017, elastic weight consolidation; Zenke, Poole and Ganguli 2017, synaptic intelligence;
  Li and Hoiem 2016, Learning without Forgetting.
- Rebuffi, Bilen and Vedaldi 2017, residual adapters; Aljundi et al. 2017, Expert Gate (in reading notes): the
  adapter family and its routing cost.
- Shen et al. 2020, backward-compatible representation learning (in [reading notes](natural-memory-prior-work.md)).
- Webb et al. 2020, context normalization (already cited in the relation codes check).
- Conneau et al. 2018, *Word translation without parallel data*, Procrustes alignment of embedding spaces, as the
  model for per-kind registration; Mikolov, Le and Sutskever 2013 for the linear-map assumption and its limits.

## Review history (2026-11-05)

**First review.** The blind spot is coherent relabelling, not all drift. Near present inputs predict part of an
absent kind's movement; the argument holds for far kinds. Order survives coarsely, with resolution lost too.
Coordinate-independent consumers were partly tried as context normalization. The margin figures were label-scored
accuracies, not internal margins. In the relation studies the senses are frozen and the core's object encoder moves.
LwF and EWC carry different predictions; the chain of local maps between frozen copies is limited to kinds present at
the boundary; stored facts must be frozen-sense traces. Added the population-level drift signal and the collision
check.

**Second review.** Context normalization is scored half, not better, and the relation codes check's own reason for
the remainder is that the rest of the bend is not affine. The collision check is a proposal first recorded here, not
a local result. Order survives one block back (Chinese ρ 0.87) and not two (English ρ 0.94 → 0.20). Resolution
survives in the codes (0.99 separable with Chinese labels), but so does Thai's (0.84–0.87), so separability comes from
the frozen senses; the placement-versus-blur test is replaced by a savings test, Chinese against Thai, from core and
from sense codes. Memory keyed on frozen senses does not drift (0 of 5,400 reads); what fails is reading values
through the moving core (0.96 → 0.72), so the problem sits in the core, and the question is restated to say so.
Routing by frozen sense code is partly tested in the adapt and growth pilots, with known failures on interleaved and
gradual streams. Test 2 is moved onto the core's object encoder and test 3 narrowed to the core's readings.
