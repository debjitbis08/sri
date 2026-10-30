# Limited, decaying memory as pressure to consolidate

Draft, 2026-10-30. Not frozen. Written at the researchers' request while the fan-in
[timing run](fan-in-pressure-proposal.md#timing-or-choice-of-the-coordinate-written-before-running)
was running. The choice of reader depends on that run (see "Arms").

## Question

Can a learner answer from memory at first, and form transferable structure later, if
memory is limited and fades?

The researchers' view:
- Memory is **limited and decaying, biased to recent items**.
- To remember something long term, the learner has to form better structure that can be
  stored. The memory's limits are themselves pressure, in the same sense as the fan-in
  limit: freedom the network keeps is freedom it uses.
- **Not everything is a concept.** Some things, such as a route from A to B, are pure
  memory.
- **Replay from memory** is a way to form reusable concepts.

The number task has both kinds already:
- **Memory:** that 五 means five is an arbitrary link between a shape and a count.
- **Concept:** once 五 is linked to five, its order among other numbers and its match to
  five taps should follow from shared structure. That is the transfer the project
  measures.

## What already exists

The pilots' D-varied+rehearsal arm replays from a bounded buffer. After each language
block, 500 of that block's episodes are kept for good. In later blocks, 4 of every 16
episodes in a batch are drawn uniformly from the buffers of earlier languages. The buffer
never fades and holds nothing from the current block. It is the reference here.

Nothing in the pilots answers from memory: every answer comes from `g`.

## Design

### The memory

A store of recent training episodes, separate from the network.

- **Writing.** After each update, one episode of that batch, drawn at random, is written
  with strength 1.
- **Fading.** Strength halves every H = 750 updates.
- **Limit.** At most K = 1,500 episodes. When full, the weakest entry is dropped. One
  entry is written per update, so the limit binds at an age of 1,500 updates (strength
  0.25).
- **Replay.** In every block, 4 of each batch's 16 episodes are drawn from the store with
  probability proportional to strength. This is the same replay share as the existing
  rehearsal, now including the current block and biased to recent items.
- **No refresh in this study.** Replaying an entry does not restore its strength. Refresh
  rules (for example, refresh only items the network still gets wrong) are a later
  question.

With these numbers, English (block 1, updates 0–2,000) leaves the store during the
Chinese block. After that, English is kept only by what the network has consolidated.

### Answering from memory

At test, for matching questions (a form against nine collections):
- The memory looks for the nearest stored form image of the same script, by pixel
  distance.
- If that distance is below τ, the memory answers with the stored count.
- τ is fixed from the training split before the run: the 95th percentile of
  nearest-neighbour distances between different renderings of the same form.
- Otherwise the network answers.

The memory never changes the code and has no loss. Three accuracies are reported for each
matching measure: network alone, memory alone (where it answers), and combined. Untaught
compositions (taps → form, comparison on forms) are answered by the network only, since
memory holds no such pairs.

This keeps the memory from taking away the pressure on the code. The network trains on
its own loss on every item, whatever memory would have answered.

### A delay period

After the three language blocks, 2,000 updates of context episodes only: taps,
collections and comparison, with no forms. All forms leave the store during this period.
Testing after it measures what the network keeps on its own.

### Arms

Fresh seeds, ten per arm, D-varied contexts, the sequence pilot's network.

| Memory | Dense readers | Fan-in readers |
| --- | --- | --- |
| None (no replay) | `none-dense` | `none-reader` |
| Persistent buffer (the existing rehearsal) | `buffer-dense` | `buffer-reader` |
| Limited, decaying store | `decay-dense` | `decay-reader` |

- **Fan-in readers:** one dimension per operand, with whichever wiring the timing run
  supports. RigL by default. If a gradual fixed coordinate matches RigL, the fixed version
  is simpler and has no dead units.
- **The question for each reader:** does the decaying store, against the persistent
  buffer, shift what the network keeps toward shared structure? The prediction is that it
  matters more with fan-in readers, which already push count onto one axis.

### Pure memory (stage 2, not in this design)

Arbitrary pairings with no shared structure, the "routes". For example, nine shapes from
an unused script, each paired with a collection by a random permutation of counts. These
should depend on memory and rehearsal only, and fade after the delay unless kept by the
network's specific capacity. They need new data and a new evaluation. Deferred until
stage 1 shows whether the store behaves as intended.

## Measures

- **Over time,** every 500 updates:
  - matching per script: network, memory and combined;
  - comparison on forms;
  - taps → withheld forms.
- **At the end of block 3 and after the delay:** the sequence pilot's full measures, the
  arrangement probe and the code-structure diagnostics of the fan-in analysis.
- **Memory diagnostics:** the store's makeup by script over time; replay counts per
  script; how often the memory answers, and its accuracy when it does.

## Predictions

Written before any run.

- **Combined taught accuracy is high early in decaying-store arms.** Memory answers
  current-block forms from the first few hundred updates.
- **Network retention after the delay:** persistent buffer > decaying store > none for
  English, since the buffer keeps rehearsing English.
- **Transfer:** decaying store ≥ persistent buffer with fan-in readers, by a small margin.
  With dense readers, no difference. Low confidence. The researchers' view predicts a
  clearer gain.
- **Interaction:** the drop in English retention from buffer to decaying store is smaller
  with fan-in readers. English forms kept on the shared count axis survive without
  rehearsal.

## Decision rules

Margins as in the audit: 0.05 for accuracies, 0.03 for AUROCs. Paired 95% t-interval
rule.

- **Primary contrasts:** decaying store − persistent buffer, within each reader, on:
  - the eight transfer measures after block 3 and after the delay;
  - network-only retention of English and Chinese after the delay.
- **Interaction:** (decay − buffer with fan-in readers) − (decay − buffer with dense
  readers), on the same measures.
- **Readings:**
  - **Limited memory helps structure** if decaying is better than buffer on at least two
    transfer measures and worse on none, for either reader.
  - **Limited memory only forgets** if retention falls and no transfer measure improves.
  - **Structure protects from fading** if the interaction is positive on English
    retention after the delay.

## Prior work and expected limits

Search and read date: 2026-10-30. Read today: the abstracts of Chaudhry et al. (2019),
Khandelwal et al. (2020) and Power et al. (2022). Kumaran et al. (2016) was seen only
through search snippets and secondary summaries. Everything else is from memory, marked
**[familiar]**, and must be read before this is frozen.

- **Complementary learning systems.** A fast store of specific experiences and a slow
  system that learns shared structure through replay (McClelland, McNaughton and O'Reilly
  1995) **[familiar]**. The design is this split.
  - The 2016 update ([Kumaran, Hassabis and McClelland](https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf),
    **[secondary summaries only]**) reportedly adds two points:
    - replay of several memories together can reveal links between them;
    - new information that fits existing knowledge is integrated faster.
  - Predicts that forms consistent with the count axis consolidate faster than arbitrary
    ones. That is the stage 2 contrast.
- **Small replay memories.** [Chaudhry et al. 2019](https://arxiv.org/abs/1902.10486)
  **[abstract]**:
  - training jointly on new data and a tiny episodic memory, one example per class,
    beats specialized continual-learning methods on four benchmarks;
  - it does not hurt generalization; the abstract reports gains of 7–17%.

  Predicts that even a small, decaying store keeps a good deal. Our existing bounded
  buffer is this baseline.
- **Experience replay and prioritized replay** (Lin 1992; Schaul et al. 2016)
  **[familiar]**. Sampling by strength is a prioritization rule, by recency rather than by
  error. A recency-only buffer is known to forget older tasks **[familiar]**; that is the
  intended pressure here, not a flaw to fix.
- **Non-parametric memory beside a network.** [Khandelwal et al. 2020](https://arxiv.org/abs/1911.00172)
  **[abstract]** interpolate a language model with nearest-neighbour retrieval from a
  datastore. This helps most on rare patterns such as facts, without retraining. It is
  the closest established form of "answer from memory". Their datastore keys are the
  model's own embeddings; ours are raw pixels, so that memory does not depend on, or
  shape, the code.
- **Forgetting curves.** Memory strength falls with time and is restored by use
  (Ebbinghaus; Anderson and Schooler 1991, base-level activation in ACT-R)
  **[familiar]**. Human forgetting is closer to a power law; exponential decay is used here
  for simplicity.
- **Generalization after memorization.** [Power et al. 2022](https://arxiv.org/abs/2201.02177)
  **[abstract]**: on small algorithmic datasets, networks can move from chance to perfect
  generalization well after overfitting, and smaller datasets need more optimization. This
  suggests the delay period and the length of training matter for when structure appears.
- **Generative replay** (Shin et al. 2017, [read in the continuous-learning notes](continuous-learning-notes.md#two-different-uses-of-internally-generated-experience)).
  The family the [foundations](foundations.md) commit to: replay generated by the learned
  system, with no store of experience.

**Relation to the foundations.**
- They commit to "no historical experience/replay store" for dreaming, and to a
  structurally plastic short-term memory satellite.
- The store here is bounded, fades, and holds only recent experience. It is closer to the
  short-term memory than to a historical archive.
- It is still a store of raw episodes. Whether this is acceptable as a step, or must later
  be replaced by generated replay, is the researchers' call.

**Known failure modes.**
- **Memory hides a weak network.** Avoided by reporting network-only accuracy everywhere.
- **Replay amount differs between arms.** Matched: 4 of 16 episodes in every replay arm.
  The no-replay arm trains on 16 new episodes instead.
- **τ too loose.** Memory then answers withheld forms with a neighbour's count. Checked
  by how often memory answers withheld forms: it should almost never answer them.
- **Fan-in readers' own costs.** Taught matching and retention fall under fan-in readers.
  Memory answering should recover taught matching in the combined score. Retention after
  the delay has no memory to help.

**What this adds.** Whether, for this network, a fading store instead of a persistent
buffer shifts what is kept toward transferable structure, and whether readers that force
a shared axis protect forms from fading. Prior work predicts the forgetting; it does not
predict the effect on transfer.

## Computation plan

- **Store:** K = 1,500 episodes, as indices into the rendered data plus counts and
  strengths. Under 1 MB per run. Replay sampling is O(K) per update. Decay is one
  multiply per update: strength = 2^(−age/H), computed on demand.
- **Memory answering:** pixel nearest neighbour over at most 1,500 stored forms of
  28 × 28. O(K × 784) per query; under a second per evaluation.
- **Runs:** 60 (6 arms × 10 seeds), each 8,000 updates (6,000 plus a 2,000-update delay)
  instead of 6,000, plus evaluation every 500 updates.
  - On CPU (6 workers × 2 threads), about 470 s per run: about 80 minutes.
  - The [CPU against GPU benchmark](cpu-gpu-benchmark-v1.md) suggests the GPU could be
    several times faster. Moving would need device handling, deterministic settings and a
    real-pilot timing first. This experiment uses fresh seeds and no CPU comparison arm,
    so it could run on the GPU.
- **Selection and tuning:** none. K, H, τ's rule, replay share and delay length are fixed
  here.

## Open choices for the researchers

- The numbers: K = 1,500 entries, H = 750 updates, one write per update, replay share 4
  of 16, delay of 2,000 updates.
- Whether replay should refresh strength, and on what condition.
- Whether replay should play back stored inputs (as here) or only codes.
- Whether a store of raw episodes is acceptable under the foundations' no-replay-store
  commitment.
- Whether to run on the GPU after porting.
