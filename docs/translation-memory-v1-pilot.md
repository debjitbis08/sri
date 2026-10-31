# Translation memory v1: pilot (seeds 299 and 298)

Status: **pilot, 2026-10-31**. Design: [draft](translation-memory-v1-draft.md). Not evidence about memory; it decides whether the design can be frozen. Sections run in the order done. First outcome: **no memory configuration passes the storage gate on the host's code.** Latest: with a frozen key encoder the perception gate passes on three seeds, but under the [binding revision](translation-memory-v1-draft.md) no one-step memory comes within 0.02 of the explicit store; see *Where this leaves the design*.

## Host stage

`main` host accuracy on its current batch at the batch's end (150-way, evaluation renderings), batches 1–6:

| arm, seed | batch 1 → 6 |
|---|---|
| mixed, 299 | 0.47, 0.69, 0.40, 0.61, 0.71, 0.84 |
| mixed, 298 | 0.53, 0.65, 0.78, 0.82, 0.86, 0.93 |
| sparse48, 299 | 0.31, 0.65, 0.72, 0.66, 0.80, 0.62 |
| sparse48, 298 | 0.17, 0.30, 0.40, 0.32, 0.36, 0.20 |

`frozen-encoder` (image path frozen after batch 1; frozen state checked) learns later batches at 0.00–0.33. Freezing the path mostly blocks acquisition, so this diagnostic will bound drift only loosely, as the third review anticipated.

## Selection

- **Map regularization:** λ = 0.01 (held-out anchor error 0.179 s²; 0.001: 0.186, 0.1: 0.197, 1: 0.219). On the confirmation seed 0.1 was slightly lower (0.180 against 0.189).
- **Storage gate (95% recovery of the written target on exact cues after all 120 writes, 16-step read):** no configuration passes. Best: 1,024 units, 41 active: 0.92 (`mixed`), 0.83 (`sparse48`). The smallest, 256 units with 5 active: 0.29 and 0.34. Recurrence lowers recovery in every configuration: at 1,024 units, 41 active, from one step to sixteen 0.94 → 0.92 (`mixed`) and 0.88 → 0.83 (`sparse48`). (31–79% of reads end in a cycle; that figure is on independent renderings, not exact cues. On exact cues the controls below give 25–35% at 1,024 units.)
- **Memory-alone recall on independent renderings** at the same point: 0.02–0.08 at every size (chance 0.007).

## Diagnosis (post hoc)

`experiments/translation_memory_diagnosis.py`, end of batch 6, seed 299; `reports/translation-memory-v1-pilot-diagnosis.json`.

- **Nearest-prototype decoding of the host's code identifies few words.** Identification of the 120 source words from evaluation renderings against prototypes of eight training renderings: 0.61 (`mixed`) and 0.47 (`sparse48`) on the 16-dimensional code; 0.68 and 0.55 on the 64-dimensional encoder output. 96% and 99% of the variance across word prototypes lies in three directions of the code (92% and 97% for the encoder output). This is the performance of one decoder, not an information ceiling: another learned metric could recover more, and the minor directions may still carry distinctions.
- **Random expansion does not separate what the input does not.** Each word's separated pattern overlaps its nearest other word's by 0.76–0.85 on average (maximum 0.95–1.00), about as much as an evaluation rendering overlaps its own word (0.74–0.77).
- **Whitening is no remedy.** Whitening the code by anchor statistics (or, as an oracle, by the word prototypes) lowers between-word overlap to 0.41–0.50 but lowers a rendering's overlap with its own word just as much (0.28–0.39), and identification does not improve (0.40–0.52).

The diagnosis uses eight-rendering prototypes from the final encoder; the memory stores single-rendering keys written at different points and mapped into the reference space. The controls below use the memory's own keys.

## Controls (post hoc, after the fourth external review)

`experiments/translation_memory_controls.py`; `reports/translation-memory-v1-pilot-controls.json`; both pilot seeds and both arms, λ = 0.01; means over the four arm-seeds.

**1. Oracle source codes (diagnostic).** Independent random sparse source patterns per word, with the pilot's writes, teacher answers, target codes and retrieval rule. At 512 and 1,024 units the memory recovers the written target from every exact cue and from cues with 60% of their active units moved, at one step and at sixteen; no cycles. At 256 units, 60% corruption: 0.76 at sixteen steps against 0.67–0.71 at one. **The storage and retrieval mechanism works when identity and stability are supplied**; at this load recurrence adds nothing above 256 units.

**2. Historical keys.** The pilot's own consolidation (same renderings, write points, map, separation and teacher answers), every retrieval method on identical queries. Agreement with the written teacher answer, all batches pooled (correct against the true translation in brackets):

| queries | flat-store (dense) | flat-sparse | assoc, 1 step | assoc, 16 steps |
|---|---|---|---|---|
| exact written keys | 1.00 (0.65) | 0.99 (0.64) | 0.89 (0.59) | 0.84 (0.58) |
| other renderings, encoded at the word's write point | 0.26 (0.18) | 0.23 (0.17) | 0.22 (0.15) | 0.21 (0.14) |
| other renderings, end of batch 6 | 0.12 (0.08) | 0.12 (0.09) | 0.10 (0.07) | 0.10 (0.06) |
| other renderings, after the final context block | 0.04 (0.03) | 0.04 (0.03) | 0.04 (0.03) | 0.04 (0.02) |

(1,024 units, 41 active; 20 active is similar, 256 units with 5 active is worse for the sparse methods: exact-key agreement 0.52 flat-sparse, 0.35 at sixteen steps.)

Reading, in order of size:
- **Across renderings, at the write point, a different rendering finds the written answer only about a quarter of the time.** This is the largest step in the table. *Corrected by the decomposition below: every row uses the anchor map, and the map accounts for much of this step (recognition 0.25 with the map, 0.59 without). The table does not show that single-rendering keys fail on their own.*
- **Time then removes most of what remains** (0.26 → 0.12 by the end of batch 6 → 0.04 after the final context block).
- **Teacher errors** cap correctness: 0.65 of written answers are correct (0.39–0.86 by arm-seed).
- **Associative interference on correlated keys** costs 0.11–0.16 on exact keys (flat-sparse 0.99 against one-step 0.89 at 41 active); **settling** costs a further 0.05.
- Separation itself costs little at 1,024 units (dense 1.00 against flat-sparse 0.99 on exact keys).

So "the key is already wrong" dominates. Storage and settling lose a smaller, separate amount on correlated keys, and none on independent ones.

## Revision (for the researchers)

The strongest candidate (fourth review): an **identity-learning objective**, training different renderings of the same source symbol to agree and different symbols to stay distinguishable (supervised contrastive learning when symbol identity defines the positives; [Khosla et al. 2020](https://arxiv.org/abs/2004.11362)), with a **dedicated key head on the wider encoder** and the objective allowed to train the encoder (a projection of inadequate frozen features cannot restore discarded information). This changes the shared perception pathway, so every comparison receives the same change and its cost counts. Whether identity survives across successive batches, when only the current batch's symbols are trained, is a hypothesis.

The revised gate tests the memory interface, not fresh prototypes: recognition of all experienced words across independent renderings, and compatibility with historical single-rendering keys, by batch age. Lowering the storage gate without this would leave the memory study testing nothing.

## Identity host: perception gate (fails)

Host stage with the [identity revision](translation-memory-v1-draft.md) (`TM_IDENTITY=1`), same seeds and arms; the gate was committed before these results. Controls: `reports/translation-memory-v1-pilot-controls-identity.json`.

| reading (pooled over arm-seeds and batches) | gate | no identity loss | identity loss |
|---|---|---|---|
| recognition, queries at the write point | ≥ 0.90 | 0.25 | **0.59** |
| recognition, end of batch 6 | ≥ 0.80 | 0.11 | **0.21** |
| recognition, after the final context block | reported | 0.03 | 0.13 |

**The gate fails, so the memory grid does not run.** Other changes: exact written keys are now stored faithfully (at 1,024 units, 41 active, all four retrieval methods agree with the written answer on 1.00 of exact keys, against 0.84–0.89 for the associative reads before; 256 units with 5 active: 0.90 at sixteen steps), and teacher answers improve (0.77 correct against 0.65). Last-batch host accuracy at the end of its batch, by arm-seed: 0.72–0.90 (0.20–0.93 before). `frozen-encoder` still learns later batches poorly (0.00–0.42). The number-structure snapshots were recorded but not analysed here, so the loss's effect on transfer is unknown.

## Decomposition (post hoc)

`experiments/translation_memory_identity_diagnosis.py`; `reports/translation-memory-v1-identity-diagnosis.json` and, for the original host, `reports/translation-memory-v1-pilot-decomposition.json`. Recognition from evaluation renderings by nearest single-rendering key; means over the four arm-seeds. Measurements only: the "same encoder" rows re-encode old renderings the system no longer holds.

| reading | no identity loss | identity loss |
|---|---|---|
| current batch's keys only, 20-way, its own encoder, no map | 0.69 | 0.98 |
| all 120 keys, the same (final batch-6) encoder, no map | 0.31 | 0.51 (batches 1–5: 0.40–0.54; batch 6: 0.80) |
| the same, prototypes of eight renderings | 0.43 | 0.69 |
| the same, other training renderings as queries | 0.33 | 0.53 |
| keys as written, queries at the write point, no map | 0.59 | 0.95 |
| keys as written, queries at the write point, map λ = 0.01 | 0.25 | 0.59 |
| keys as written, end of batch 6, no map | 0.14 | 0.21 |
| keys as written, end of batch 6, map λ = 0.01 | 0.11 | 0.21 |

Reading (identity host):
- **The objective works for words being taught** (0.98 among the 20 current words).
- **It does not hold 120 words apart, even with no drift and no map** (0.51). Older batches are lower than the newest, and even the newest drops from 0.98 to 0.80 with 120 candidates, so lost training and a larger candidate set both contribute and are not separated here. Seen renderings do little better than new ones (0.53 against 0.51), so this is not overfitting to particular images.
- **Stored keys go stale.** Without a map, the end-of-batch-6 reading is about 0.98–0.99 for batch 6 and at most 0.20 for every older batch (0.00–0.02 for batch 1).
- **The anchor map moves accuracy rather than adding it.** It recovers some old keys and damages current ones: 0.95 → 0.59 at the write point, no pooled gain at the end of batch 6, at any λ tried (0.19–0.24; the selected λ was chosen on anchors for the host's code, not for these keys). The unmapped 0.95 overstates identity, since a word's competitors there are keys written by other encoder states, which have drifted away.

Correction: earlier statements in this report that single-rendering keys do not generalize across renderings, and that the map merely fails to prevent drift, were too strong; the map itself does damage on these keys.

## Next revision (for the researchers)

The two remaining failures are that identity for words no longer taught is not held, and that stored keys drift. Candidates:

1. **A stable key pathway.** A separate small encoder trained for identity on a broad background alphabet, then frozen or slowed, on the metric-learning premise that it discriminates symbols it never trained on (Siamese and prototypical networks **[familiar; read before use]**). Drift is removed by construction, at constant storage, and the host keeps learning, unlike `frozen-encoder`, which freezes the host's own path. Cheap to test without host training: recognition of held-out symbols. Constraint: the 512-symbol pool has about 210 symbols outside the lexicon's 150 + 150 alphabets and anchors, so a background alphabet needs more symbols. Whether perception that stops adapting suits the program's goals is the researchers' call.
2. **Hold the reference during training.** Penalize change in the stored anchors' keys while the encoder learns (feature distillation, Li and Hoiem 2017 **[familiar; read before use]**), instead of correcting afterwards with a map. Constant storage; keeps the encoder learning. Its risk is the map's: anchors may not represent lexicon words.
3. **Rewrite keys from replayed renderings.** Not pursued now. Bounded replay need not grow with facts, since it can keep a fixed subset; its unresolved problem here is how to maintain keys for facts whose renderings have been discarded.

## Stable keys: perception gate (passes)

The [stable key revision](translation-memory-v1-draft.md): a separate, fully frozen encoder, with rules committed before results and the test symbols read once. `experiments/translation_stable_keys.py`; `reports/translation-memory-v1-stable-keys-selection.json`, `reports/translation-memory-v1-stable-keys-confirmation.json`.

- **Symbols.** 3,207 candidates outside the pool leave 2,187 after duplicate-glyph thinning: 200 validation, 1,987 background.
- **Selection (validation symbols, seed 299).** All sixteen readings lie between 0.94 and 0.999. Rule's choice: base network, key width 32, 8,000 steps, **0.996**, 839,040 bytes. Wide with key width 32 reaches 0.999 at four times the bytes. Raw pixels 0.09, untrained base network 0.05.
- **Gate (test symbols: the 120-word lexicons, the buffered rendering as key, five independent renderings as queries, all 120 candidates).** **0.962 pooled** (seed 299: 0.952; seed 298: 0.972) against the working threshold 0.90. **Passes.** Raw pixels 0.10–0.11, untrained network 0.05–0.06 on the same images.

| candidates (teaching order) | 20 | 40 | 60 | 80 | 100 | 120 |
|---|---|---|---|---|---|---|
| seed 299 | 0.99 | 0.96 | 0.97 | 0.95 | 0.95 | 0.95 |
| seed 298 | 1.00 | 0.995 | 0.99 | 0.975 | 0.97 | 0.97 |

Readings:
- **Competition costs little over this range** (1.00–0.99 down to 0.97–0.95). The slope is what would limit a larger lexicon; 120 words does not show where it ends.
- **Test is below validation** (0.962 against 0.989–0.996). The task's pool comes from different Unicode blocks than the validation symbols, so validation overstates transfer to these symbols.
- **All 120 candidates held fixed across seven checkpoints give identical keys, queries and recognition**, on both seeds. This only checks the implementation, since no host trains in this test; the digest check during integration is the first real one.
- **Scope.** This is recognition through perception pretrained on about 2,000 symbols, which then stops adapting. It says nothing about continually adapting memory perception. Each key is still stored from one rendering, and 3–5% of renderings still reach the wrong key.

## Memory on frozen keys (pilot readings)

`TM_STABLE_KEYS=1`: the memory stage replays the original host's saved runs (no identity loss), with keys from the frozen encoder, both arms and branches. No map, so `assoc-nomap` is gone. `reports/translation-memory-v1-selection-stable-keys.json`; `reports/translation-memory-v1-memory-stable-keys-pilot-results.json.gz` (the rule's choice) and `-pilot-1024-results.json.gz` (a sensitivity run chosen after seeing the selection; its router thresholds are those of the selected size, so its router column does not count). Four arm-seeds of `main`, on the seeds used for selection: **not evidence for any contrast.**

Checks: at every point the encoder digest is unchanged, evaluation keys are bitwise identical, and evaluation preserves state.

**Selection.** All six sizes pass the storage gate on seed 299. The rule picks the smallest, 256 units with 5 active, cap 1. On seed 298 it recovers 0.942 (`mixed`) and 0.967 (`sparse48`), passing only when pooled.

| after the final context block, independent renderings | host | assoc, memory alone | flat-sparse | flat-store |
|---|---|---|---|---|
| batches 1–4, by arm-seed: 256 units, 5 active | 0.00–0.01 | 0.27–0.42 | 0.30–0.50 | 0.42–0.77 |
| batches 1–4, by arm-seed: 1,024 units, 41 active | 0.00–0.01 | 0.38–0.73 | 0.42–0.74 | 0.42–0.77 |
| persistent bytes, 256 units, 5 active | | 32,768 | 1,416 | 15,600 |
| persistent bytes, 1,024 units, 41 active | | 524,288 | 10,080 | 15,600 |

Readings, all descriptive:
- **The host alone keeps nothing and the memory does.** Old words fall to 0.00–0.01 in the host, and batch 6 to 0.13 (from 0.65) after 2,000 context-only updates.
- **Forgetting in the memory is small at the larger size and real at the smaller.** Batch 1, memory alone, right after its write and at the end: 0.455 → 0.435 at 1,024 units; 0.395 → 0.22 at 256. Its teacher was right on 0.46.
- **Teacher errors are now the largest loss at 1,024 units.** Only 0.65 of written answers are correct (0.39–0.86 by arm-seed), and recall follows that figure closely.
- **The rule's size is costly.** The storage gate tests exact cues only, and 256 units with 5 active loses about 0.2 on independent renderings against 1,024 units. Most of that is separation: `flat-sparse` on the same patterns loses nearly as much, `flat-store` nothing.
- **Recurrence adds nothing.** Mean recall on batches 1–4: one step 0.354, sixteen steps 0.332 (256 units); 0.541 and 0.544 (1,024 units). On corrupted exact cues sixteen steps beat one only at 256 units with 40% moved (0.794 against 0.779), and are lower or equal elsewhere. Free recall from 200 random starts regenerates 66 of 120 stored pairs at 256 units and 17 at 1,024 (one arm-seed read).
- **Routing has no job in this schedule.** A word is written before it is first evaluated, so the memory already holds current words. Gate weight does not vary with age (at the end of batch 6, 0.74 for batches 1–5 and 0.76 for batch 6 at 1,024 units), and the gate never beats memory alone by more than 0.01 on old batches. The dissociation is one-sided: the host fails old words, but nothing tests the memory failing on words it does not yet hold.
- **Sentences follow word recall.** Whole-sentence accuracy on unseen combinations of old words at 1,024 units: 0.26 (single-word 0.59, position 0.57–0.61). At 256 units: 0.05 (0.35). Reversal is learned, though the mixing is not crisp in the one arm-seed inspected; the claim stays composition with a learned permutation.
- **`frozen-encoder`** teaches poorly (0.09–0.36 correct), so its memory is poor too. With keys already frozen it has lost its role as a drift diagnostic.

**The explicit stores dominate at 120 words.** `flat-store` matches or beats every associative reading at 1/33 of the bytes of the 1,024-unit memory, and with 120 × 32 = 3,840 multiplications a read, against about 600,000 active connections over sixteen steps.

## Capacity on oracle codes (post hoc)

`experiments/translation_memory_capacity.py`; `reports/translation-memory-v1-capacity.json`. Independent random codes and exact cues, so a best case for the memory: the largest load tried (steps of √2) with at least 95% recovery, seeds 299 and 298 pooled.

| units, active | max load, 1 step | max load, 16 steps | joint memory, bytes per fact | source-to-target part only (a quarter), bytes per fact **[derived]** | flat-sparse, bytes per fact | flat-store, bytes per fact |
|---|---|---|---|---|---|---|
| 256, 5 | 800 | 566 | 41 | 10 | 12 | 130 |
| 256, 10 | 566 | 400 | 58 | 14 | 22 | 130 |
| 512, 10 | 1,600 | 1,600 | 82 | 20 | 22 | 130 |
| 512, 20 | 800 | 800 | 164 | 41 | 42 | 130 |
| 1,024, 20 | 3,200 | 3,200 | 164 | 41 | 42 | 130 |
| 1,024, 41 | 1,131 | 1,131 | 464 | 116 | 84 | 130 |

Readings:
- **On storage the joint memory never beats a slot store on the same sparse code**, and recurrence pays for three quarters of its weights while adding nothing here.
- **A source-to-target memory read in one step ties `flat-sparse` on bytes and can beat `flat-store`.** Its read work does not grow with facts, while exhaustive slots grow linearly. That is the only scaling advantage these numbers suggest. It is derived from the one-step load, not run as its own memory, and real keys will reduce it.
- **Failure is abrupt** (1,024 units with 20 active: 0.99 at 3,200, 0.37 at 4,525), as the clipped rule predicts.

## Binding revision: the size rule finds no memory (pilot)

The [binding revision](translation-memory-v1-draft.md), rules committed first. `experiments/translation_binding.py`; host stage `reports/translation-memory-v1-host-mid-results.json.gz` (its seed-299 learning curves equal the first host's, so the replication is exact); `reports/translation-memory-v1-binding-selection.json`. The run stage did not run, since there is no selected size.

**Perception gate on fresh seed 297:** 0.952 at all 120 candidates. Passes again (0.99 at 60, 0.98 at 80, 0.96 at 100).

**Size selection (seed 299, label writes, after all 120 writes).** With label writes the arms do not differ, since the host plays no part in the memory. Every size passes exact-cue recovery. **None comes within 0.02 of `flat-store`, so nothing is selected, and as specified the grid is not expanded.** Seed 297's memory readings were therefore never read and stay available as a fresh confirmation.

| units, active | one-step, all words (old words) | joint recurrent, 16 steps | flat-sparse | flat-store | gap to flat-store |
|---|---|---|---|---|---|
| 256, 5 | 0.625 (0.610) | 0.640 | 0.663 | 0.952 (0.943) | 0.327 |
| 256, 10 | 0.693 (0.682) | 0.652 | 0.790 | 0.952 | 0.259 |
| 512, 10 | 0.803 (0.797) | 0.800 | 0.817 | 0.952 | 0.149 |
| 512, 20 | 0.845 (0.853) | 0.828 | 0.888 | 0.952 | 0.107 |
| 1,024, 20 | 0.893 (0.883) | 0.898 | 0.917 | 0.952 | 0.059 |
| 1,024, 41 | 0.915 (0.907) | 0.923 | 0.925 | 0.952 | **0.037** |

At the closest size, most of the gap is the expansion (`flat-sparse` loses 0.027 to `flat-store` on identical events), and associative storage adds 0.010. The trend suggests a larger memory would close it, at more bytes than the 8.4 times `flat-store` it already costs.

## Sensitivity at 1,024 units, 41 active (not selected)

The size that came closest, read on seed 299 only, so seed 297 stays unseen. Thresholds were chosen as the rule specifies, at this size. `reports/translation-memory-v1-binding-sensitivity-1024-41-results.json.gz`. **Two arm-seeds, a size that failed the rule: descriptive only.** Checks: state preserved and keys identical at every evaluation; the standalone one-step reads equal the joint network's first step on corrupted real patterns.

**Retention.** After the final context block, old words (batches 1–4): host 0.00.

| memory alone | one-step | joint recurrent | flat-sparse | flat-store |
|---|---|---|---|---|
| label writes, correct | 0.907 | 0.912 | 0.920 | 0.943 |
| teacher writes, correct (range over the two arms) | 0.52–0.54 | 0.52–0.55 | 0.54–0.58 | 0.55–0.58 |
| teacher writes, fidelity to the written answer | 0.90 | 0.90 | 0.92–0.93 | 0.94–0.95 |

Under label writes, retained recall sits close to perception's own limit. Under teacher writes the memory keeps what it was told, about as faithfully, and what it was told was right 0.67 of the time. Recurrence: 0.907 at one step, 0.912 at sixteen.

**Routing before the write** (`pre_write`, batches 2–6, mean; the two arms are within 0.04):

| | current words | older words |
|---|---|---|
| host alone | 0.65–0.69 | 0.00 |
| memory alone, one-step | 0.00 | 0.95 |
| learned gate, one-step | 0.51–0.53 | 0.77–0.78 |
| confidence router, one-step | 0.49–0.50 | 0.84 |
| memory alone, flat-store | 0.00 | 0.96 |
| learned gate, flat-store | 0.55–0.60 | 0.81–0.82 |
| confidence router, flat-store | 0.41–0.45 | **0.96** |

Pre-write evaluation does give routing a job, and the gate weight now separates the states (0.23–0.25 on current words against about 0.71 on older). **By the stated test, no router on the one-step memory passes: each gains current words but gives up old ones (0.95 → 0.78 or 0.84).** The fixed router on `flat-store` is the only one that preserves old words, because a dense key's cosine separates absent from present cleanly.

**The routers do not hold up at the final point.** On old words, against 0.907 memory alone: one-step gate 0.49–0.56, one-step router 0.68. `flat-sparse` gate 0.19–0.32. The joint recurrent gate reaches 0.83–0.91 with the same memory-alone accuracy, so **that gap is not a retrieval advantage**. A threshold set on batches 2–3 also meets lower overlaps at full load. The gate is fragile, and as a mixture of probabilities it can let an overconfident host outvote a partial memory read; settling may help by sharpening reads to exact codes. These two causes are hypotheses, not tested here.

**Sentences** read through the gate, so they inherit this. Whole-sentence accuracy under label writes: one-step 0.04–0.09, flat-store 0.53–0.60, joint recurrent 0.56–0.76. These measure the gate as much as composition, and should not be read as comparing memories.

**Cost at this size.** Associative weights 131,072 bytes (one-step) against 524,288 (joint), 10,080 (`flat-sparse`) and 15,600 (`flat-store`). Beside them in every condition: the frozen encoder at 839,040 and the buffer at 125,800, which dominate total storage. A read: about 12,800 active connections (1,048,576 dense) for one-step, against 120 × 32 = 3,840 multiplications for `flat-store`.

## Where this leaves the design

At 120 words `flat-store` is ahead on accuracy, bytes, work per read and routing, and nothing yet measured favours the associative memory on real keys. The one scaling argument left, read work that does not grow with facts, is derived from oracle codes, and is the question a load curve on real keys would answer. Label writes need no host, so that curve is cheap.

Decisions for the researchers: whether to keep testing an associative memory or adopt the explicit store as the system (the earlier decision made it a baseline); how to repair routing, which fails for reasons separate from the choice of memory; and how far to pursue the expansion loss. Still to do before any main run: the contract, the literature tagged *read before freezing*, `rehearsal-matched`, key encoders and host runs for seeds 360–369.

## Computation (measured costs only)

Host stage: 819 s wall on the GTX 1650, four tasks in parallel (761–776 s each). Selection: 27 s wall on the CPU with 11 workers (λ 2.5 s, grid 25 s). Diagnosis: about 5 s. Controls: 22 s wall on the CPU. Identity host stage: 918 s wall on the same GPU (about 12% longer than the original). Decomposition: about 6 s per host. Stable keys, selection: symbol filter 28 s, rendering 39 s, four fits 26–32 s each on the same GPU; confirmation: 69 s wall, including one further fit and its rendering. Encoder 839,040 persistent bytes. Memory selection on frozen keys: 38 s wall, CPU. Pilot replay (8 tasks, 11 workers): 47 s wall and 320 s worker CPU at 256 units; 206 s and 1,436 s at 1,024 units. Capacity: 30 s wall. Binding revision: host stage 576 s wall on the GPU (seeds 299 and 297, 4 tasks), seed 297 encoder and gate 68 s wall, selection 34 s and sensitivity 193 s wall on the CPU. Timing is in the runtime files.
