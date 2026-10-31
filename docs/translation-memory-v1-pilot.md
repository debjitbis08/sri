# Translation memory v1: pilot (seeds 299 and 298)

Status: **pilot, 2026-10-31**. Design: [draft](translation-memory-v1-draft.md). Not evidence about memory; it decides whether the design can be frozen. Outcome by the draft's rule: **no memory configuration passes the storage gate, so the design is revised before freezing.** The router selection and confirmation did not run.

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
- **Single-rendering keys do not generalize across renderings.** Even with no drift and exact dense lookup, a different rendering of the same word finds its own key only about a quarter of the time. This is the largest loss.
- **Drift then removes most of what remains** (0.26 → 0.12 by the end of batch 6 → 0.04 after the final context block); the anchor map does not prevent it.
- **Teacher errors** cap correctness: 0.65 of written answers are correct (0.39–0.86 by arm-seed).
- **Associative interference on correlated keys** costs 0.11–0.16 on exact keys (flat-sparse 0.99 against one-step 0.89 at 41 active); **settling** costs a further 0.05.
- Separation itself costs little at 1,024 units (dense 1.00 against flat-sparse 0.99 on exact keys).

So "the key is already wrong" dominates: across renderings and across time. Storage and settling lose a smaller, separate amount on correlated keys, and none on independent ones.

## Revision (for the researchers)

The strongest candidate (fourth review): an **identity-learning objective**, training different renderings of the same source symbol to agree and different symbols to stay distinguishable (supervised contrastive learning when symbol identity defines the positives; [Khosla et al. 2020](https://arxiv.org/abs/2004.11362)), with a **dedicated key head on the wider encoder** and the objective allowed to train the encoder (a projection of inadequate frozen features cannot restore discarded information). This changes the shared perception pathway, so every comparison receives the same change and its cost counts. Whether identity survives across successive batches, when only the current batch's symbols are trained, is a hypothesis.

The revised gate tests the memory interface, not fresh prototypes: recognition of all experienced words across independent renderings, and compatibility with historical single-rendering keys, by batch age. Lowering the storage gate without this would leave the memory study testing nothing.

## Computation (measured costs only)

Host stage: 819 s wall on the GTX 1650, four tasks in parallel (761–776 s each). Selection: 27 s wall on the CPU with 11 workers (λ 2.5 s, grid 25 s). Diagnosis: about 5 s. Controls: 22 s wall on the CPU. Timing is in the runtime files.
