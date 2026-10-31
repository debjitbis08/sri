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
- **Storage gate (95% recovery of the written target on exact cues after all 120 writes, 16-step read):** no configuration passes. Best: 1,024 units, 41 active: 0.92 (`mixed`), 0.83 (`sparse48`). The smallest, 256 units with 5 active: 0.29 and 0.34. Recurrence lowers recovery in every configuration (checkpoint 1 is highest), and 31–79% of exact-cue reads end in a cycle.
- **Memory-alone recall on independent renderings** at the same point: 0.02–0.08 at every size (chance 0.007).

## Diagnosis (post hoc)

`experiments/translation_memory_diagnosis.py`, end of batch 6, seed 299; `reports/translation-memory-v1-pilot-diagnosis.json`.

- **The host's code carries little word identity.** Nearest-prototype identification of the 120 source words from evaluation renderings: 0.61 (`mixed`) and 0.47 (`sparse48`) on the 16-dimensional code; 0.68 and 0.55 on the 64-dimensional encoder output. 96% and 99% of the variance across word prototypes lies in three directions of the code (92% and 97% for the encoder output). Any memory keyed on these representations, the explicit stores included, is capped near those figures before storage is considered.
- **Random expansion does not separate what the input does not.** Each word's separated pattern overlaps its nearest other word's by 0.76–0.85 on average (maximum 0.95–1.00), about as much as an evaluation rendering overlaps its own word (0.74–0.77). Near-duplicate keys interfere at write time, which is why even exact cues fail.
- **Whitening is no remedy.** Whitening the code by anchor statistics (or, as an oracle, by the word prototypes) lowers between-word overlap to 0.41–0.50 but lowers a rendering's overlap with its own word just as much (0.28–0.39), and identification does not improve (0.40–0.52).

The host still answers recent words well because its matching head discriminates a batch of 20 words with targets attached; the code it builds does not keep 120 symbols apart. The memory's failure is a perception failure upstream of storage (the first of the two problems in the program: grouping and separating inputs, the part with standard solutions).

## Revision options (for the researchers)

1. **A perception objective for identity.** Train the host to keep source symbols apart across renderings (instance or same-symbol contrastive learning on every source rendering it sees), so the code carries identity for all experienced words, and key the memory on that code. Established (contrastive instance discrimination); changes the host for every variant equally.
2. **A dedicated key pathway.** A learned key projection of the encoder output, trained by the same identity objective, used only by the memory. Leaves the host's code unchanged; the key pathway's drift is then what the map must handle.
3. **Keep the representation and lower the gate.** Accept recovery near 0.9 and recall near chance: the memory study would then test nothing.

Option 1 or 2 is needed before the storage comparison means anything. Either should be checked first with the diagnosis script (identification of 120 words well above 0.9) before rerunning the grid.

## Computation (measured costs only)

Host stage: 819 s wall on the GTX 1650, four tasks in parallel (761–776 s each). Selection: 27 s wall on the CPU with 11 workers (λ 2.5 s, grid 25 s). Diagnosis: about 5 s. Timing is in the runtime files.
