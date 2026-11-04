# Shared output v1, stage 0: naming-only transfer on existing checkpoints

Status: **stage 0 of the [draft contract](shared-output-v1-contract.md), run 2026-11-04, seeds 300–303. A diagnostic on existing
checkpoints, not headline evidence.**

- **Code:** `experiments/shared_output.py` (`names` on the GPU, `stage0` on the CPU).
- **Reports:**
  - `reports/shared-output-v1-stage0.json`;
  - `reports/shared-output-v1-stage0-runtime.json`;
  - `reports/shared-output-v1-names-runtime.json`, which holds seeds 301–303 only; seed 300's names were encoded in an
    earlier launch, with the same cost (585 images, 19 s).
- **Name codes:** `runs/shared-output-v1/names-seed{300..303}.npz` (not tracked).

## Design

- **Host.** The step 8 `joined-mlp-calibrated` delay checkpoints (`sparse48`, `g` and `P` unchanged), with a fresh Adam and
  RigL off.
- **Names.** 45 per seed, 5 per count, drawn from the pairing pool after memory's sets A, B and unseen. They are taught only by
  matching: 200 updates of delay-style context episodes, each with 9 naming rows added (a name against the update's 9 candidate
  collections through `g(same)`).
- **Leakage audit.** It ran every update. 1,800 naming rows per seed, and no name in any `next` pair, comparison draw, sequence
  row or candidate slot.
- **Inventory.** 8 training collections per count (72), stored as raw frozen codes and re-read through the current core.
- **Continuation.**
  - Queries: (evaluation collection n∓1, cur), with n in 2–8.
  - Readout: `P`, then the nearest inventory item.
  - Inputs for cur:
    - **name (direct):** the name's own core code;
    - **name, grounded:** the name's recall over the inventory through `g(same)`, either the top item's state (hard) or the
      softmax-weighted state (soft), then fed to `P`. This is evaluation only: nothing was trained this way;
    - **familiar items:** an evaluation collection, a Devanagari form (trained counts) or taps.

## Results

Mean over four seeds, with the per-seed values in brackets. Chance is 0.11.

| Read | Update 0 | Update 200 |
|---|---|---|
| **Name acquisition** | | |
| Name → 9 collections, `g(same)` | 0.14 | 0.46 (0.59, 0.39, 0.40, 0.46) |
| Name recall over the 72-item inventory | 0.15 | 0.45 |
| Names whose nearest count mean is their own count | 0.14 | 0.26 |
| **Continuation with a name as the current item**, ascending / descending | | |
| Direct | 0.20 / 0.14 | 0.39 / 0.27 |
| Grounded, top item | 0.19 / 0.17 | **0.56 / 0.50** |
| Grounded, soft | 0.19 / 0.17 | 0.42 / 0.37 |
| **Familiar current item**, ascending / descending | | |
| Collection (within kind) | 0.94 / 0.89 | 0.91 / 0.88 |
| Taps (mixed kind, collection history) | 0.98 / 0.91 | 0.97 / 0.93 |
| Devanagari form (mixed kind) | 0.67 / 0.58 | 0.48 / 0.21 |
| **Rejection of the reverse and repeated counts** (AUROC, mean of directions) | | |
| Name, direct | 0.59 | 0.65 |
| Name, grounded top item | 0.59 | 0.75 |
| Collection | 0.96 | 0.95 |
| **Familiar matching** | | |
| Taps → collections | 0.97 | 0.94 |
| Devanagari → collections, trained counts | 0.47 | **0.20** |
| Devanagari → collections, withheld counts | 0.62 | **0.07** |

## What this shows

- **Grounding before the relation transfers better than the name's own code.** Fed through its recalled participant, a name
  continues at 0.56 / 0.50, against 0.39 / 0.27 when its own code goes to `P`. Grounding is higher in every seed and direction.
  This holds without any training through the grounded path.
- **The name's code is not substitutable.** Only 0.26 of names sit nearest their own count, although 0.46 are matched
  correctly. `g` accepts many names without placing them near the count, as prediction 1 of the contract expected.
- **Acquisition is the limit.** Naming reaches 0.46 after 200 updates and is still rising, far below the contract's 0.80 gate.
  The grounded read is bounded by it: a wrongly identified name continues from the wrong count.
- **Mixed-kind histories are readable when the item is well identified.** (collection, taps) reads as well as
  (collection, collection) with no mixed-kind training. The lower Devanagari read follows Devanagari's own weak matching at the
  delay (0.47).
  - The contract's rule would keep mixed-kind rows: Devanagari is not within 0.05 of collections. But the evidence points to
    identification, not history kind, as the cause.
- **Naming interferes with familiar forms.** 200 updates of naming lower Devanagari matching from 0.47 to 0.20 on trained counts
  and from 0.62 to 0.07 on withheld counts, while taps hold.
  - Names and forms share the image projection, and only names are taught here. Context episodes carry no forms.
  - In the pilot, names are taught alongside Devanagari, which may protect it. This must be measured.

## Computation

| Phase | Measure |
|---|---|
| Name rendering and frozen encoding (GTX 1650) | 585 images per seed: 1.5 s rendering, 19.2 s encoding |
| Stage 0 job (3 workers, CPU) | 43 s wall |
| Per seed | about 14 s wall; most of it is loading the 270 MB code table |
| Training, 200 updates | about 2 s CPU |
| Reads, at 4 points | about 0.9 s CPU |
| Selection and tuning | none |

Seed 300's record comes from the earlier single-seed run (39 s wall), which the durable runner reused.
