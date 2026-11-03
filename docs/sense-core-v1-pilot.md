# Step 8: does frozen sight serve the core?

Status: **pilot, 2026-11-04, seeds 300–303. Not a contract.** This is step 8 of the [sense integration plan](sense-integration-plan.md). Code: `experiments/sense_core.py`, with tables from `experiments/sense_core_summary.py`. Report: `reports/sense-core-v1-pilot.json.gz`.

## Question

Memory can read frozen sight ([stability pilot](sense-stable-v1-pilot.md)), but the core has a different job: counts, symbols across scripts, and relations such as comparison, succession and recall across senses. Do frozen DINOv2-S and TrOCR-small features support that job at least as well as the image encoder the core trains for itself? If not, integrating them would set the core back.

## Design

The core is `sparse48`. It is trained and read exactly as in the [limited-memory study](limited-memory-v1-results.md)'s main stage (`experiments.limited_memory` `run_main`):
- **Training:** English, Chinese and Devanagari numerals in three blocks, then a delay, then relearning.
- **Reads at every snapshot:** matching by count on paired and withheld counts, the eight transfer measures (comparison, taps ↔ glyphs across senses, withheld matching), succession and composition.

Only the image encoder changes:

| Arm | Image encoder | Trained parameters in it |
|---|---|---|
| `existing` | the core's own small CNN, trained with the core | 213,888 |
| `joined-mlp` | frozen DINOv2-S patch mean and TrOCR-small class token, each standardized with fixed statistics, concatenated (768), then a trained MLP 768 → 128 → out, the shape of the existing encoder's last two layers | 106,688 |
| `joined-linear` | the same frozen blocks, then one trained linear layer | 49,216 |
| `dinov2-mlp` | DINOv2-S alone, then the MLP | 57,536 |

**The core gets the raw frozen blocks through a small projection.** It does not get memory's centred, query-weighted code, as the external reviewer asked. Memory isn't used here, and its baseline stays kept+frozen.

**Frozen codes are cached, not recomputed.** They are computed once for every image the core sees: the three scripts and Thai digits (the study's transfer script), dice, and the evaluation candidate sets. Training and reads look them up by image content, and every run had zero cache misses.

**Standardization statistics** come from all cached images, evaluation images included. That shares input statistics with no labels attached.

## Results

**Reference check.** `existing` equals the limited-memory study's `sparse48` at seeds 300–303 in 1,540 of 1,540 measures, at every snapshot.

Each cell is the mean over 4 seeds, with the range in brackets. Chance for matching is 1/9.

**The script just taught**, read at the end of its own block:

| Measure | `existing` | `joined-mlp` | `joined-linear` | `dinov2-mlp` |
|---|---|---|---|---|
| Block 1, English matching, paired | 0.17 (0.12–0.27) | **0.73 (0.60–0.81)** | 0.30 (0.10–0.41) | 0.42 (0.16–0.68) |
| Block 1, comparison, English | 0.73 (0.70–0.77) | **0.97 (0.95–0.98)** | 0.78 (0.56–0.87) | 0.90 (0.80–0.95) |
| Block 2, Chinese matching, paired | 0.45 (0.20–0.60) | 0.65 (0.12–0.98) | 0.66 (0.14–0.91) | **0.95 (0.94–0.97)** |
| Block 2, Chinese matching, withheld | 0.37 (0.20–0.47) | 0.65 (0.37–0.90) | 0.56 (0.18–0.75) | **0.82 (0.62–0.92)** |
| Block 3, Devanagari matching, paired | 0.54 (0.50–0.61) | **0.97 (0.94–0.99)** | 0.89 (0.84–0.92) | 0.92 (0.87–0.94) |
| Block 3, Devanagari matching, withheld | 0.32 (0.22–0.42) | **0.95 (0.87–1.00)** | 0.59 (0.33–0.82) | 0.65 (0.50–0.77) |
| Block 3, taps → Devanagari, withheld | 0.73 (0.62–0.93) | **0.99 (0.97–1.00)** | 0.75 (0.27–0.93) | 0.66 (0.43–1.00) |

**At block 3**, `joined-mlp` minus `existing`, paired by seed:

| Measure | Difference (range over seeds) |
|---|---|
| Devanagari matching, paired | +0.43 (+0.36 to +0.46) |
| Devanagari matching, withheld | +0.63 (+0.45 to +0.78) |
| Taps → Devanagari, withheld | +0.26 (+0.07 to +0.38) |
| Taps → Chinese, withheld | +0.15 (+0.03 to +0.25) |
| Comparison, Chinese | +0.10 (+0.06 to +0.16) |
| Comparison, Devanagari | +0.01 (−0.01 to +0.06) |
| Comparison, English | +0.01 (−0.04 to +0.04) |
| Chinese matching, withheld (taught at block 2) | +0.11 (−0.12 to +0.27) |
| English matching, paired (taught at block 1) | +0.06 (+0.03 to +0.09) |

**Composition at block 3** (collections of dots, one step up, correct distance): 0.95 (0.93–0.97) for `joined-mlp` against 0.49 (0.46–0.54) for `existing`. One step down: 0.86 against 0.46. On Chinese forms, composition is low in every arm (0.10–0.23).

**Earlier scripts are still forgotten.** At block 3, English matching is 0.07–0.13 in every arm, and Chinese 0.10–0.23.

**After relearning**, 200 updates on English with rehearsal from a buffer, `joined-mlp` recovers far more than `existing`:

| Measure | `existing` | `joined-mlp` |
|---|---|---|
| English matching, paired | 0.22 | 0.74 |
| Chinese matching, paired | 0.28 | 0.80 |
| Chinese matching, withheld | 0.21 | 0.61 |
| Comparison, English | 0.64 | 0.88 |

## What this shows

- **Frozen sight serves the core better than the core's own encoder on these inputs.** With half as many trained parameters in the image path, the joined frozen blocks plus a small MLP give clearly higher reads on:
  - matching the script being taught, on taught counts and on withheld counts (Devanagari withheld 0.95 against 0.32);
  - recall across senses on withheld counts (taps → Devanagari 0.99 against 0.73);
  - composition of dot collections (0.95 against 0.49).
- **Relations are at least as good.** Comparison is equal on two scripts and +0.10 on Chinese.
- **The projection matters.**
  - One linear layer recovers part of the gain, but not all.
  - DINOv2 alone is best on Chinese but weaker than the joined code on Devanagari and on recall across senses. TrOCR earns its place here too.
- **Forgetting is the core's problem, not the sense's.** Every arm forgets earlier scripts by block 3 to the same degree. Better sight does not change that, though it makes relearning much faster.
- **Answer to step 8:** yes. Frozen established models, through a small trained projection, support counts, symbols and relational use in the core, on the core's own curriculum.

## What this does not establish

- **Significance.** These are pilot seeds, and four is too few for intervals. Several measures vary widely across seeds in every arm, for example Chinese matching at block 2 (0.12–0.98 for `joined-mlp`).
- **Inputs beyond these scripts.** These are 28×28 renderings of numerals and dice, upsampled for the frozen models. Natural-image inputs to the core are untested.
- **Memory and core together.** The core and memory have not yet been run together in one forward pass with frozen sight. That is step 10.
- **What the projection learns,** and whether a frozen projection would do as well.

## Prior work

These are familiar methods, not re-read for this pilot.

- **Linear probes and small heads on frozen foundation features:** the DINOv2 paper's own evaluations (Oquab et al. 2023) and the linear-probe protocol generally. The result here is the expected one for frozen self-supervised features on small classification-like tasks.
- **Feature-extractor transfer:** Yosinski et al. 2014. Frozen features with a trained head beat a small network trained from scratch when the target task has little data per class.

## Scaling

- **Per input:** one forward pass per frozen model, once per distinct image (0.020 s DINOv2-S, 0.054 s TrOCR-small on this GPU), after which training reuses the cached code. Images seen once cost their encoding once.
- **Per update:** a lookup plus a 768 → 128 → out MLP, cheaper than the existing CNN.
- **Cache:** 768 floats per distinct image, 87,660 images here (270 MB). With non-repeating inputs there is no reuse, and the cost per input is the forward passes.

## Computation

**Frozen-code cache** (GTX 1650), built in three parts:

| Part | Sources | Wall | CPU |
|---|---|---|---|
| 1 | the three scripts and dice, 72,270 images | 3,013 s | 2,984 s |
| 2 | Thai digits, 11,700 images | 429 s | 409 s |
| 3 | evaluation candidate sets, 13,590 images | 497 s | 478 s |

Images missing from the cache were found by a block-3 snapshot (`pilot capture`), and are the same for seeds 300 and 301.

**Core runs:** three workers, 16 runs of 160–212 s wall each. 982 s wall for the 13 runs of the final launch; three `existing` runs had finished in an earlier launch. 3,209 s aggregate worker CPU, peak RSS 3.6 GB, GPU peak 351 MB per frozen-encoder worker and 89 MB for `existing`.
