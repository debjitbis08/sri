# Two-script reads v1

Status: **contract and results, 2026-11-04.** The contract part was written before the full run; the one-task check ran first. These are reads on existing checkpoints, with no training. The user asked for them
after the [K-arm pilot](shared-output-v1-k-arms-pilot.md). Results follow below.

## Question

Every relation is trained within one kind: `next`, `more` and continuation over collections and taps, plus matching from forms to
collections. Do expressions whose two items come from **different scripts** work? Examples are comparing a Devanagari 3 with a
Chinese 5, or continuing (Chinese 2, Devanagari 3) → 4. Earlier reads mixed only collections, taps, names and one script.

## Design

- **Checkpoints.** The [shared output pilot](shared-output-v1-pilot.md) block-3, delay and relearn checkpoints of `gp`,
  `common-raw`, `common-k-cand` and `common-k-shared`, at seeds 300–303. The grounded arms are left out: grounding was dropped.
- **Kinds.** Collections, taps, English, Chinese, Devanagari and Thai digits, with 30 evaluation items per count. Thai is never
  taught, so it is the untaught-script control. English is taught in block 1 and relearned at the end. Chinese is taught in block 2
  and Devanagari in block 3.
- **Reads, for every ordered pair of kinds (A, B).**
  - **Matching:** `same` from an item of A against a gallery of B, one item per count. The answer is correct if the top count is
    the item's own. Chance is 1/9. This is Part D's gallery read, generalized to every pair of kinds.
  - **Comparison:** `more(a of A, b of B) > 0` exactly when a > b, over all a ≠ b. This is Part D's comparison read.
  - **Continuation:** (prev of A, cur of B) → the top participant of M, ascending and descending, for n = 2–8. There is no
    abstention, since the pilot's thresholds are not re-selected.
- **Splits.** Rows are split by whether any count involved is withheld in its script. Chinese and Devanagari each withhold two
  counts, which are never matched in training.
- **Controls.** The same-kind diagonal. Collection rows and columns reproduce the pilot's familiar reads (checked at seed 300: the
  collection comparison is 1.00 and (collection, Devanagari) continuation 0.97, as in the pilot). Thai × Thai measures what
  perception alone gives within an untaught script.

## Predictions

1. **[local evidence] Comparison transfers across scripts.** Comparison of untaught forms within a script persists (Devanagari
   0.84). If each form's place in the count order is shared, then pairing two scripts should cost little.
2. **[local evidence] Matching across scripts is weak.** `same` is trained only from forms to collections, never between forms. Form
   matching faded during the delay.
3. **[derived] Continuation across scripts follows the weaker item's identification,** as mixed-kind histories did.
4. **Thai stays near chance on comparison and continuation.** It may be above chance on Thai × Thai matching, because renderings
   of one glyph resemble each other.

## Prior work

These are **[familiar]** sources, recalled rather than re-read. Shared entries are in
[prior work](prior-work.md#shared-output-scorer-and-naming-only-transfer).

- **Code-switching** in bilinguals: mixed-language arithmetic is slower and less accurate than single-language arithmetic.
  Arithmetic facts are partly stored in the language of learning (Spelke and Tsivkin 2001; Dehaene's triple-code model, 1992).
  Magnitude comparison is format-independent; exact facts are format-bound.
- **Cross-lingual transfer in shared multilingual encoders** (multilingual BERT, Pires et al. 2019) is better for shared
  structure than for item pairings.
- **Known failure mode:** a shared space that aligns each script with collections need not align scripts with each other. This
  is the hubness and alignment problem of cross-lingual embeddings.

## Computation

- **Theoretical cost.** Each checkpoint takes 36 ordered pairs. For matching, that is 9 counts × 30 items × 9 candidates of `same`;
  for comparison, 72 count pairs × 30; for continuation, 14 × 30 × 72 participants. With 16 tasks × 3 points, everything is about
  2.5 million reader calls of about 15k FLOPs each, roughly 40 GFLOP.
- **Storage.** No new state. The report is a few hundred KB.
- **Measured** in `reports/two-script-reads-v1-runtime.json`. The one-task check took 29 s on 1 CPU thread with a 2.5 GB peak RSS,
  and counts as selection cost.

## Commands

```
.venv/bin/python -m experiments.two_script_reads
.venv/bin/python -m experiments.two_script_reads summary
```

## Results

Run 2026-11-04: 16 tasks, 161 s wall on 4 CPU workers, 528 s aggregate CPU, 2.5 GB peak RSS per worker. The report is
`reports/two-script-reads-v1.json`. Means are over four seeds, with the range of seed means in brackets. Reads are grouped as
follows:
- **Two scripts:** the six ordered pairs of English, Chinese and Devanagari.
- **Same script:** the diagonal of those three scripts.
- **Script–collection:** forms read against collections, both ways.

Chance is 0.5 for comparison and 1/9 for matching and continuation.

**Comparison: works across scripts, as well as within one.**

| Group | Point | `gp` | `common-raw` | `common-k-cand` | `common-k-shared` |
|---|---|---|---|---|---|
| Chinese ↔ Devanagari | block 3 | 0.81 (0.78–0.83) | 0.78 | 0.88 | **0.90** (0.87–0.93) |
| | delay | 0.81 | 0.74 | 0.85 | **0.88** |
| Two scripts (with English) | block 3 | 0.69 | 0.67 | 0.68 | 0.72 |
| Same script | block 3 | 0.75 | 0.72 | 0.77 | 0.81 |
| Script–collection | block 3 | 0.83 | 0.81 | 0.82 | 0.85 |
| Thai × Thai (untaught) | block 3 | 0.46 | 0.47 | 0.48 | 0.51 |

The pairs that include English are lower because English has faded by block 3 (English matching 0.12). On Chinese ↔ Devanagari
rows that touch a withheld count (forms never matched, placed only through relations), comparison is as good as on paired rows:
`gp` 0.83 against 0.79, `common-k-cand` 0.88 against 0.88, both at block 3.

**Matching: fails across scripts.**

| Group | Point | `gp` | `common-raw` | `common-k-cand` | `common-k-shared` |
|---|---|---|---|---|---|
| Two scripts | block 3 | 0.15 | 0.16 | 0.17 | 0.17 |
| | delay | 0.13 | 0.14 | 0.18 | 0.17 |
| Chinese ↔ Devanagari | block 3 | 0.20 | 0.23 | 0.24 | 0.26 |
| Script–collection | block 3 | 0.41 | 0.36 | 0.44 | 0.42 |
| Same script | block 3 | 0.63 | 0.55 | 0.66 | 0.63 |
| Thai × Thai (untaught) | block 3 | **0.61** | 0.45 | 0.63 | 0.53 |

Same-script matching is no evidence of a shared count. Thai is never taught, yet Thai × Thai matching reaches about the same level,
from glyph resemblance alone.

**Continuation: weak across scripts, limited by identification.**

| Group | Point | `gp` | `common-raw` | `common-k-cand` | `common-k-shared` |
|---|---|---|---|---|---|
| Two scripts | block 3 | 0.23 | 0.26 | 0.23 | 0.21 |
| | delay | 0.19 | 0.17 | 0.21 | 0.19 |
| Chinese ↔ Devanagari | block 3 | 0.30 | 0.35 | 0.33 | 0.31 |
| Same script | block 3 | 0.41 | 0.41 | 0.41 | 0.41 |
| Script–collection | block 3 | 0.47 | 0.48 | 0.50 | 0.45 |
| Collections | block 3 | 0.91 | 0.90 | 0.93 | 0.84 |
| Thai × Thai | block 3 | 0.16 | 0.14 | 0.14 | 0.12 |

**After English relearning, every cross-script read rises,** including those that do not involve English. For Chinese ↔ Devanagari
in `gp`, from delay to relearn:
- matching goes 0.18 → 0.61;
- continuation 0.23 → 0.66;
- comparison 0.81 → 0.89.

In `common-k-cand` the same reads reach 0.69, 0.76 and 0.93. Relearning trains English forms against collections. It appears to
re-align the shared form pathway for every script, consistent with the pilot's paired Chinese matching rising from 0.11 at delay to
0.79 after relearning.

## What this shows

1. **Prediction 1 holds.** Order carries across scripts. Comparing a Chinese form with a Devanagari form is as accurate as comparing
   two forms of one script, and withheld forms compare as well as paired ones. The count order is a shared structure that every
   script maps into.
2. **Prediction 2 holds.** Exact identity does not carry across scripts. `same` from a Chinese form to a Devanagari gallery is near
   chance, although each script reaches collections at 0.36–0.44. Each script is aligned with collections loosely, and the scripts
   are not aligned with each other well enough for exact matches. That is the known failure mode of cross-lingual alignment through a
   hub.
3. **Prediction 3 holds.** Continuation needs the exact count, so it sits between the two, near the weaker item's identification.
4. **Prediction 4 holds**, apart from glyph resemblance within Thai.
5. **The general pattern matches the delay results.** What a relation needs only coarsely (order) persists and transfers; what
   needs an exact pairing (identity) fades and fails to transfer. `common-k-shared` is best on comparison (0.90 Chinese ↔
   Devanagari), where every slot reads K. `common-k-cand` is best on matching and continuation after the delay.

**Caveats.** The comparison accuracy does not show whether close pairs (n against n + 1) are compared correctly or only distant
ones. No cross-script expression was trained, so none of this tests learning from mixed examples. Pilot checkpoints, four seeds.

## Computation (measured)

- 16 tasks × 3 checkpoints, 161 s wall on 4 CPU workers, 528 s aggregate worker CPU.
- 2.5 GB peak RSS per worker, mostly frozen sensory code caches.
- The one-task check took 29 s.
