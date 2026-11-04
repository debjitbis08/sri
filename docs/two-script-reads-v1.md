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
  trained as numbers: it enters no training row in any block or in relearning, and the image statistics come from English and dots
  only. It does pass through the same frozen senses, projection and core, so it is the control for a script never trained as numbers. English is taught in block 1 and relearned at the end. Chinese is taught in block 2
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

## Follow-up reads (2026-11-05)

Written before the run. Same checkpoints, arms, seeds and items; `experiments/two_script_reads_v2.py`, report
`reports/two-script-reads-v2.json`. "Untaught" above means never trained as numbers: see *Design*.

**Why.** Cross-script comparison is high (0.81–0.90 for Chinese ↔ Devanagari) and cross-script matching low (0.20–0.26). Three
explanations, not exclusive:
1. **Coarse placement.** Comparison is right for distant pairs even if each form is placed only within about one count; matching
   needs the exact count.
2. **Reader support.** `more` was trained on mixed pools of collections and taps; `same` always had a collection in its second
   slot, never a form.
3. **Matching runs through collections.** Cross-script matching succeeds about when both forms are identified as the right
   collection count (0.44 × 0.44 ≈ 0.2 against 0.24 observed).

**Reads.**
- **Comparison by distance:** the v1 comparison read split by |a − b| = 1, 2 and 3+.
- **Matching via collections:** each item's count identified through `same` against M (a count's score is its best participant's
  logit). Reported: identification per kind; *both correct* (the query and the gallery item of its count both identified as that
  count); a *mediated* pick (the gallery item whose softmax over count scores best matches the query's, by dot product); *direct*
  matching recomputed on the same items; their *agreement*; and direct accuracy split by *both correct*.

**Predictions.**
- Explanation 1 predicts cross-script comparison near chance at distance 1 and high at 3+, with the within-script and
  script–collection reads following the same profile.
- Explanation 3 predicts mediated ≈ direct, high agreement, and direct accuracy far higher when both are identified than when not.
- If direct exceeds mediated, the forms share some count identity that does not pass through collections.
- Explanation 2 is not separated by these reads: it would need a reader trained with forms in the second slot.

### Follow-up results

Run 2026-11-05: 159 s wall on 4 CPU workers, 520 s aggregate CPU. Below are `common-k-cand`'s numbers: the mean over seeds, with the
range of seed means in brackets. `gp` shows the same pattern; both arms are in the report.

**Identification over M** (chance 0.11):

| Point | Collections | Taps | English | Chinese | Devanagari | Thai |
|---|---|---|---|---|---|---|
| Block 3 | 0.91 | 0.98 | 0.13 | **0.21** | 0.96 | 0.13 |
| Delay | 0.94 | 1.00 | 0.14 | 0.22 | 0.79 | 0.12 |
| Relearn | 0.84 | 1.00 | 0.68 | 0.86 | 0.83 | 0.12 |

**Comparison by distance** (chance 0.5):

| Group | Point | 1 apart | 2 apart | 3+ apart |
|---|---|---|---|---|
| Chinese ↔ Devanagari | block 3 | 0.67 (0.60–0.75) | 0.86 (0.74–0.93) | **0.97** (0.93–1.00) |
| | delay | 0.67 | 0.81 | 0.94 |
| | relearn | 0.87 | 0.94 | 0.95 |
| Script ↔ collection | block 3 | 0.73 | 0.81 | 0.86 |
| Collections | block 3 | 0.99 | 1.00 | 1.00 |
| Thai × Thai | block 3 | 0.54 | 0.48 | 0.46 |

**Matching via collections** (chance 0.11):

| Group | Point | Direct | Mediated | Both identified | Agreement | Direct, both identified | Direct, not |
|---|---|---|---|---|---|---|---|
| Chinese ↔ Devanagari | block 3 | 0.24 | 0.25 | 0.21 | 0.83 | **0.76** | **0.11** |
| | delay | 0.27 | 0.26 | 0.20 | 0.74 | 0.75 | 0.15 |
| | relearn | 0.69 | 0.78 | 0.71 | 0.71 | 0.74 | 0.58 |
| Script ↔ collection | block 3 | 0.44 | 0.44 | 0.40 | 0.82 | 0.79 | 0.24 |
| Thai × Thai | block 3 | 0.63 | 0.47 | 0.13 | 0.53 | 0.67 | 0.63 |

**What this shows.**
1. **Explanation 1 holds.** Cross-script comparison is graded by distance: 0.67 for adjacent counts and 0.97 for counts 3 or more
   apart. Forms are placed in the right order but blurrily. Collections are placed sharply (0.99 even when adjacent).
2. **Comparison does not go through identification.** At block 3, Chinese forms are identified as the right collection count
   only 0.21 of the time. Yet a Chinese and a Devanagari form 3 or more apart are compared correctly 0.97 of the time. The order
   information is in the form's code without the form being matched to a count.
3. **Explanation 3 holds before relearning.**
   - Direct cross-script matching equals the mediated pick (0.24 against 0.25), and the two agree on 0.83 of queries.
   - When both forms are identified, direct matching is 0.76. When they are not, it is at chance (0.11).
   - So direct matching between scripts succeeds only through collections, and it is low because Chinese is barely identified.
     Nothing shows a form-to-form identity that bypasses collections.
4. **After relearning, the mediated pick beats direct matching** (0.78 against 0.69). The route through collections is then the
   better one.
5. **Thai × Thai matching is glyph resemblance, confirmed.** It does not depend on identification (0.67 against 0.63), and the
   mediated pick is far lower than the direct one.
6. **Explanation 2 is not separated,** as stated in the predictions.

## Identity from order (2026-11-05)

Written before the run. Same checkpoints, arms, seeds and items; `experiments/two_script_reads_v3.py`, report
`reports/two-script-reads-v3.json`. The user asked for it: *make the system derive something it has forgotten through other
relations.*

**Why.** At block 3 Chinese forms are identified over M only 0.21 of the time, yet they compare correctly with collections
(0.73 adjacent, 0.86 at 3+ apart). If a form's place in the order survives, its count may be recoverable from its comparisons with
collections the system still holds sharply, without the lost `same` pairing.

**Order profile.** For an item x and each count c, s_c(x) is the mean over M's participants of count c of
½[more(x, m) − more(m, x)], the antisymmetrized `more` logit. Positive means "x is above c". Nine numbers per item, read from the
trained `more` reader only.

**Two derivations.**
- **Placement (fixed rule, diagnostic).** n̂ = argmax_n [Σ_{c<n} s_c − Σ_{c>n} s_c]: the count most consistent with the profile.
  A hand-written rule over the system's outputs, so a diagnostic, not part of the system.
- **Learned read (neural).** A linear softmax read from the 9-number profile to the count, fitted per checkpoint only on kinds
  whose identity is present: collections and taps (first half of their items; the second half is scored). No script item is
  used in fitting. Scored on every kind.

**Reported per kind and point.** `same` identification (as in the follow-up), placement and learned-read accuracy, accuracy
within one count, and the **recovered share**: derived accuracy on the items `same` misidentifies (chance about 1/8).

**Predictions.**
1. **[derived]** Chinese at block 3: derived identity clearly above `same`'s 0.21, mostly within one count (adjacent comparison is
   only 0.73), so exact accuracy roughly 0.35–0.5 and within-one roughly 0.8 or more.
2. Devanagari, already identified, gains little; derived identity may fall below `same`.
3. Thai stays near chance: its forms have no order (Thai × Thai comparison 0.46–0.54).
4. If the learned read matches placement, the information is usable by a learned read trained only where identity is present;
   if it is far below, the profile of forms differs in shape from that of collections.

**Prior work** ([familiar]). Ranking from pairwise comparisons (Borda and Copeland counts, Bradley–Terry); placement is a
Borda-style score. The **symbolic distance effect** (Moyer and Landauer 1967): comparison accuracy rises with distance, as found
above, and is the classic sign of an analogue order code rather than stored pairings. **Transitive inference** (in animals and
networks): order can be derived for pairs never trained together. Known failure: with a compressed order code, placement is
biased toward the middle counts; end counts are easier.

**Cost.** Per checkpoint, 6 kinds × 9 counts × 30 items × 72 participants × 2 directions ≈ 233k `more` reads (~15k FLOPs each,
about 3.5 GFLOP), and a 9 × 9 softmax read fitted by L-BFGS on 540 items. 16 tasks × 3 points ≈ 170 GFLOP. No new state kept.

### Identity-from-order results

Run 2026-11-05: 158 s wall on 4 CPU workers, 513 s aggregate CPU; the one-task check took 60 s. `common-k-cand`, mean over seeds
(range of seed means). Collections and taps are scored on their held-out half. Chance is 0.11 exact and about 0.31 within one count.

| Kind | Point | `same` | Placement | Within one | Recovered | Learned read | Within one | Recovered |
|---|---|---|---|---|---|---|---|---|
| Collections | block 3 | 0.93 | 0.97 | 1.00 | 0.65 | 0.96 | 1.00 | 0.67 |
| Chinese | block 3 | **0.21** (0.10–0.38) | **0.19** (0.09–0.31) | 0.65 (0.47–0.82) | **0.05** | 0.22 | 0.65 | 0.09 |
| | delay | 0.22 | 0.20 | 0.55 | 0.06 | 0.21 | 0.54 | 0.06 |
| | relearn | 0.86 | 0.72 | 0.99 | 0.49 | 0.65 | 0.90 | 0.42 |
| Devanagari | block 3 | 0.96 | 0.85 | 0.99 | 0.79 | 0.89 | 1.00 | 0.79 |
| | delay | 0.79 | 0.68 | 0.95 | 0.23 | 0.73 | 0.91 | 0.27 |
| English | block 3 | 0.13 | 0.14 | 0.31 | 0.05 | 0.13 | 0.33 | 0.07 |
| Thai | block 3 | 0.13 | 0.14 | 0.34 | 0.02 | 0.13 | 0.34 | 0.02 |

`gp` is the same: Chinese at block 3 is 0.19 by `same`, 0.16 by placement, 0.62 within one, recovered 0.06.

**Placement of each count's mean Chinese profile** (the count the average form of each true count is placed at):

| True count | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Block 3 | 2 | 3 | 4 | 5 | 6 | 5 | 6 | 6 | 7 |
| Relearn | 2 | 3 | 4 | 4 | 5 | 6 | 7 | 8 | 9 |

**What this shows.**
1. **Prediction 1 fails.** Order does not recover Chinese identity: placement 0.19 and the learned read 0.22, against `same`'s
   0.21. On the items `same` gets wrong, order gets 0.05, *below* chance. The two reads fail on the same items, the same way.
2. **The Chinese order is kept but miscalibrated against collections.** The mean profiles rise monotonically with the true count
   (every column increases from count 1 to 9 apart from one dip at 6), so the forms are ordered among themselves. But the line is
   shifted up by one at counts 1–5 and compressed at 6–9 (7, 8, 9 placed at 6, 6, 7). A Chinese 1 is judged above a collection
   of one. This is why Chinese ↔ Devanagari comparison works: both lines are ordered, and a comparison between them only needs
   relative order. Matching and placement both need the line's alignment to collections, and that is what was lost.
3. **Within one count is well above chance** (0.65 against 0.31), consistent with a shifted line rather than lost order.
4. **The learned read equals placement** (prediction 4): the profile is usable, but what it carries is the miscalibrated line.
5. **Devanagari and Thai behave as predicted** (2 and 3). Devanagari's count 1 is placed at 2 (0.07 correct), the same end effect.
6. **What was forgotten is an alignment, not a set of pairings.** One monotone map between the Chinese line and the collection line
   would restore most of identity. Relearning English seems to do roughly that for Chinese (the relearn row is almost aligned).
   That is a structural target a consolidation step could aim at: realign a whole kept order with the collection line from a few
   anchors, rather than relearn each count.

**Not tested.** Identity from rank *within* the script (Chinese × Chinese comparisons plus one anchor), which would use only the
kept order and not its alignment.
