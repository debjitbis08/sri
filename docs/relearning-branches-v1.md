# Can the core keep a learned relation through new learning? Relearning branched from one state

Status: **diagnostic continuations, 2026-11-04, seeds 300–303. Not a contract.** Follows the [same-relation diagnosis](same-relation-diagnosis-v1.md). Code: `experiments/relearning_branches.py`, with tables from `experiments/relearning_branches_summary.py`. Report: `reports/relearning-branches-v1.json.gz`.

**The question (user, 2026-11-04):** can the core preserve a learned relation through new learning? Explicit reverse-direction training was deferred, because the branches investigate why learned reverse validity breaks without building the answer into training.

## Method

Each seed of step 8's calibrated core was rerun exactly as limited-memory main and stopped at the first relearning update. Its full state was captured there:
- the weights and Adam's state;
- every random stream;
- the rehearsal buffer;
- the first relearning batch;
- the step counter and torch's random state.

**Both reproduction checks are exact in every seed:**
- the captured weights equal step 8's delay checkpoint (maximum difference 0);
- the unchanged branch ends equal to step 8's relearning checkpoint (maximum difference 0).

**Three branches of 200 updates** start from that state, each changing one thing:
- `as-run`: relearning as run, with English form episodes, rehearsal of earlier scripts, and taps–collection context episodes.
- `no-forms`: the delay's material continued instead, so context episodes only, with no forms.
- `frozen-image`: relearning as run, with the image projection (shared by collections and forms) not updated. Collection codes can still move through the core.

**Every 10 updates,** all reads use identical pairs and candidates:
- the `same` relation's symmetry, pair discrimination, ranking and geometry;
- collections → taps on fresh and stored dot values, in both comparator orders;
- taps → collections;
- English matching, for reacquisition;
- memory and core together, through stage 2's memory.

Cells are the mean over seeds, with the range in brackets.

## Results

| Read | Branch | 0 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| Collections → taps, fresh | `as-run` | 0.89 (0.84–0.94) | 0.26 (0.11–0.39) | 0.42 | 0.45 | 0.67 (0.49–0.81) |
| | `no-forms` | | 0.88 | 0.92 | 0.87 | 0.89 (0.86–0.93) |
| | `frozen-image` | | 0.39 (0.17–0.66) | 0.42 | 0.66 | 0.61 (0.39–0.79) |
| Collections → taps, stored values | `as-run` | 0.94 | 0.26 | 0.44 | 0.49 | 0.73 (0.52–0.85) |
| | `no-forms` | | 0.93 | 0.95 | 0.91 | 0.95 (0.92–0.97) |
| | `frozen-image` | | 0.41 | 0.45 | 0.71 | 0.63 (0.46–0.82) |
| Taps → collections (trained) | `as-run` | 0.99 | **0.50 (0.29–0.60)** | 0.94 | 0.96 | 0.97 |
| | `no-forms` | | 0.98 | 0.99 | 0.92 | 0.97 |
| | `frozen-image` | | 0.57 | 0.77 | 0.93 | 0.95 |
| Symmetry: same-count violation | `as-run` | 0.10 | 0.59 | 0.44 | 0.33 | 0.20 |
| | `no-forms` | | 0.10 | 0.08 | 0.10 | 0.10 |
| | `frozen-image` | | 0.43 | 0.41 | 0.21 | 0.32 |
| Collection class spacing | `as-run` | 26.7 | 8.9 | 6.5 | 6.7 | 8.2 |
| | `no-forms` | | 27.6 | 27.6 | 27.2 | 27.7 |
| | `frozen-image` | | 13.4 | 5.6 | 5.2 | 5.4 |
| Same-count collection–taps distance, over spacing | `as-run` | 0.28 | **1.46** | 1.06 | 0.95 | 0.73 |
| | `no-forms` | | 0.29 | 0.27 | 0.28 | 0.28 |
| | `frozen-image` | | 0.96 | 1.06 | 0.98 | 1.15 |
| English matching (relearned) | `as-run` | 0.15 | 0.23 | 0.29 | 0.37 | 0.54 (0.12–0.74) |
| | `no-forms` | | 0.15 | 0.15 | 0.15 | 0.14 |
| | `frozen-image` | | 0.17 | 0.23 | 0.29 | 0.34 (0.11–0.51) |
| Memory present: set A, symbol → taps | `as-run` | 0.76 | 0.19 | 0.36 | 0.38 | 0.60 (0.42–0.70) |
| | `no-forms` | | 0.76 | 0.78 | 0.75 | 0.77 (0.72–0.84) |
| | `frozen-image` | | 0.32 | 0.36 | 0.58 | 0.52 (0.37–0.69) |
| Memory present: set A, taps → symbol | `as-run` | 0.81 | 0.41 | 0.79 | 0.81 | 0.81 |
| | `no-forms` | | 0.81 | 0.79 | 0.76 | 0.82 |
| | `frozen-image` | | 0.48 | 0.68 | 0.81 | 0.79 |

Set B and the reads on stored values follow set A. Pair AUROC in the evaluated order:
- `as-run`: 0.98 to 0.73 at 20, recovering to 0.93;
- `no-forms`: 0.98–0.99 throughout;
- `frozen-image`: 0.88 at 200.

**The first updates' losses** (match, succession, comparison, sequence prediction), means over seeds:

| | Updates 0–2 | Updates 10–19 | Updates 190–199 |
|---|---|---|---|
| `as-run` | 11.9–12.8, 2.5–2.6, 0, 60–94 | 4.7, 1.9, 0.1, 8.5 | 0.8, 0.2, 0, 0.8 |
| `no-forms` | 0.00–0.01, 0.00–0.01, 0, 0.02–0.04 | 0.01, 0.01, 0, 0.03 | 0.07, 0.01, 0, 0.05 |

A 9-way matching loss of 12 is far above a uniform guess (2.2). The forgotten forms are confidently wrong.

## What this shows

**Wording corrected (2026-11-04, after review).** The relearning material includes English episodes and rehearsal of earlier scripts. These branches have not isolated English, and reverse performance does partly recover.

- **Reintroducing the form-relearning material triggers the decline in these branches.** That material is English episodes plus rehearsal of earlier scripts. Continuing the delay's material for the same 200 updates instead keeps every measured property:
  - the relation's symmetry and discrimination;
  - the code geometry;
  - collections → taps on fresh and stored values (0.89 and 0.95);
  - memory-present reads.
  It is not an effect of continued training as such.
- **The damage is an abrupt disruption at the start of relearning, then a partial recovery.**
  - Within 20 updates the code space shrinks threefold, and the two senses' count lines separate: same-count distance goes from 0.28 to 1.46 of the class spacing.
  - Even the trained direction, taps → collections, falls to 0.29–0.60.
  - These are the updates where the forgotten forms give matching losses near 12 and sequence losses of 60–94, against about 0.01 during the delay.
- **The trained direction recovers fast; the reverse recovers partly and more slowly.**
  - By 50 updates, taps → collections is back to 0.94, and its pair AUROC to 0.90.
  - Collections → taps recovers from 0.26 at 20 updates to 0.67 by 200, still rising. The senses' alignment recovers to 0.73 of the class spacing, against 0.28 before.
  - Within 200 updates the reverse does not regain its earlier level, while the trained direction does.
- **Freezing the image projection does not prevent the disruption, and does not help.**
  - With the shared projection fixed, collection codes still shrink and the senses still separate, through the parts of the core still training.
  - Collections → taps ends at 0.61 against 0.67, and English reacquisition is slower (0.34 against 0.54).
  - Updating the image projection is therefore not required for the damage.
  - This does not identify another single cause. The core's projection, the readers, the taps encoder and their interactions all keep changing in this branch.
- **Memory still holds the facts; what moves is the interpreter.** Memory-present symbol → taps falls and partly recovers exactly with the core's own collections → taps reading. Taps → symbol, which uses the trained direction, recovers to 0.79–0.81 by 50 updates in both relearning branches.

**Why does learning new forms break a learned relation?** These runs locate the damage in time and in the material:
- It comes from the first updates on the reintroduced form material, which the core gets confidently wrong.
- It is not specific to the image projection.
- The trained direction recovers within 50 updates; the reverse only partly within 200.

They do not establish the mechanism of the disruption. Two candidates consistent with this are not tested:
- the size of those first error signals;
- Adam's moment estimates, calibrated on the delay's near-zero losses, which turn large errors into large steps on every parameter at once.

## What this does not establish

- **The mechanism of the early disruption** (above), or which loss carries it. The sequence-prediction loss is the largest by far at the start.
- **Whether reverse validity would re-emerge with longer relearning.** Collections → taps is still rising at 200 updates in `as-run` (0.63 at 150, 0.67 at 200).
- **Explicit reverse training,** deferred.

## Next

These were approved with measurement first, and run: [relearning dynamics](relearning-dynamics-v1.md). The original recommendation:

The smallest discriminating continuations from the same captured states, one change at a time, with the same reads:
1. **Per-loss attribution of the disruption:** relearning as run without the sequence-prediction loss on form episodes, and separately without the form matching loss. Which error signal disrupts the relation?
2. **Update size against error size:** as run, but logging each parameter group's update norm over the first 20 updates. This shows where the disruption enters without changing training.
3. **A longer `as-run` continuation, 1,000 updates:** does reverse validity re-emerge under continued learning, or settle below its earlier level?

## Computation

- **GPU** (GTX 1650), three workers:
  - rerun to the capture point 155–205 s per seed;
  - three branches of 200 updates with 21 read points each, 12–16 s per branch;
  - 8 min 10 s wall in all, 1,084 s aggregate worker CPU, peak RSS 3.7 GB.
- **Training:** the only training was these continuations.
