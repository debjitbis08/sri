# Limited-memory study v1: results

Design, arms, endpoints and decision rules: [proposal](limited-memory-proposal.md), written
before the runs. Run 2026-10-30 on the GPU (`NFI_DEVICE=cuda`, six processes). **A pilot.**
The three stages and their seeds:

| Stage | Seeds | Arms | Runs | Role |
|---|---|---|---|---|
| `main` | 300–309 | dense, sparse64, sparse48, mixed, mixed-h2000, mixed-h500, mixed-h125, mixed-reset | 80 | selection; descriptive |
| `fork` | 300–309 | four continuations (H = ∞, 2000, 500, 125) of each `mixed` end-of-block-3 parent | 40 | handoff test; descriptive (selection run's parents) |
| `confirm` | 310–319 | sparse48, mixed, mixed-reset, mixed-h125 | 40 | the readings are decided here alone |

- **Commands:**
  - `NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.limited_memory pilot {main,fork}`;
  - `NLM_SELECTED=mixed-h125 NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.limited_memory pilot confirm`;
  - `.venv/bin/python -m experiments.limited_memory_analysis pilot {main,fork,confirm}`.
- **Artifacts:** `reports/limited-memory-v1-{main,fork,confirm}-{results.json.gz,runtime.json,analysis.json,analysis-runtime.json}`;
  checkpoints and per-task records in `runs/limited-memory-v1-*` (not tracked).
- **Revisions:** `main` at `21211d0`, `fork` at `169f62a`, `confirm` at `8016741`. The runtime
  files say `dirty: true` because the stage's own output files were untracked while it ran.
- **Device:** GPU results are not comparable with the earlier CPU fan-in runs; every
  comparison below is within this device and these seeds.
- **Intervals:** unadjusted paired 95% t-intervals with the audit margins (0.05 for
  accuracies), as in the earlier pilots. "Inconclusive" is not "equivalent".

## Summary

**No reading in the proposal is met on the confirmation run. The one large effect, a handoff
in the fork, comes from the selection run only and has not been repeated.**

- **Selection.** All three decaying arms passed the taught-matching gate; the highest
  average-coordinate transfer (a) was `mixed-h125` (0.715; h500 0.711; h2000 0.671).
- **Confirmation (seeds 310–319).** `mixed-h125` against:
  - `mixed`: (a) +0.037 [0.001, 0.074] inconclusive; Chinese paired at block 2 −0.059
    inconclusive; structure-only Chinese +0.010 inconclusive.
  - `mixed-reset`: (a) +0.019 [−0.011, 0.049] equivalent. **Decay does not beat the reset
    control on the primary endpoint.**
  - `sparse48`: (a) −0.034 [−0.076, 0.008] inconclusive; taps → withheld Devanagari −0.310
    [−0.57, −0.05] worse; everything else inconclusive or equivalent.
- **Memory and structure both: not met.** It needs `mixed-h125` equivalent to or better
  than `sparse48` on taps → withheld Chinese and English comparison, and better on
  Chinese paired at its block. On seeds 310–319 the first is inconclusive (+0.010), the
  second equivalent, and the third inconclusive (+0.047 [−0.05, 0.15]). On the selection
  seeds the third was better (+0.130) and the first two held, so the reading was met
  there and did not repeat.
- **Greater reliance on structure: not met.** Structure-only Chinese against `mixed` is
  +0.010 inconclusive on confirmation. On the selection seeds it was +0.047 inconclusive.
- **Decay beyond reset: not met.** (a) is equivalent to `mixed-reset` on confirmation
  (and on selection, −0.000). On the selection seeds `mixed-h125` was better than
  `mixed-reset` on Chinese paired at block 2 (+0.142) and taps → withheld Chinese
  (+0.083); neither repeated (+0.037 and −0.002, inconclusive).
- **Fork handoff, repeated on fresh parents.** `fork-h125` is consistent with handoff in the
  full cohort and in the eligible subset on both the selection parents (seeds 300–309,
  7 of 10 eligible) and the fresh parents (seeds 310–319, 5 of 10 eligible). `fork-h500`
  and `fork-h2000` are not on either. This is the only effect of the study that is large
  against its interval and that repeated; see "Fork repeat".
- **Sparse beats mixed on the primary endpoint, in both cohorts.** `sparse48` has the
  highest (a) in both (0.781 selection, 0.763 confirmation), with every `mixed` arm
  below it.

## Main stage (seeds 300–309; selection, descriptive)

Means:

| Arm | (a) avg-coordinate | (b) Chinese paired, block 2 | (c) structure-only Chinese | G (relearning) |
|---|---|---|---|---|
| dense | 0.660 | 0.647 | n/a | 0.128 |
| sparse64 | 0.758 | 0.488 | n/a | 0.070 |
| sparse48 | 0.781 | 0.470 | 0.173 | 0.082 |
| mixed | 0.686 | 0.593 | 0.097 | 0.114 |
| mixed-h2000 | 0.671 | 0.611 | 0.070 | 0.092 |
| mixed-h500 | 0.711 | 0.592 | 0.134 | 0.092 |
| mixed-h125 | 0.715 | 0.600 | 0.144 | 0.092 |
| mixed-reset | 0.716 | 0.457 | 0.099 | 0.110 |

- **Structure beats dense memory on cross-sense transfer, and costs taught matching.**
  `sparse48` is above `dense` on (a) in the means (0.781 against 0.660) and below it on
  Chinese paired at its block (0.470 against 0.647). The `mixed` arms sit between the two
  in the means; `mixed` against `dense` is inconclusive on both (+0.026 and −0.054).
  `sparse48` against `dense` is not among the prespecified contrasts.
- **The decay half-life orders (a) weakly.** h2000 0.671, h500 0.711, h125 0.715, with
  `mixed` at 0.686. Against `mixed`, none is decisive (h125 +0.030 [−0.01, 0.07],
  inconclusive).
- **`mixed-reset` pays for its resets.** Chinese paired at block 2 is 0.457, below
  `sparse48` by 0.013 and below every decaying arm (h125 +0.142 [0.05, 0.23], better).
  Its (a) equals `mixed-h125`'s (0.716).
- **Effects on taps → withheld Devanagari** are the largest differences in the study:
  `sparse48` is 0.787 and `sparse64` 0.863, against 0.41 (`dense`), 0.36 (`mixed`) and 0.47
  (`mixed-h125`); `sparse48` is above `mixed-h125` by 0.313 [0.07, 0.55]. The largest of
  these gaps do not appear for the other transfer measures.
- **Relearning (descriptive).** Starting accuracy on English matching after the delay
  is about 0.08–0.10 in every arm and 0.20–0.30 after 200 relearning updates; G is
  not used to rank arms.
- **Disruption and silencing measures** are in the analysis files and are descriptive.

## Fork stage (seeds 300–309; `mixed` parents; descriptive)

Four continuations of 2,000 updates per parent, with the rehearsal buffer consulted for
all three languages. Differences are against the H = ∞ branch from the same parent.

- **Eligibility.** 7 of 10 parents were eligible (silencing the memory part lowers Chinese
  matching on the even items by more than 0.05: seeds 300, 301, 302, 306, 307, 308, 309),
  so the conditional reading is inferential. Silencing drops range from 0.02 to 0.23.
- **Results at 2,000 updates:**

| Branch | Structure-only Chinese (full cohort) | Memory dependence (full cohort) | Consistent with handoff |
|---|---|---|---|
| fork-h2000 | +0.019 [−0.013, 0.051] inconclusive | −0.011 [−0.045, 0.022] equivalent | no |
| fork-h500 | +0.039 [−0.020, 0.098] inconclusive | −0.038 [−0.109, 0.033] inconclusive | no |
| fork-h125 | +0.298 [0.161, 0.435] better | −0.331 [−0.466, −0.197] worse | yes |

  The eligible-subset results agree in direction: h125 +0.316 and −0.361; h500 +0.053
  [0.007, 0.099] (better) with dependence −0.053 inconclusive.
- **The rise is progressive.** Along `fork-h125` the structure-only difference is 0, 0.04,
  0.12, 0.24, 0.30 at 0, 500, 1,000, 1,500 and 2,000 updates, and the dependence
  difference 0, −0.12, −0.17, −0.30, −0.33. It is not a step at one checkpoint.
- **What this shows.** It is evidence consistent with handoff, not a demonstration:
  continued supervised training can teach the structure part directly, and the memory
  part's decay removes the alternative route. It is a result on one cohort's parents.
  The repeat on fresh parents is below.

## Fork repeat (seeds 310–319; `mixed` parents of the confirm stage; stage `fork-confirm`)

Run after the results above, with the protocol unchanged (2,000 updates, margin 0.05, at
least 5 of 10 eligible); `NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.limited_memory pilot fork-confirm`
(about 9 minutes), then `experiments.limited_memory_analysis pilot fork-confirm`. Runner
change committed as `94fae4a`; analysis `reports/limited-memory-v1-fork-confirm-analysis.json`.

- **Eligibility:** 5 of 10 parents (seeds 310, 311, 313, 314, 317), exactly the minimum, so
  the conditional reading is inferential but rests on five seeds.
- **Results at 2,000 updates (full cohort):**

| Branch | Structure-only Chinese | Memory dependence | Consistent with handoff |
|---|---|---|---|
| fork-h2000 | +0.020 [−0.018, 0.058] inconclusive | −0.030 [−0.101, 0.040] inconclusive | no |
| fork-h500 | +0.033 [−0.039, 0.106] inconclusive | −0.093 [−0.158, −0.028] lower | no |
| fork-h125 | +0.237 [0.093, 0.382] better | −0.309 [−0.443, −0.174] lower | yes |

  The eligible-subset results agree: h125 +0.229 [0.020, 0.437] and −0.335
  [−0.494, −0.176] (consistent with handoff); h500 dependence −0.091 [−0.182, −0.001]
  with structure-only +0.025 inconclusive.
- **Agrees with the selection parents** in sign and size for h125 (+0.298 and −0.331
  there). The rise again is progressive (full cohort, structure-only: 0, −0.008, 0.023,
  0.094, 0.237; dependence: 0, −0.026, −0.119, −0.170, −0.309), later than on the
  selection parents.
- **`fork-h500` now shows lower dependence without a rise in structure-only accuracy,** the
  proposal's "erasure without handoff" pattern. On the selection parents it was
  inconclusive on both. This is the one difference between the cohorts.
- **Limit:** the repeat uses a new cohort of seeds, but the same design, the same
  curriculum and the same fork protocol. It does not show that the transfer survives a
  change of rewiring rule (RigL picks new connections by gradient; random rewiring picks them at random), which the random-rewiring follow-up tests.

## Confirmation (seeds 310–319; the readings are decided here)

| Arm | (a) | (b) | (c) | G |
|---|---|---|---|---|
| sparse48 | 0.763 | 0.500 | 0.152 | 0.079 |
| mixed | 0.692 | 0.606 | 0.120 | 0.094 |
| mixed-reset | 0.711 | 0.510 | 0.128 | 0.094 |
| mixed-h125 | 0.730 | 0.547 | 0.130 | 0.086 |

Paired differences, `mixed-h125` against each arm (mean, 95% interval, outcome):

| Against | (a) | (b) Chinese paired | (c) structure-only | taps → withheld Chinese | English comparison | taps → withheld Devanagari |
|---|---|---|---|---|---|---|
| mixed | +0.037 [0.001, 0.074] inconclusive | −0.059 [−0.163, 0.044] inconclusive | +0.010 [−0.061, 0.082] inconclusive | −0.032 inconclusive | −0.004 equivalent | +0.150 inconclusive |
| mixed-reset | +0.019 [−0.011, 0.049] equivalent | +0.037 [−0.037, 0.110] inconclusive | +0.002 inconclusive | −0.002 inconclusive | −0.019 equivalent | +0.160 inconclusive |
| sparse48 | −0.034 [−0.076, 0.008] inconclusive | +0.047 [−0.053, 0.148] inconclusive | −0.022 [−0.068, 0.024] inconclusive | +0.010 inconclusive | −0.009 equivalent | −0.310 [−0.57, −0.05] worse |

- **Secondary measure that holds:** `mixed-h125` is better than `mixed` and than `mixed-reset`
  on Devanagari comparison (+0.100 [0.056, 0.144] and +0.079 [0.033, 0.125]). It was not
  among the prespecified readings, so it is reported, not a reading.
- **English matching after the delay** is equivalent across the arms (−0.011 [−0.023, 0.002]
  against `mixed`): none retains English better than the others.
- **Gain compensation and erasure without handoff** are not tested here: they need the
  memory-part dependence and norm measures, which are in the analysis files (descriptive).

## What the results support

- **Consistent with the proposal's expectations (descriptive):** sparse structure carries
  cross-sense transfer (the means order `sparse48` > `mixed` > `dense` on (a), and
  `sparse48` is far above the rest on taps → withheld Devanagari), and costs taught
  matching in the means.
- **Not supported:** that decaying memory in the mixed network improves transfer beyond a
  weight-reset control, or beyond `sparse48`. At this size the memory units give no
  decisive benefit on (a), and `mixed` is lower than `dense` on taught Chinese in the
  means (−0.054, inconclusive).
- **Supported (descriptive of a fork, not of the main endpoints):** at H = 125 the structure
  part takes over taught Chinese matching as the memory part decays, in two independent
  cohorts. It is evidence consistent with handoff, not a demonstration.
- **Open:** whether the pattern depends on RigL's gradient-based choice of new connections, as against random rewiring. The proposal's
  [follow-up](limited-memory-proposal.md) reruns the arms with random rewiring.
- **Alternative explanations not excluded:**
  - Shared nonlinear layers can make the silenced-memory drop an interaction, not stored
    knowledge (interaction residuals are in the analysis).
  - The decayed arm's weights restart toward their initial values, so lower dependence may
    reflect small memory-column norms rather than moved knowledge (norms are recorded;
    gain compensation is not separately tested here).
  - Absolute accuracies after the delay and at the start of relearning are low
    (0.08–0.10 English matching), so the endpoints sit near the bottom of their range.

## Computation

Complexity is as in the proposal: per run, 8,200 updates of an update cost about that of
the fan-in runner plus 16 extra dense reader units (time linear in updates and in
parameters; storage dominated by the data and the checkpoints).

| Stage | Runs | Wall (runs) | Effective per run | Peak RSS (child) | Checkpoints |
|---|---|---|---|---|---|
| main (selection, all 8 arms) | 80 | 4,399 s | 55 s | 2.55 GB | 511 MB |
| fork | 40 | 510 s | 12.7 s | 2.40 GB | 168 KB (records) |
| confirm | 40 | 2,222 s | 56 s | 2.55 GB | 279 MB |

- **Selection cost.** The `main` stage is the selection cost: 4,399 s of run time plus 28 s of
  data preparation and 3 s of rendering. The confirmation cost is 2,222 s plus the same
  preparation. The analyses each take under a second.
- **Per-run GPU memory:** about 90 MB allocated at peak; six processes share the 4 GB card.
- **Results files:** 1.3 MB (main), 60 KB (fork), 676 KB (confirm), gzip.
- Nondeterministic timing is in the `-runtime.json` files, separate from the analysis
  JSON, which holds the deterministic scientific content.
