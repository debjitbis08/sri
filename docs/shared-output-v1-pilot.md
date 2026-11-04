# Shared output v1: pilot results

Status: **pilot of the [draft contract](shared-output-v1-contract.md), run 2026-11-04, seeds 300–303, four arms. Not
headline evidence. The pre-declared criteria fail, so the confirmation does not proceed as designed.**

- **Commands:**
  - `NFI_DEVICE=cuda SO_WORKERS=1 /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot reference`;
  - `NFI_DEVICE=cuda SO_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot pilot`;
  - tables: `.venv/bin/python -m experiments.shared_output_summary`.
- **Reports:**
  - `reports/shared-output-v1-pilot.json` and `-runtime.json`;
  - `reports/shared-output-v1-reference-check.json`.
- **Checkpoints:** `runs/shared-output-v1/pilot/` (not tracked).
- **Reference check:** `gp-reference` reproduces step 8 `joined-mlp-calibrated` at seed 300 in **2,455 of 2,455 measures**.
- **Audit:** in every run, 18,000 naming rows, all in block 3, and no name in any `next`, comparison or continuation row or
  candidate.

## Summary

- **Grounding every relational slot during training breaks the relations themselves.** In `gp-grounded` and `common`,
  continuation from *familiar* items falls to 0.18–0.42, against 0.88–0.94 in the raw arms. Composition, comparison of
  collections and every withheld-Devanagari read fall with it. N1 is lower than in `gp`, not higher.
- **Grounding at read time helps a raw-trained system.** In `gp` at block 3, a name replaced by its top participant before the
  readout continues at **0.77**, against 0.47 with its own code. This is the external diagnostic from stage 0, now on a fully
  trained run.
- **Name identification fades quickly after teaching stops.** In `gp`, names are identified at 0.82 at the end of block 3, 0.37
  after the delay and 0.21 after relearning. Their continuation follows: given a correct identification, `gp`'s direct N1 is
  0.55, 0.78 and 0.62 at the three points.
- **Sharing the scorer alone changes little.** `common-raw` − `gp` on N1 at delay is +0.05, across seeds −0.10 to +0.19.
  `common-raw` keeps the familiar reads and loses some withheld-Devanagari and English reads.
- **Every never-taught symbol is accepted at identification** (0.95–1.00) in every arm. The validation threshold on `same`
  does not reject unfamiliar inputs.

## Results

Mean over four seeds, with the range in brackets. Chance is 1/9 for recall. N1 is recall of the continuation, with
abstention below the frozen threshold counted as wrong. Grounded arms are read with hard grounding, raw arms directly.

**Naming and transfer**

| Read | Point | `gp` | `gp-grounded` | `common` | `common-raw` |
|---|---|---|---|---|---|
| Name identification, counts 2–8 | block 3 | **0.82** (0.76–0.90) | 0.59 | 0.39 | 0.59 |
| | delay | 0.37 | 0.23 | 0.24 | 0.24 |
| | relearn | 0.21 | 0.22 | 0.19 | 0.20 |
| N1, mean of directions | block 3 | 0.47 (0.42–0.57) | 0.21 | 0.25 | **0.53** (0.36–0.63) |
| | delay | 0.32 | 0.20 | 0.23 | 0.36 |
| | relearn | 0.22 | 0.11 | 0.22 | 0.29 |
| N1, name replaced by its top participant (raw arms) | block 3 | **0.77** (0.70–0.86) | — | — | 0.64 |
| | delay | 0.36 | — | — | 0.35 |
| N1 given correct identification | block 3 | 0.55 | 0.25 | 0.24 | 0.70 |
| | delay | 0.78 | 0.24 | 0.29 | 0.83 |
| Familiar (collection, collection) | block 3 | 0.91 | **0.18** | **0.29** | 0.88 |
| | delay | 0.93 | 0.35 | 0.42 | 0.94 |
| Never-taught symbols accepted | delay | 1.00 | 1.00 | 1.00 | 1.00 |

**Rejection, at delay**

| Read | `gp` | `gp-grounded` | `common` | `common-raw` |
|---|---|---|---|---|
| Names: correct item removed, abstains | 0.13 | 0.12 | 0.02 | 0.11 |
| Names: reverse completion accepted | 0.08 | 0.25 | 0.34 | 0.14 |
| Names: repeated count accepted | 0.16 | 0.28 | 0.42 | 0.21 |
| Names: correct completion accepted | 0.37 | 0.36 | 0.52 | 0.44 |
| Familiar: correct item removed, abstains | 0.49 | 0.03 | 0.04 | 0.44 |

**Earlier abilities, at block 3**

| Read | `gp` | `gp-grounded` | `common` | `common-raw` |
|---|---|---|---|---|
| Devanagari matching, paired | 0.97 | 0.92 | 0.89 | 0.86 |
| Devanagari matching, withheld | 0.99 | **0.13** | **0.13** | 0.59 |
| Taps → Devanagari, withheld | 0.92 | 0.09 | 0.09 | 0.67 |
| Composition, collections up 1 | 0.93 | 0.18 | 0.42 | 0.90 |
| Comparison, collections | 1.00 | 0.87 | 0.90 | 0.98 |
| Collection code spread, between counts / within | 19.2 | 4.4 | 10.4 | 17.5 |

After relearning, English matching (paired) is 0.68 in `gp`, against 0.46–0.50 in the other arms.

**Pilot criteria.**
- **Gates.** `gp` passes. `gp-grounded` and `common` fail both identification and familiar continuation. `common-raw` fails
  identification (0.59).
- **Transfer.** `common` − `gp` at delay is −0.09 (every seed negative): fail.
- **Rejection.** Removed-set false acceptance is +0.11: fail.
- **Prior abilities:** fail on most reads.
- **Attribution of N1 at delay.**
  - `gp-grounded` − `gp`: −0.12.
  - `common` − `gp-grounded`: +0.03.
  - `common-raw` − `gp`: +0.05.

## What this shows

1. **Grounding during training starves the code of relational gradient.**
   - Under the gradient rule, relational losses reach only the participants' states, never the items. Raw forms and taps are
     shaped by matching alone.
   - The earlier pilots found that the relational objectives arrange the shared code: transfer needs that arrangement
     ([sequence pilot](sequence-pilot-v1-results.md)).
   - Here the withheld Devanagari counts, which are never matched and are placed only by relational rows, read at 0.13 in
     the grounded arms, against 0.99 in `gp`. Collection codes are far less spread across counts (4.4 against 19.2), and names are identified worse (0.59 and 0.39
     against 0.82).
2. **Soft grounding makes the relational inputs blurred mixtures.**
   - Early in training, identification is weak, so every grounded state is a near-average of M. Continuation and composition
     then cannot be learned even for familiar items.
   - Grounded candidates make this worse: training candidates are mixtures, not items.
   - This is the contract's predicted failure "grounded states are not readable", in a stronger form than stated: familiar
     items fail too, not only names.
3. **The structure works as a read on a raw-trained system.**
   - Grounding a name at read time in `gp` lifts N1 from 0.47 to 0.77 at block 3.
   - This matches stage 0 and the derived bound: identification 0.82 × familiar continuation 0.91 is about 0.75.
   - **Substitutivity holds when it is applied to a code the relations have arranged, not when it replaces the arranging.**
4. **Identification is not retained.** Names are taught only in block 3 and never rehearsed. Their identification falls from
   0.82 to 0.37 over the 2,000 delay updates. Read-time grounding is bounded by it (0.36 at delay), so retention of names, not
   the readout, limits transfer after the delay.
5. **Abstention is not working.** Never-taught symbols are accepted at identification at 0.95–1.00. The thresholds come from
   familiar items, and nothing trains `same` to fall below them for unfamiliar inputs.

These are explanations consistent with the reads, not yet separated by a test. In particular, (1) and (2) are confounded in
the grounded arms.

## What this does not establish

- **Significance.** These are four pilot seeds.
- **Which of starvation and blurring causes the grounded failure.** Separating them would take arms such as hard grounding in
  training, raw candidates, or relational gradient passed to the raw item alongside the grounded read.
- **Memory-present reads.** They are not implemented yet.
- **Whether name identification can be retained** by rehearsal or memory.

## Possible next steps (not decided)

- **Train raw, read grounded,** as a contract arm: `gp` with grounding applied at read time. Stage 0 and this pilot both
  point there.
- **Diagnose the grounded-training failure before any remedy:** soft against hard grounding, grounded against raw
  candidates, and relational gradient to the raw item with and without grounding.
- **Retention of identification:** names in rehearsal, or held in memory as image-free traces, as in step 10.
- **Rejection of unfamiliar inputs:** training `same` with negatives from never-seen symbols, or a threshold relative to the
  best participant.

## Computation

| Phase | Wall | Notes |
|---|---|---|
| Reference check (1 run, GPU) | 193 s | |
| Pilot, 16 runs on 3 GPU workers | 2,144 s | 6,269 s aggregate worker CPU |
| Per run, `gp` / `common-raw` | 230–261 s | |
| Per run, `gp-grounded` / `common` | 507–576 s | about 2.2×, from re-reading M and grounding every relational input each update |
| Reads per run | 2–5 s CPU | 3 read points |

- **Memory:** peak RSS about 3.0 GB per worker (mostly the frozen-code table); GPU peak 358–370 MB per worker.
- **Reader parameters:** 13,003 for `gp` and `gp-grounded`, 13,025 for `common` and `common-raw`.
- **Selection and tuning:** none. The smoke run (20 updates per block, two arms, 242 s) is not part of the results.
