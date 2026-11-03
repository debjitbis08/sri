# Fixing centre and gate drift, including value decoding

Status: **pilot, 2026-11-04, seeds 392 and 393, both orders. Not a contract.** This follows the [adapt pilot](sense-adapt-v1-pilot.md), where single kinds lost up to 0.16 as later kinds arrived, and the corrections an external reviewer gave to the proposed fix. Code: `experiments/sense_stable.py`, with tables from `experiments/sense_stable_summary.py`. Report: `reports/sense-stable-v1-pilot.json.gz`.

## Question

When sight adapts by opening a centre or refitting the gate, can memory keep what it already holds exactly? That covers both how a query finds its unit and how the unit's value is decoded. Persistent learning needs this: a fact that read correctly yesterday must not degrade because sight learned about something else.

## Design

Facts, streams, the shift rule and the gate refits are exactly those of the adapt pilot's `found/online` arm.

| Arm | Units | Query | Value decoding | Gate |
|---|---|---|---|---|
| `online` | one memory | coded on its own nearest centre | candidates coded on their own nearest centre | current |
| `kept` | partitioned by the centre each fact was written under | coded on **every** centre, and each centre's units scored in that centre's coordinates; best unit wins | candidates coded on the **winning unit's** centre | current |
| `kept+frozen` | as `kept` | as `kept` | as `kept` | each centre keeps the gate as refitted when it opened: the query-dependent function, not an average weight |

**One centre per fact, for both sides.** A fact's cue picks its centre, and its target is coded on the same centre. So a unit's key and value share one coordinate system, and the value is decoded in it.

**Writes never mix coordinate systems.** Units are partitioned by centre, and a write can merge only into units of its own centre. Each partition holds up to 2,048 units, and no write was forced in any run.

**All centres are searched,** so the centre a fact was written under can never be skipped.

**Invariant expected of `kept+frozen`.** An unchanged query scored against an unchanged old unit keeps exactly the same score and the same value decoding when centres open or the gate refits.

**Two reads per stage** separate drift from interference:
- **pre:** after the kind's passing inputs have opened centres and refitted the gate, before its facts;
- **post:** after its facts.

Comparing pre at stage s with post at stage s−1 is **drift**: sight changed, and nothing was written in between. Comparing post with pre at the same stage is **interference**: new units were written, and sight did not change.

**Measures.** All are per query, at every stage. They don't use the matched threshold, which is recalculated at each stage:
- the score against the best unit that holds the query's own fact;
- the best unit, identified by (centre, unit);
- whether the value decodes correctly;
- acceptance at τ;
- unseen inputs' best scores and acceptance.

## Results

**Reference check.** `online` equals the adapt pilot's `found/online` in 168 of 168 reads, at every stage.

**Drift.** Counts are summed over all stages, kinds and views in four runs, 5,400 query reads per arm. Score changes are the largest single change.

| Arm | Own-fact score | Best score | Best unit changed | Value flips | Acceptance flips | Unseen score | Unseen acceptance flips |
|---|---|---|---|---|---|---|---|
| `online` | 0.48 | 0.47 | 301 | 133 | 143 | 0.35 | 41 |
| `kept` | 0.25 | 0.25 | 250 | 72 | 104 | 0.15 | 29 |
| **`kept+frozen`** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

**Interference.** Same counts, 5,400 query reads.

| Arm | Own-fact score | Best score | Best unit changed | Value flips | Acceptance flips | Unseen score | Unseen acceptance flips |
|---|---|---|---|---|---|---|---|
| `online` | 0 | 0.29 | 69 | 11 | 2 | 0.23 | 0 |
| `kept` | 0 | 0.17 | 44 | 5 | 0 | 0.20 | 0 |
| `kept+frozen` | 0 | 0.22 | 48 | 6 | 2 | 0.20 | 0 |

Later facts never changed an old unit's score: no write merged into a unit holding an older fact. In 48 of 5,400 reads a new unit outscored the old correct one, all on strong views.

**End of the run against just after teaching.** Mean over the kinds taught before the last, as a range over runs, and the worst single kind:

| Measure | `online` | `kept` | `kept+frozen` |
|---|---|---|---|
| Mild, matched | −0.04 to 0.00 (worst −0.16) | −0.02 to +0.01 (−0.09) | **0.00 (0.00)** |
| Mild, value at τ | −0.04 to 0.00 (−0.20) | 0.00 to +0.01 (−0.04) | **0.00 (0.00)** |
| Mild, value answering all | −0.03 to −0.01 (−0.13) | −0.01 (−0.07) | **0.00 (0.00)** |
| Strong, matched | −0.05 to +0.02 (−0.11) | −0.02 to +0.02 (−0.09) | −0.01 to 0.00 (−0.04) |
| Strong, value at τ | −0.02 to 0.00 (−0.09) | −0.02 to +0.02 (−0.09) | **0.00 (0.00)** |
| Strong, value answering all | −0.04 to +0.01 (−0.16) | −0.01 to +0.01 (−0.07) | −0.01 to 0.00 (−0.04) |
| Unseen at τ, mild and strong | up to +0.02 | up to +0.04 | **0.00** |

**Final reads,** mean over kinds, range over runs:

| Arm | Mild matched | Mild τ | Mild unseen at τ | Strong matched | Strong τ | Strong unseen at τ |
|---|---|---|---|---|---|---|
| `online` | 0.67–0.76 | 0.67–0.74 | 0.11–0.13 | 0.35–0.40 | 0.16–0.20 | 0.03–0.05 |
| `kept` | 0.68–0.79 | 0.70–0.77 | 0.13–0.15 | 0.36–0.40 | 0.18–0.21 | 0.05–0.06 |
| `kept+frozen` | 0.70–0.77 | 0.70–0.78 | 0.13–0.15 | 0.36–0.40 | 0.20 | 0.06 |

## What this shows

- **Keeping each fact's centre and freezing a gate per centre removes drift completely.** Over 5,400 query reads, no score, unit choice, value decoding or acceptance changed when centres opened or the gate refitted, and nothing changed for unseen inputs either. The invariant holds exactly, not approximately.
- **Each fix alone is not enough.**
  - Keeping centres alone halves the drift.
  - The current gate still moves query weights, so 72 value decodings still flip.
- **The remaining loss is interference, and it is small.** New facts' units outscore an old correct unit in 48 of 5,400 reads, all on strong views. That costs up to 0.04 on a single kind's strong reads and nothing on mild reads.
- **Recall improves too.** Final mild reads are 0.70–0.77 against 0.67–0.76, and 0.70–0.78 at τ against 0.67–0.74, because no fact is lost to drift.
- **The cost is slightly more false answers and slower reads.**
  - Searching every centre's coordinates lets unseen inputs find a match a little more often: 0.13–0.15 at τ against 0.11–0.13 mild, 0.06 against 0.03–0.05 strong.
  - Reads take 0.35–0.53 ms per query against 0.14–0.16 ms. That is still small beside encoding, at 0.075 s per input for both models.

## What this does not establish

- **Interference control.** New units can still beat an old correct match. That is the general capacity problem of any memory, and nothing here addresses it.
- **Fresh seeds, or streams where kinds interleave.** The shift rule still assumes each kind arrives for a while at a time, and in seed 393 it still forms one mixed centre at the start.
- **That the stored state stays bounded.**
  - Every centre and gate snapshot is kept forever.
  - The kept gate codes grow by about 3.1 MB per centre and make up most of the 24–27 MB of state in these runs. The units take about 6 MB.

## Prior work

These are familiar methods, not re-read for this pilot.

- **Fixed coordinates per stored item:** backward-compatible representation learning (Shen et al. 2020) keeps old embeddings valid by constraining new ones. Here nothing is retrained; each unit simply keeps the coordinates it was written in.
- **Per-task normalisation and heads that are never overwritten:** for example multi-head continual learning, and parameter isolation such as PackNet and Progressive Networks. The frozen gate per centre is a small form of this.
- **Searching several coordinate systems and taking the best score:** mixture-of-experts retrieval and multi-index search. The rise in false answers matches the known hubness and score-calibration issues of comparing scores across indexes; CSLS (Conneau et al. 2018) is a candidate correction.

## Scaling

For a query, with k centres, h units in total, d the joined width and a gate of fixed size:
- k centred codes and k gate evaluations cost O(k d);
- scoring every unit once in its own centre's coordinates costs O(h d);
- so a read costs O(h d + k d), not O(k h d).

Value decoding codes candidates on one centre, the winner's. Stored state grows as O(k) for centres and gate snapshots (k × (768 floats + 770 parameters)), plus the kept gate codes (2 × 512 × 768 floats per centre unless capped), plus O(h d) for units.

## Computation

GTX 1650, two workers. 2,907 s wall, 5,752 s aggregate CPU, peak RSS 2.7 GB. No model was trained. The gates have 770 parameters, with 300 steps per refit and 6–7 refits per run, as in the adapt pilot. Read latency per query, excluding encoding: `online` 0.14–0.16 ms, `kept` and `kept+frozen` 0.35–0.53 ms. Stored state per run is 23.7–27.0 MB, the same for all three arms because the units total about the same.
