# Number systems meta v1: pilot results

Date: 2026-10-28. Specification: [number systems meta v1](number-systems-meta-v1-spec.md)
(draft; not frozen). Command: `NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta pilot`.
Artifacts: `reports/number-systems-meta-v1-pilot-results.json.gz`, `-runtime.json`;
checkpoints in gitignored `runs/`. One seed (100). **Pilot: descriptive, not headline
evidence.** No setting was tuned; all are the spec's defaults.

## Source training at equal budget

| Arm | Source steps | Episodes | Training encodings | Training CPU (s) | Labelled accuracy | Complementary | Cross-representation probe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ordinary | 3,000 | 0 | 5.376 M | 1,143 | 0.93 | 0.11 | 0.27 |
| ordinary + dev | 3,000 | 0 | 5.376 M | 1,184 | 0.87 | 0.11 | 0.24 |
| meta | 1,657 | 413 | 5.375 M | 1,182 | 0.90 | 0.06 | 0.23 |
| meta-I | 1,657 | 413 | 5.375 M | 1,150 | 0.93 | 0.09 | 0.18 |

- Budgets match in encodings, and training CPU time agrees within 4%.
- The episodes cost meta about 45% of its source steps. Source label accuracy held up;
  the cross-representation probe dropped slightly (0.23 against 0.27).
- **Episode losses barely fell.** Over training, the query loss went from 3.04 to 2.58,
  and the preservation loss (two sources after adaptation) from 6.06 to 5.76. Ten
  satellite updates at learning rate 1e-2 still wreck the sources at the end of training.

## Few-shot adaptation (interface mode, learning rate 1e-2, up to 200 updates)

Best accuracy within the budget, means over tree, german and thai_digits:

| Examples per number | No source | ordinary | ordinary + dev | meta | meta-I |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.17 | 0.23 | 0.22 | 0.24 | 0.25 |
| 8 | 0.22 | 0.33 | 0.34 | 0.33 | 0.38 |
| 32 | 0.31 | 0.55 | 0.55 | 0.61 | 0.61 |

At the episode budget (10 updates), means over image held-outs and k:

| | ordinary | ordinary + dev | meta | meta-I |
| --- | --- | --- | --- | --- |
| held-out accuracy | 0.13 | 0.15 | 0.19 | 0.22 |
| source retention (after / before) | 0.25 | 0.28 | 0.24 | 0.40 |

- **Meta adapts somewhat faster in the regime it was trained for.** At 10 updates it
  reaches 0.19–0.22, against 0.13–0.15 for the ordinary arms. At 32 examples per number
  its best accuracy is 0.61 against 0.55 for both ordinary arms, so the gain is not just
  the extra development data. The clearest single case: thai_digits at k = 32 reaches
  0.73 with meta, against 0.58–0.59 for the ordinary arms.
- **The gain does not depend on the shared order.** Meta-I matches or exceeds meta.
- **Retention fails everywhere.** Interface adaptation at learning rate 1e-2 cuts the
  sources' labelled accuracy to 0.17–0.34 of its starting value after 200 updates in every
  arm. No image run meets the retention requirement. The preservation term did not
  protect the sources for meta. Meta-I keeps more at 10 updates (0.40); with one seed I
  can't tell whether that is real.
- Hand (sequence medium, no episodes) reaches the criterion at k = 8 and 32 in every arm.
  Retention there is about 0.93, because only Morse shares the satellite.

## Cost to criterion

The criterion was fixed as the no-source accuracy at k = 32 after 200 updates:
- tree 0.37, thai_digits 0.39, hand 0.74;
- **german 0.16, which is degenerate.** No source barely learns German, so every arm
  meets it before any update. German cost to criterion is uninformative.

| Held out, k | No source | ordinary | ordinary + dev | meta | meta-I |
| --- | --- | --- | --- | --- | --- |
| tree, 8 | > 200 | > 200 | 50 | 50 | 30 |
| tree, 32 | 200 | 40 | 20 | 50 | 30 |
| thai_digits, 8 | > 200 | > 200 | > 200 | > 200 | 50 |
| thai_digits, 32 | 200 | 75 | 75 | 40 | 30 |
| hand, 8 | > 200 | 75 | 100 | 100 | 50 |
| hand, 32 | 200 | 75 | 100 | 50 | 30 |

Updates to criterion, accuracy only. With the retention requirement, every image cell is
> 200.

Meta-I is fastest in every row except tree at k = 32, where ordinary + dev is faster (20
against 30). It is also fastest on Hand, the medium without
episodes, so its advantage is not specific to the trained image satellite. One seed and
coarse scoring points do not support a ranking.

## Reading

- Within this pilot, the learning-efficiency objective did speed up adaptation a little at
  an equal budget. The data-matched control shows it is not just the development data. It
  did not help retention, and its benefit did not depend on compatible number order.
- The preservation term failed because the inner adaptation is too destructive. The
  episodes' preservation loss stayed near its starting value. The shared-satellite
  setting may need a smaller inner learning rate, a larger λ, or more episodes before the
  term can work. Those are the tuning choices the spec leaves to development
  representations.
- Shared number structure is still not doing measurable work in learning cost: meta-I
  ≥ meta here, and I14 ≥ C14 in Stage A.
- Design faults to fix before more runs:
  1. A no-source criterion breaks when no-source does not learn (German). Use a fixed
     accuracy level, set per representation from the development representations, or
     report updates to several levels.
  2. The scoring grid is coarse at low update counts.
  3. One seed.

## Cost

Wall 1,561 s:
- data 28 s;
- source phase 1,210 s, with 4 arms in parallel and about 1,200 s CPU each (episodes
  about 510 s of that in the meta arms);
- adaptation 322 s, 60 runs on 6 workers.

Worker CPU 6,670 s; largest worker peak RSS 1.9 GB. Selection cost: two smoke runs
(about 30 s each, one failing on a result-key bug). Using 6 workers instead of 11 kept
the adaptation phase short (about 5 min for 60 runs).
