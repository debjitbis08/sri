# Number construction v1, Part A: pilot results

Protocol: [number construction v1 spec, "Part A
protocol"](number-construction-v1-spec.md#part-a-protocol-pilot-fixed-before-running-2026-10-29).
These pilots were run autonomously while the researchers were away, with decisions
recorded in the spec. One seed each; **pilots, not headline evidence.**

## Pilot 1 (2026-10-29)

- **Command:** `NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_part_a pilot`.
- **Artifacts:** `reports/number-construction-v1-part-a-pilot-results.json.gz`, `-runtime.json`.
- **Withheld counts:** chinese 4 and 8; devanagari_digits 2 and 5.

### What went wrong

1. **The persistent-state arms are unstable.**
   - A-episode, A-episode+teach and A-episode-interleaved couple each form to the
     successor of the *previous form's coupled state*. Along an episode the successor is
     applied again and again, so states can grow geometrically.
   - A-episode's matching loss reached about 9,000 at update 600 and above 17,000 early
     in block 3. The teach and interleaved arms never learned to match (final losses
     2.2–2.8, against ln 9 = 2.20 for chance).
   - Their evaluations are at chance and are not interpreted.
2. **Training and test collections differ.**
   - Training episodes keep the dot radius fixed, so ink grows with count.
   - Test candidates are fixed-area dots, so ink is constant.
   - Chinese reached a training matching loss of 0.19–0.53 but only 0.43–0.45 isolated
     matching on test.
3. **1,000 updates per block is too short** for English words (A-order 0.16 and
   A-shuffled 0.15 at the end of block 1) and for Devanagari digits (training loss 1.2–1.6
   at the end of block 3).
4. **Blocked teaching through one shared satellite causes forgetting,** as the
   protocol anticipated. Chinese falls from 0.43–0.45 at the end of its block to
   0.09–0.11 after the Devanagari block. So the end-of-training evaluation mostly measures
   forgetting.

### What the stable arms show

Isolated collection matching at the end of each language's own block (chance 0.11):

| Arm | Chinese paired | Chinese withheld | Devanagari paired | Devanagari withheld |
| --- | --- | --- | --- | --- |
| A-order | 0.45 | **0.43** | 0.22 | 0.20 |
| A-shuffled | 0.43 | 0.25 | 0.22 | 0.18 |

- **Withheld Chinese forms, never directly paired with a collection, were matched almost
  as well as paired ones under episode order** (0.43 against 0.45). Their meaning came
  only through counting transitions. This is relational grounding through succession, as
  in the earlier cross-relation pilot.
- Shuffled presentation of the same evidence gave less for the withheld forms (0.25). One
  seed; a difference to check, not a result.
- **Held-out succession** (pairwise next-scores, end of training): collections 0.80 and
  0.68; Devanagari 0.76 and 0.80 (A-order and A-shuffled).
- **Two-sided relational context for withheld forms** (end of training, after
  forgetting): A-order 0.17 → 0.33, with the swapped-item control following the context
  equally (0.34). This repeats B1's propagation pattern.
- **Transfer** (thai_digits, 4 examples per count, 200 updates): chance in every arm
  (0.08–0.16).
- **Confidence hooks** are uninformative at these error rates (86–89% isolated errors).
  AUROC for predicting errors is 0.41–0.57 for every signal.

### Changes for pilot 2

Recorded in the spec before running; see Pilot 2 below.

## Pilot 2 (2026-10-29)

Amendments: [spec, "Part A pilot 2 amendments"](number-construction-v1-spec.md#part-a-pilot-2-amendments-2026-10-29-after-pilot-1-before-running):
- coupling to the grounded previous step;
- gradient clipping;
- half fixed-area training collections;
- 2,000 updates per block;
- full evaluation at every block's end.

Command: `NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_part_a pilot2`.
Artifacts: `reports/number-construction-v1-part-a-pilot2-results.json.gz`, `-runtime.json`.
Same withheld counts (chinese 4 and 8; devanagari 2 and 5).

### Persistent-state arms still fail

- **A-episode:** matching loss 34 at the start of block 2, and its *next* loss stays at
  9.8–11.3 through block 2.
- **A-episode+teach:** diverges in block 1 (losses 59, 125 and 189 at its end).
- **A-episode-interleaved:** never learns to match (2.0–2.7, against chance 2.20).

Grounded coupling and clipping did not fix the persistent-state formulation. Their
evaluations stay at chance and are not interpreted. So the core-as-teacher question is
still untested.

### Stable arms: grounding through succession replicates; order does not help

Isolated collection matching at the end of each language's own block (chance 0.11):

| Arm | English (all paired) | Chinese paired | Chinese withheld | Devanagari paired | Devanagari withheld |
| --- | --- | --- | --- | --- | --- |
| A-order | 0.20 | 0.45 | **0.38** | 0.23 | 0.27 |
| A-shuffled | 0.32 | 0.50 | **0.38** | 0.60 | 0.30 |

Pilot 1 (1,000 updates per block), for comparison: A-order 0.45 / 0.43 and A-shuffled
0.43 / 0.25 for Chinese paired / withheld.

- **Withheld Chinese forms, never directly paired, reach 76–84% of paired accuracy** in
  both arms. The meaning comes through counting transitions alone. With pilot 1, that is
  four stable-arm runs in which withheld Chinese forms clearly exceed chance (0.25–0.43).
- **Episode order gives no advantage** over shuffled presentation of the same evidence
  (0.38 against 0.38). Pilot 1's difference (0.43 against 0.25) did not replicate.
- **Devanagari** is mixed. A-shuffled learns its paired forms much better (0.60 against
  0.23), and the withheld forms lag (0.30).
- **Held-out succession** at the end of the Chinese block: Chinese 0.98–0.99,
  collections 0.92–0.93. **Reverse direction** (collection → Chinese form): 0.44–0.53.
- **English words learn slowly** (0.20–0.32 after 2,000 updates).
- **Forgetting stays severe.**
  - Chinese falls to 0.16–0.19 after the Devanagari block.
  - English falls to 0.06–0.11 after the Chinese block.
  - Transfer to Thai stays near chance (0.13–0.20).

### Relational context and confidence

These figures pool the withheld forms of both later languages. At the end of block 2,
the Devanagari ones are untrained, which dilutes them.

**Relational context at the end of the Chinese block:**
- two-sided context: isolated 0.26–0.29 → revised 0.62. The swapped-item control
  follows the context (0.53–0.58), the B1 propagation pattern again;
- one-sided context: energy side choice 0.62–0.68 against restricted recognition
  0.62–0.64, the B1-ambiguous pattern again.

**The first informative confidence signal: agreement.**

| Arm, block end | Accuracy when the isolated answer agrees with the two-sided revised answer | When they disagree | Agreement rate |
| --- | --- | --- | --- |
| A-order, Chinese | 0.58 | 0.06 | 0.28 |
| A-shuffled, Chinese | 0.62 | 0.07 | 0.28 |
| A-shuffled, Devanagari | 0.61 | 0.11 | 0.32 |

- When instinct and relational inference agree, the instinctive answer is usually
  right. When they disagree, it is almost always wrong.
- The classic signals stay uninformative (AUROC 0.37–0.60): margin, best score,
  energy and ‖f‖.
- Agreement needs the revision, so it cannot gate revision directly. It is, however, a
  natural **training target for a learned confidence predictor**, as the external review
  suggested. That predictor would learn, from pre-revision features, when instinct is
  likely to agree with the core.

## Cost (pilot 2)

- Wall 741 s on 5 workers (arms 711 s in parallel).
- Worker CPU 2,882 s: about 690 s of training for each persistent arm, 262 s for A-order
  and 457 s for A-shuffled.
- Largest worker peak RSS 1.4 GB.
- Selection cost: one smoke run (about 20 s).

## Cost (pilot 1)

- Wall 390 s on 5 workers (data 28 s, rendering 2 s, arms 360 s in parallel).
- Worker CPU 1,464 s: about 342 s of training for each episode arm, 132 s for A-order
  and 229 s for A-shuffled, and about 10 s of transfer each.
- Largest worker peak RSS 1.2 GB.
- Selection cost: one smoke run (16 s) and a renderer test fix.
