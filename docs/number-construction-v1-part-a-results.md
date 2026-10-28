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

Recorded in the spec before running.

## Cost (pilot 1)

- Wall 390 s on 5 workers (data 28 s, rendering 2 s, arms 360 s in parallel).
- Worker CPU 1,464 s: about 342 s of training for each episode arm, 132 s for A-order
  and 229 s for A-shuffled, and about 10 s of transfer each.
- Largest worker peak RSS 1.2 GB.
- Selection cost: one smoke run (16 s) and a renderer test fix.
