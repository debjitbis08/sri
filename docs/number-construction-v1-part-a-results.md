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

## Pilot 3 (2026-10-29): three seeds

Amendment: [spec, "Part A pilot 3 amendment"](number-construction-v1-spec.md#part-a-pilot-3-amendment-2026-10-29-after-pilot-2-before-running):
- scale-relative persistent coupling, after diagnosing a runaway in state magnitude;
- seeds 100–102.

Command: `NCA_SEED=<seed> NSY_WORKERS=5 .venv/bin/python -m experiments.number_construction_part_a pilot3`.
Artifacts: `reports/number-construction-v1-part-a-pilot3{,-seed101,-seed102}-results.json.gz`
and `-runtime.json`.

Withheld counts:

| Seed | Chinese | Devanagari |
| --- | --- | --- |
| 100 | 4, 8 | 2, 5 |
| 101 | 2, 5 | 3, 7 |
| 102 | 4, 7 | 3, 5 |

Seed 100 reproduces pilot 2's non-persistent arms exactly.

### Persistent-state arms: still failing, recorded as unresolved

- In full runs the scale-relative coupling did not hold. A-episode+teach reached a
  matching loss of about 75,000 in block 2 (seed 100).
- All three persistent arms stay at chance on every measure, in every seed.
- An earlier replay that looked stable used a different training stream.

### Stable arms across three seeds

At the end of each language's own block (chance 0.11); values for seeds 100 / 101 /
102, then the mean:

| Measure | A-order | A-shuffled |
| --- | --- | --- |
| English (all paired) | 0.20 / 0.21 / 0.20 (0.21) | 0.32 / 0.23 / 0.21 (0.25) |
| Chinese paired | 0.45 / 0.58 / 0.50 (0.51) | 0.50 / 0.54 / 0.58 (0.54) |
| **Chinese withheld** | 0.38 / 0.55 / 0.30 (**0.41**) | 0.38 / 0.53 / 0.47 (**0.46**) |
| Devanagari paired | 0.23 / 0.41 / 0.56 (0.40) | 0.60 / 0.60 / 0.37 (0.52) |
| **Devanagari withheld** | 0.27 / 0.45 / 0.48 (**0.40**) | 0.30 / 0.25 / 0.37 (**0.31**) |
| Held-out succession, Chinese / collections | 0.98 / 0.88 | 0.99 / 0.90 |
| Reverse direction (collection → Chinese form) | 0.48 | 0.51 |
| Chinese after the Devanagari block (retention) | 0.20 | 0.16 |
| English after the last block | 0.09 | 0.09 |
| Thai after 200 transfer updates | 0.12 | 0.11 |

- **Grounding through succession holds in all six stable runs.** Forms never directly
  paired with a collection are matched at 0.25–0.55 (chance 0.11). Means are 0.31–0.46,
  against 0.40–0.54 for paired forms of the same language.
- **Episode order gives no consistent advantage** over shuffled presentation of the
  same evidence. A-order is better on Devanagari withheld forms (0.40 against 0.31) and
  worse on Chinese ones (0.41 against 0.46). Seed variation (up to 0.25) is larger than
  the arm differences.
- **Forgetting and transfer** are unchanged.

### Relational context and confidence across seeds

At the end of the Chinese block, pooled withheld forms (Devanagari untrained):
- two-sided context: revised 0.59 / 0.55 against isolated 0.29 / 0.31 (A-order /
  A-shuffled), with the swapped item following the context (0.55 / 0.52);
- one-sided context: energy side choice 0.74 / 0.71 against restricted recognition 0.71 /
  0.69.

That is the same pattern as B1 and B1-ambiguous: relations propagate grounded meaning,
and add little beyond filtering when the item must decide.

**Agreement as a confidence signal, all six stable runs:**
- accuracy when instinct agrees with the relational answer: 0.56–0.62;
- when they disagree: 0.06–0.10;
- agreement rate: 0.26–0.30.

The margin is weakly predictive (AUROC 0.54–0.64). The SRI energy is consistently
*inverted* (0.36–0.44): higher energy goes with fewer errors here. So it is not usable
as a confidence signal without understanding why.

## Pilot 4 (2026-10-29): core as teacher by pseudo-label fast mapping

Design: [spec, "Part A pilot 4"](number-construction-v1-spec.md#part-a-pilot-4-core-as-teacher-by-pseudo-label-fast-mapping-2026-10-29-before-running).
Arm A-order+pseudo:
- at withheld steps with both neighbouring collections visible, the core infers the
  count by two-sided revision, without gradient;
- the withheld form's isolated state is trained to match a collection of the inferred
  count;
- the true count of a withheld form is never used.

Command: `NCA_SEED=<seed> NSY_WORKERS=3 .venv/bin/python -m experiments.number_construction_part_a pilot4`.
Artifacts: `reports/number-construction-v1-part-a-pilot4{,-seed101,-seed102}-results.json.gz`
and `-runtime.json`. The comparison is A-order from pilot 3 (same code path and seeds).

Values for seeds 100 / 101 / 102, then the mean:

| End of own block | A-order | A-order+pseudo |
| --- | --- | --- |
| Chinese paired | 0.45 / 0.58 / 0.50 (0.51) | 0.53 / 0.47 / 0.52 (0.51) |
| **Chinese withheld** | 0.38 / 0.55 / 0.30 (**0.41**) | 0.42 / 0.42 / 0.37 (**0.40**) |
| Devanagari paired | 0.23 / 0.41 / 0.56 (0.40) | 0.34 / 0.38 / 0.45 (0.39) |
| **Devanagari withheld** | 0.27 / 0.45 / 0.48 (**0.40**) | 0.37 / 0.37 / 0.55 (**0.43**) |
| Two-sided revision, withheld, end of Chinese block | 0.59 | 0.46 |
| Agreement: accuracy when agreeing / disagreeing | 0.60 / 0.08 | 0.55 / 0.10 |

- **Inferred counts used as teaching targets were right 55–76% of the time** (diagnostic
  only, by block and seed).
- **Learning from relational inference did not raise the withheld forms' isolated
  matching** (0.40 against 0.41 for Chinese, 0.43 against 0.40 for Devanagari, within
  seed variation).
- **It reduced the later benefit of two-sided revision** (0.46 against 0.59), plausibly
  because the noisy targets pull withheld forms toward wrong counts.

**Reading:**
- In this setup, the fast-mapping signal adds nothing beyond what succession training
  already supplies. Both carry the same relational information, and the pseudo-labels
  are noisy.
- A sharper test of learning from inference needs two things:
  1. targets filtered by confidence, for example only where instinct and the relational
     answer agree, the signal that predicted correctness here;
  2. a setting where the relational inference is the *only* route to the withheld
     form's meaning, with no succession loss on that form. Fast mapping would then be
     tested on its own.

## Status after pilots 1–4

- **Established in the stable arms (three seeds):**
  - forms never directly paired with collections acquire quantity meaning through
    counting transitions alone, at 80–100% of paired accuracy, recognized in isolation;
  - the learned successor carries grounded meaning one step;
  - agreement between instinct and relational inference predicts correctness strongly.
- **Not supported:**
  - an advantage of coherent episode order over shuffled evidence;
  - a gain from pseudo-label fast mapping.
- **Unresolved:**
  - persistent state between steps: every formulation tried diverged or failed to learn;
  - forgetting under blocked teaching with one shared satellite: problem 1, untouched;
  - transfer to a new language: near chance;
  - slow learning of English words.
- **Confidence (the researchers' question):** a usable signal exists. It is agreement
  with the core, available after revision. The next step is a predictor trained to
  anticipate it from pre-revision features, which then gates revision and the curiosity
  questions of Part C.

## Cost (pilot 4)

- Three seeds in parallel, 346–350 s wall each (one arm per seed).
- Worker CPU 281–283 s per seed; largest worker peak RSS 1.2 GB.
- Selection cost: one smoke run (about 10 s).

## Cost (pilot 3)

- Three seeds in parallel, about 2,100 s wall each.
- Worker CPU 7,681–7,761 s per seed. Slower than pilot 2 because 15 workers shared 12
  CPUs.
- Largest worker peak RSS 1.4 GB.
- Selection cost: two replay diagnostics, one of 2,000 updates and one of 2,300 updates
  (about 5 and 8 minutes).

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
