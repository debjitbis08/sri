# Taught truth v1: results

Contract: [taught truth v1](taught-truth-v1.md), frozen at commit f22db02. Run 2026-11-06, seeds 440–443, all 12 runs
complete.

- **Commands:**
  - `EV_WORKERS=8 .venv/bin/python -m experiments.taught_truth main` (16.4 min wall);
  - `.venv/bin/python -m experiments.taught_truth_posthoc` (post-hoc, not in the contract; under 1 s).
- **Artifacts:**
  - `reports/taught-truth-v1{,-summary,-runtime}.json`;
  - `reports/taught-truth-v1-posthoc{,-runtime}.json`;
  - checkpoints in `runs/taught-truth-v1/` (not tracked).

## Summary

1. **Truth as learned values works as a mechanism.** One predictive reader learns:
   - every taught statement: individual T/F, kind quantities, anonymous-individual defaults and unknown colours, all at
     1.00;
   - the K3 connective tables (1.00);
   - filling the tail (hits@1 1.00).

   Truth, "don't know" and the quantities are codes it predicts, with no truth unit.
2. **"Don't know" is used where it is right, and not as an escape:**
   - held-out colours of colourless animals: unknown 0.64–0.65;
   - colours of elephant, flamingo and crow, which nothing tells the learner: unknown 0.56–0.58, instead of guessing;
   - inferable held-out facts: unknown 0.00.

   It is learned as a pattern of colour statements. It does not come from the reader's state of knowledge (point 4).
3. **Taught truth costs something on T/F inference** (against the built-in unit):
   - order 0.61–0.65 against 0.84;
   - grandparent 0.56–0.68 against 0.99;
   - hierarchy 0.61–0.84 against 0.86;
   - category from use 0.79–0.86 against 0.90.

   Inheritance, symmetry and inverse are within 0.03–0.06.
4. **The kind → individual link does not generalize.** New individuals known only by their kind get "never → F" right
   (0.77–0.97), but "sometimes → unknown" almost never (0.00–0.03):
   - the anonymous individuals of held-out kinds;
   - the dax ("a bird");
   - the held-out named animals.

   For held-out anonymous individuals the reader answers mostly F, whatever the kind statement says, and ranks unknown
   about 5th of 7. The link was learned per individual, not through the kind: individual → kind → value is a two-step
   chain, the failure v1–v3 kept finding.
5. **Kind quantities from members (aggregation)** reach 0.63–0.69 accuracy (mean recall over the 4 values 0.49–0.57;
   chance 0.25):
   - never 0.76–0.81 and sometimes 0.69 are learned;
   - always 0.30–0.40 and usually 0.19–0.38 are not.

## Results

Means of 4 seeds.
- **T/F tests:** balanced accuracy, where a positive counts only if predicted T and a negative only if predicted F.
  AUROC (bracketed) on logit(T) − logit(F).
- **Value tests:** accuracy against the expected value.

| Test | `unit` (built-in truth) | `pred` | `pred-flat` |
|---|---|---|---|
| **S1: T/F as before** | | | |
| P1 inheritance (`default`) | 0.86 (0.96) | 0.83 (0.92) | 0.80 (0.90) |
| P1 hierarchy | 0.86 (0.98) | 0.61 (0.84) | 0.84 (0.93) |
| P2 symmetry | 0.71 (0.83) | 0.68 (0.79) | 0.68 (0.79) |
| P3 inverse | 0.82 (0.92) | 0.80 (0.88) | 0.76 (0.87) |
| P4 transitivity | 0.84 (0.93) | 0.61 (0.75) | 0.65 (0.74) |
| P5 opposite from meaning | 0.48 (0.20) | 0.45 (0.33) | 0.50 (0.47) |
| P6 grandparent | 0.99 (1.00) | 0.68 (0.90) | 0.56 (0.85) |
| P6 same colour | 0.53 (0.57) | 0.52 (0.58) | 0.56 (0.58) |
| G1 word order | 1.00 (1.00) | 0.98 (1.00) | 0.97 (1.00) |
| G2 category from meaning | 0.68 (0.84) | 0.70 (0.82) | 0.78 (0.89) |
| G3 category from use | 0.90 (0.96) | 0.79 (0.92) | 0.86 (0.96) |
| G4 silly vs sensible | 0.56 (0.61) | 0.52 (0.58) | 0.52 (0.58) |
| Validation (trained kinds of statement) | — | 0.78 (0.90) | 0.79 (0.92) |
| **S2: kind quantities (held out, 39)** | — | 0.69 | 0.63 |
| always / usually / sometimes / never | — | 0.40 / 0.38 / 0.69 / 0.81 | 0.30 / 0.19 / 0.69 / 0.76 |
| **S3: default to held-out named animals** | — | 0.65 | 0.65 |
| always → T / usually → T / sometimes → U / never → F | — | 0.76 / 0.60 / 0.00 / 0.94 | 0.74 / 0.50 / 0.01 / 0.97 |
| **S3b: default to new anonymous individuals** | — | 0.55 | 0.60 |
| always / usually / sometimes / never | — | 0.42 / 0.34 / 0.00 / 0.77 | 0.33 / 0.25 / 0.03 / 0.86 |
| **S4: don't know** | | | |
| (a) held-out colourless animals → U | — | 0.64 | 0.65 |
| (b) P7 colours nothing tells → U | — | 0.58 | 0.56 |
| control: U on inferable held-out facts | — | 0.00 | 0.00 |
| **S5: the dax ("a bird")** | | | |
| always → T / usually → T / sometimes → U / never → F | 1.00 / 0.50 / — / 0.92 | 0.85 / 0.38 / 0.00 / 0.94 | 0.80 / 0.50 / 0.03 / 0.94 |
| other classes rejected | 1.00 | 1.00 | 1.00 |
| **Fit** | | | |
| individual / kind / anonymous / connectives / fill-the-tail hits@1 | 1.00 / — / — / — / — | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |

Parameters: `unit` 28,737; `pred` 33,809; `pred-flat` 23,409.

**Post-hoc (not in the contract):** what the reader answers on held-out anonymous individuals (`pred`, share of
answers):

| Expected (kind value) | T | F | other | Rank of U among 7 (mean) |
|---|---|---|---|---|
| always → T | 0.42 | 0.54 | 0.04 | 5.8 |
| usually → T | 0.34 | 0.62 | 0.03 | 5.3 |
| sometimes → U | 0.32 | 0.54 | 0.14 (U 0.00) | 4.6 |
| never → F | 0.11 | 0.77 | 0.12 | 5.1 |

On trained anonymous individuals every row is answered correctly (U rank 1.0 for "sometimes"). `pred-flat` is the same
pattern (F 0.56–0.86).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Taught truth costs nothing on T/F: `pred` within 0.05 of `unit` | **Fails** on order (−0.23), grandparent (−0.31), hierarchy (−0.25), category from use (−0.11). **Holds** on inheritance, symmetry, inverse, word order |
| 2 | Kind quantities ≥ 0.6; always/never easier than usually/sometimes | Accuracy 0.63–0.69 **holds**; mean recall 0.49–0.57. Never is easiest, as predicted; **always is hard** (0.30–0.40), sometimes is not (0.69) |
| 3 | Default read: always/never ≥ 0.9, usually ≥ 0.8, sometimes → U ≥ 0.5 | **Fails** except never → F (0.94–0.97). Sometimes → U 0.00–0.01 |
| 4 | Don't know: (a) ≥ 0.8; (b) ≥ 0.5; control ≤ 0.15 | (a) 0.64–0.65 (**fails**); (b) 0.56–0.58 (**holds**); control 0.00 (**holds**) |
| 5 | The dax follows the defaults for always and never | **Holds** for never (0.94) and for always in `pred` (0.85); usually 0.38–0.50; sometimes → U 0.00–0.03 |
| 6 | Connectives fitted exactly | **Holds** (1.00) |

## Interpretation

- **[read: results] Taught truth works as a representation; the reasoning that uses it is the problem.** Every taught
  statement is learned, including the algebra. What fails is the same as before: anything that must pass through a middle
  entity. "A reptile sometimes swims, so this reptile's swimming is unknown" is individual → kind → kind value → default.
  The reader cannot route through the kind, so it answers from similarity to trained items, which says mostly F.
- **[read: results] "Don't know" is learned as content, not as a state.** U is given where it was taught: colour
  statements about colourless animals. It is not given where the reader lacks information (new individuals with
  "sometimes" properties). A U that reflects the reader's own knowledge would need the reader to know what it has
  inferred. That points at Stage 2: a loop that fetches the kind and reads its value, and could end in U when nothing is
  found.
- **[derived, untested] Why taught truth costs on order and grandparent.** The built-in unit gives every relation's
  yes/no a dedicated, shared output direction. With taught truth, each answer is a code matched against 7 candidates. The
  reader must also keep "individual" and "kind" values apart (an individual answering "usually" is wrong), and some
  capacity goes to that. The drop is largest on the tests that were already the hardest inferences (order, composition).
  This is not tested here.
- **[read: results] Always is hard for kinds, sometimes is not.** From members, "sometimes" (some yes, some no) is a local
  pattern, and "never" is the majority. "Always" requires every member to agree, which a reader that saw a few
  members cannot be sure of. That is close to the brief's point that always quantifies over all admissible cases.

## Computation (measured)

- 12 runs on 8 workers: 16.4 min wall; 6,873 s aggregate worker CPU (9.5 CPU min per run; projected 5). The fill-the-tail
  rows over all 225 entities and the 7-way value loss each step are more than the projection allowed for.
- Peak RSS per worker is under 400 MiB.
- Post-hoc: 0.1 s.
- Smoke: 13 s wall.
- No tuning or selection runs.

## Next (for the user)

Stage 2, `loop-gen` on this reader ([multistep-reader-v1](multistep-reader-v1.md) to be revised): each step predicts a
code (the kind, the named animal, the parent), and the last step predicts the truth value. Its first test should be the
failure measured here: S3b and the dax's "sometimes → unknown", which needs exactly one fetch (the individual's kind).
Next come the chains of v1–v3 (category from meaning, silly phrases, grandparent, and the longer closures).
