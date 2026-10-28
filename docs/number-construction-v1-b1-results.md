# Number construction v1, B1: results on frozen pilot checkpoints

Date: 2026-10-28. Protocol: [number construction v1 spec, "B1
protocol"](number-construction-v1-spec.md#b1-protocol-fixed-before-running-2026-10-28),
fixed before running, plus the swapped-item control added after the smoke run.

- **Command:** `NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_b1 pilot`.
- **Artifacts:** `reports/number-construction-v1-b1-pilot-results.json.gz`, `-runtime.json`.
- **Scope:**
  - frozen checkpoints: number systems v1 C14 and I14 at step 3,000, and the four meta
    pilot arms;
  - one seed; 20 items per number per representation reported (half B), 20 used for
    selection (half A);
  - descriptive, **not headline evidence**.
- **What B1 can test.** These checkpoints never saw counting sequences, so B1 tests
  whether the existing successor carries usable relational information. It does not
  test acquisition.

The solver reproduces the experiment's core states to within 10⁻⁶ in every checkpoint.

## Main result

Collection matching against 9 fixed-area dots candidates (chance 0.11). Interior counts
2–8, representations other than dots, half B. Context is collections (dots of n ± 1,
meanings inferred) with the true successor, at the μ selected on half A.

| Checkpoint | Order | Immediate | Always revise | Withheld only: immediate → revised | Oracle-context bound |
| --- | --- | --- | --- | --- | --- |
| v1 C14 | compatible | 0.21 | 0.66 | 0.22 → 0.67 | 0.71 |
| meta ordinary | compatible | 0.32 | 0.72 | 0.28 → 0.72 | 0.78 |
| meta ordinary + dev | compatible | 0.21 | 0.62 | 0.18 → 0.63 | 0.69 |
| meta meta | compatible | 0.24 | 0.63 | 0.22 → 0.64 | 0.73 |
| v1 I14 | private | 0.14 | 0.20 | 0.14 → 0.21 | 0.25 |
| meta meta-I | private | 0.17 | 0.16 | 0.16 → 0.16 | 0.17 |

**Perturbed successor** (a random rotation of A and t, perception unchanged): always
revising *lowers* accuracy to 0.13–0.18 in the compatible checkpoints.

**Same-representation forms as context** (neighbouring forms, not collections): no gain
in any checkpoint (always-revise within ±0.02 of immediate).

So, with compatible order, the learned successor applied to grounded neighbours carries
correct relational information: correct, because a perturbed successor destroys the gain.
That information is absent with private orders, and absent when the neighbours are
ungrounded forms.

## The swapped-item control: the item contributes nothing

The item is replaced by images of a different number from the same representation, with
the same context. Main interior, half B.

| Checkpoint | μ | Revised accuracy, true item | Swapped item: answer follows context | Swapped item: answer follows item |
| --- | --- | --- | --- | --- |
| v1 C14 | 0.1 | 0.46 | 0.26 | 0.12 |
| | 1 | 0.65 | 0.62 | 0.06 |
| | 30 (selected) | 0.66 | 0.66 | 0.05 |
| meta ordinary | 0.1 | 0.55 | 0.27 | 0.12 |
| | 1 | 0.71 | 0.67 | 0.05 |
| | 30 (selected) | 0.72 | 0.72 | 0.04 |

- At the selected μ, revised answers follow the context exactly as often whether the
  item is the right number or a wrong one. The answer is read off the neighbours through
  the successor, and the item is ignored.
- At small μ the item does contribute. Revised accuracy with the true item exceeds the
  context-following share with a swapped item by 0.16–0.28 at μ = 0.1–0.3. But accuracy
  there (0.46–0.65) is *below* the context-only level (0.66–0.72).
- **No μ combines the item with the context to beat the context alone.**

So B1 shows relational *propagation* (grounded neighbour → successor → answer), not
*integration* of an uncertain recognition with relational evidence. The withheld forms do
not acquire or reveal a meaning. Their position does all the work, which is exactly the
"position becomes meaning" risk the withholding policy guards against in Part A.

## Confidence gating had nothing to select

| Checkpoint | Always | Gated (rate) | Random at gated rate | Oracle selector |
| --- | --- | --- | --- | --- |
| v1 C14 | 0.66 | 0.66 (0.99) | 0.65 | 0.73 |
| meta ordinary | 0.72 | 0.72 (1.00) | 0.72 | 0.79 |
| meta meta | 0.63 | 0.63 (0.99) | 0.63 | 0.71 |
| v1 I14 | 0.20 | 0.21 (0.99) | 0.20 | 0.30 |

- Where revision helps, it helps almost uniformly, so the controller revises everything.
- Where it hurts (perturbed successor), the controller learns to rarely revise (rate
  0.00–0.51) and recovers the immediate accuracy.
- The controller learns whether revision helps *in general*, not *for which item*. The
  oracle selector's margin (0.07–0.10 over always revising) shows some room for selection
  that the pre-revision features did not capture.
- Calibration of the controller's help probability is good (ECE 0.00–0.05 in the main
  group), but uninformative here.

## Confidence features

AUROC for predicting immediate errors, representations other than dots (0.5 = no
information):

| Feature | Image, range over checkpoints | Sequence (Morse) |
| --- | --- | --- |
| SRI equilibrium energy / ‖f‖² | 0.50–0.63 | 0.12–0.56 |
| ‖f‖ alone | 0.34–0.43 | 0.22–0.57 |
| negative matching margin | 0.53–0.58 | 0.34–0.74 |

- All are weak predictors of error.
- The energy is no better than the margin for images, and inconsistent, sometimes
  strongly inverted, for Morse.
- ‖f‖ alone is below 0.5: larger features go with fewer errors.
- The SRI energy is not a useful confidence feature in these checkpoints, as the
  external review cautioned.

## Other observations

- **The old label readout barely moves under revision** (withheld 0.05–0.12 →
  0.03–0.16). Revision acts on the core's inner layer, and the path to the label output
  dilutes it.
- **Boundary counts** (1 and 9, one-sided context) show similar gains in the compatible
  checkpoints (for example meta ordinary 0.37 → 0.84).
- **Morse** gains most in several checkpoints (meta: 0.09 → 0.64). Its immediate
  matching against dots is near chance, and the context supplies the answer.

## Reading

1. **Positive.**
   - With compatible order, the learned successor is accurate enough to carry grounded
     meaning one step.
   - Private orders and a perturbed successor remove the gain.
   - This is the first direct evidence in this line that the relation the core learned
     can drive correct answers.
2. **Negative.**
   - The revision ignores the item. It reads the answer from the neighbours.
   - With n − 1 and n + 1 both grounded, the context alone determines the answer, so this
     task cannot test whether relations help *interpret* an uncertain form.
3. **Gating** adds nothing when revision helps uniformly, and the confidence features
   are weak.

Implications for the next design:

- **Tasks must make context ambiguous and the item informative**, so that only
  integration succeeds. For example:
  - the item is next to a grounded count, but whether it comes before or after is
    unknown;
  - context two steps away;
  - noisy or partly wrong context.

  The reviewer's case, "recognition suggests two meanings and position favours one",
  needs a candidate set that recognition narrows and context resolves.
- **Report context-only accuracy as the baseline revision must beat,** not immediate
  accuracy.
- **Part A must make the withheld forms carry their meaning in isolation.** B1 shows
  that position alone can supply the answer, so the isolated condition is the acquisition
  test.

## Cost

- Wall 44 s:
  - data 28 s;
  - rendering 4 s (candidates and context, about 60,000 dots images);
  - B1 12 s, 6 checkpoints in parallel.
- Worker CPU 61 s; largest worker peak RSS 1.1 GB.
- Per checkpoint (meta arms, 3,840 items):
  - 3,840 item solves;
  - 34,560 candidate solves;
  - 7,680 context solves;
  - 3,840 revision solves per configuration (42 configurations: 3 contexts × 2
    successors × 7 μ), plus the swapped-item revisions;
  - 43 factorizations per medium.
- Selection cost: 3 smoke runs (about 10 s each; one failed on negative stream keys).
  μ and the threshold were selected on half A.
