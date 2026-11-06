# Dual-route truth v1: intuitive verdicts and taught logic in one currency

Status: **draft contract, 2026-11-06. Not frozen; awaiting user review.**

## User direction (2026-11-06)

After the [taught-truth cost diagnosis](taught-truth-cost-diagnosis.md):

> Logic dictates that I answer sometimes or unknown, but intuitively I have made up my mind. True and False are always
> there as pure outputs (no codes) but we add logic on top of it. So there are two ways T and F can be produced and the
> one mostly tracks the other, but sometimes there can be a conflict.

Agreed in discussion: both routes produce verdicts, in **one shared currency**. The logic layer gets no T/F symbols of
its own. What T/F cannot express (unknown, later disputed) is a pattern of support, not a new code.

## Why

- **The diagnosis:** truth produced as a learned code, compared by cosine, generalizes worse to unseen facts than a
  plain truth unit, even on the same training (grandparent 0.66 against 0.97). Training many statement kinds in one
  reader also pushes unseen facts toward F.
- **What only the taught layer gives:**
  - unknown where nothing is known (0.57–0.64);
  - kind quantities;
  - the K3 algebra.
- **The loop never stepped,** because nothing required it ([multi-step reader v1](multistep-reader-v1-results.md)).
  Conflict between the two routes is a candidate reason for later steps; that is not in v1.

## Design

Shared object codes E and relation codes R (one representation of the world), and two routes:

| Route | Output | Trained on |
|---|---|---|
| **Intuition** | one scalar logit u per statement: the plain verdict (`tworel` form, as the built-in unit) | individual T/F statements (closed world), BCE |
| **Logic** | **two support scalars** (s⁺, s⁻) ∈ [0, 1]², read from its own pair space: `σ(S_L([G_L([c_a, c_b, r]), r]))` | individual statements as support targets: T = (1, 0), F = (0, 1), U = (0, 0) |

The logic route also holds the taught parts that are not verdicts:
- **Kind quantities** (always / usually / sometimes / never) stay taught codes, read by a code head as in taught truth
  v1. They are statements about kinds, not verdicts.
- **Anonymous-individual defaults** are given as support targets (sometimes → (0, 0)).
- **Fill-the-tail** (the predict half, needed later by the loop): the code head over entity codes, as taught truth v1.
- **Connectives** become a learned operator on support pairs: (pair₁, pair₂, operator code) → pair. It is taught on the
  K3 table at the corners (not T = F, T and U = U, …). Fit only in v1.

### Reading a route's state

| State | Intuition | Logic |
|---|---|---|
| T | u > 0 | s⁺ > ½, s⁻ ≤ ½ |
| F | u ≤ 0 | s⁻ > ½, s⁺ ≤ ½ |
| unknown | — | both ≤ ½ |
| disputed | — | both > ½ (not taught in v1; reported if it occurs) |

### Agreement and conflict

**Conflict:** intuition says T and logic says F or unknown, or the reverse. Combined answers are fixed rules compared
side by side; there is no learned arbiter in v1:

| Rule | Answer |
|---|---|
| R1 intuition | intuition's verdict |
| R2 logic | logic's state |
| R3 logic overrides when decisive | logic's T or F if it is decisive; otherwise intuition's verdict |
| R4 conflict means unknown | the agreed verdict; unknown when the routes conflict |

## Arms

| Arm | Codes |
|---|---|
| `dual-shared` | both routes read the same E and R |
| `dual-separate` | each route has its own E and R (no shared representation) |

**References:** `unit-one` and `pred-one` from multi-step reader v1, same statements, splits and seeds, read from their
checkpoints (not retrained).

Seeds 450–453 (the multi-step splits), 8 runs. Statements as in multi-step reader v1 (`src/everyday_truth_v2.py`).
Training:
- every statement group weighted equally, as before;
- the intuition loss on individual T/F only;
- full batch, Adam lr 3e-3, weight decay 1e-4, 3,000 steps.

## Reads

- **Each route alone** on the chain and inference tests: L1, L2, P6, P4, P3, P1.
- **S3b and the dax ("sometimes → unknown")** and **S4 (unknown)** for logic and for the combined rules.
- **Agreement rate** per test.
- **Where conflicts occur:** their share on "sometimes" items, unknown items and chains.
- **Conflict as an error signal:** AUROC of "the routes conflict" for intuition's held-out errors, and the same with a
  graded disagreement score.
- **Combined rules R1–R4** on every test.
- **Code arrangement:** inheritance, and the dax's other-classes rejection.

## Predictions

1. **[diagnosis]** The intuition route matches `unit-one` on chains within 0.05. Sharing codes with logic does not hurt
   it: `dual-shared` against `dual-separate` within 0.05.
2. **[diagnosis: the readout]** The two-scalar logic route beats `pred-one`'s code readout on chains by ≥ 0.10. That
   is the readout half of the cost, removed. It stays below intuition, because of the statement-mix half.
3. **[taught truth v1]** "Sometimes → unknown" on new individuals stays low in logic (≤ 0.3). It is still the two-step
   chain. Unknown on S4 is kept (≥ 0.5).
4. **[De Neys; derived]** Conflicts concentrate on "sometimes" and unknown items and on chains. Conflict flags
   intuition's held-out errors with AUROC ≥ 0.65.
5. **[derived]** R3 keeps intuition's chain accuracy within 0.03 and logic's unknown on S4. It is the best single rule.

## Prior work

No new search; sources **[familiar]** unless noted, plus the [shared-output brief](shared-output-brief.md) and taught
truth v1.

- **Dual-process accounts of reasoning:**
  - associative against rule-based systems (Sloman 1996, "The empirical case for two systems of reasoning");
  - System 1 / System 2 (Kahneman 2011; Evans and Stanovich 2013);
  - belief bias in syllogisms (Evans, Barston and Pollard 1983): intuition often wins;
  - conflict detection (De Neys 2012, "Bias and conflict"): people register intuition–logic conflict even when they give
    the intuitive answer.
- **Mīmāṃsā:** intrinsic validity (*svataḥ prāmāṇya*): a cognition arrives already accepted and is overturned only by
  defeat; Nyāya's extrinsic validity (*parataḥ*). In the brief, these motivate R1 and R3 as candidate stances. They are
  an analogy, not an implementation.
- **Two support scalars:**
  - Belnap's four values as the bilattice FOUR, pairs of evidence for and against (Ginsberg; Fitting);
  - subjective logic's belief, disbelief and uncertainty (Jøsang);
  - in deep learning, evidential classifiers that output evidence per class (Sensoy et al. 2018).

  The logic route's (s⁺, s⁻) is the bilattice reading with learned support.
- **Fast and slow in AI:** for example Booch et al. 2021, "Thinking fast and slow in AI". Their designs pair a learned
  fast system with a symbolic slow one. Here both routes are neural (the program's purely-neural constraint).
- **What this adds:** a test of whether a plain innate verdict plus a taught logic layer in the same currency keeps the
  intuitive route's generalization, recovers the taught layer's unknown, and produces a conflict signal that marks
  errors.

## Scale

- Intuition costs one pass per question.
- Logic costs one pass for two scalars; kind and fill questions read codes.
- Conflict is a comparison of two outputs, with no search.
- A new value pattern costs nothing; a new kind quantity costs one code.

## Computation (declared)

- About 40 CPU min per run (`unit-one` 13 + `pred-one` 24 in multi-step reader v1).
- 8 runs ≈ 5–6 CPU h. **6 workers** (the multi-step run showed 8 workers on 6 cores running about 3.5× slower), about
  1–1.5 h wall.
- The runner logs progress every 250 steps.
- Stored: reports; checkpoints in `runs/` (not tracked). No tuning.

## Open decisions for the user

1. **Arbitration.** Fixed rules R1–R4 compared in v1, as drafted? Or should a learned arbiter be included already?
   It would learn when to trust which route; a fixed rule is easier to interpret first.
2. **Shared codes.** Should both routes see one representation of the world, as drafted? Or should intuition and logic
   each keep their own (`dual-separate` as the main design, not only a control)?
