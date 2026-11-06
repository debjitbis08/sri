# Dual-route truth v1: results

Contract: [dual-route truth v1](dual-route-truth-v1.md), frozen at commit 53da70f. Run 2026-11-06, seeds 450–453,
8 runs.

- **Command:** `EV_WORKERS=6 .venv/bin/python -m experiments.dual_route_truth main` (28 min wall).
- **Artifacts:** `reports/dual-route-truth-v1{,-summary,-runtime}.json`; checkpoints in `runs/dual-route-truth-v1/`
  (not tracked).
- **References:** `unit-one` and `pred-one` from [multi-step reader v1](multistep-reader-v1-results.md), with the same
  statements, splits and seeds.

## Summary

1. **With separate codes, the intuitive route keeps the built-in unit's reasoning:** grandparent 1.00, size 0.97, named
   relations 0.89, ancestor 0.71, order 0.79. That is equal to or above `unit-one`.

   **With shared codes it does not:** grandparent falls to 0.53, order to 0.71, named relations to 0.83. Training the
   logic route on the same codes damages intuition's generalization. That is the statement-mix effect found in the
   [diagnosis](taught-truth-cost-diagnosis.md), now reaching intuition through the shared codes.
2. **The two-scalar logic route ranks unseen facts much better than the code readout,** but still rejects them:
   - AUROC on chains 0.91–0.98 (ancestor 0.93–0.98, grandparent 0.93–0.96, size 0.93–0.97), against `pred-one`'s
     0.82–0.91;
   - balanced accuracy only +0.04 to +0.11 over `pred-one`.

   Removing the code readout fixed the ranking half of the cost; the threshold half (unseen facts placed on the F side)
   remains.
3. **Conflict between the routes is a real signal:**
   - graded disagreement flags intuition's held-out errors with AUROC 0.74 (shared) and 0.77 (separate); binary conflict
     0.60–0.65;
   - conflicts concentrate where logic says "don't know": on held-out unknown colours, the routes conflict 0.50–0.53 of
     the time, against an overall conflict rate of 0.10–0.15;
   - with separate codes, conflicts are frequent on "sometimes" and "usually" individuals (0.49, 0.59), rare on
     "never" (0.09).
4. **Fixed rules cannot give both strengths:**
   - R1 (intuition) is best on chains but never says unknown.
   - R4 (unknown when they conflict) recovers unknown (held-out unknown colours 0.50–0.53; "sometimes → unknown" 0.15–0.49;
     the dax's "sometimes" 0.28–0.47), but it costs chain accuracy (size −0.12 to −0.20) and says unknown on 9–13% of
     inferable facts.
   - **R3 (logic overrides when decisive) can never answer unknown, by construction.** When logic is undecided it
     falls back to intuition. My prediction that R3 would keep logic's unknown was inconsistent with its own
     definition.
   - R3 also loses chain accuracy, because logic's confident F overrides intuition's correct T.

## Results

Balanced accuracy on held-out T/F tests (a positive is correct only if answered T, a negative only if answered F;
unknown is wrong). Mean of 4 seeds.

| Test | Shared R1 / R2 / R3 / R4 | Shared agree | Separate R1 / R2 / R3 / R4 | Separate agree | `unit-one` | `pred-one` |
|---|---|---|---|---|---|---|
| P1 inheritance | 0.86 / 0.82 / 0.83 / 0.76 | 0.90 | **0.91** / 0.83 / 0.85 / 0.77 | 0.86 | 0.88 | 0.84 |
| P2 symmetry | 0.69 / 0.69 / 0.70 / 0.63 | 0.90 | 0.70 / 0.68 / 0.68 / 0.59 | 0.85 | 0.71 | 0.68 |
| P3 inverse | 0.71 / 0.70 / 0.70 / 0.65 | 0.93 | 0.75 / 0.69 / 0.69 / 0.65 | 0.91 | 0.74 | 0.68 |
| P4 order | 0.71 / 0.70 / 0.71 / 0.64 | 0.85 | **0.79** / 0.69 / 0.69 / 0.64 | 0.80 | 0.78 | 0.61 |
| P6 grandparent | 0.53 / 0.61 / 0.61 / 0.51 | 0.93 | **1.00** / 0.56 / 0.56 / 0.56 | 0.85 | 0.97 | 0.53 |
| P6 same colour | 0.54 / 0.54 / 0.55 / 0.51 | 0.96 | 0.54 / 0.58 / 0.58 / 0.46 | 0.89 | 0.58 | 0.53 |
| G2 category from meaning | 0.74 / 0.70 / 0.71 / 0.66 | 0.94 | 0.65 / 0.67 / 0.69 / 0.52 | 0.84 | 0.62 | 0.70 |
| G4 silly vs sensible | 0.61 / 0.58 / 0.58 / 0.52 | 0.85 | 0.56 / 0.52 / 0.52 / 0.45 | 0.83 | 0.58 | 0.57 |
| L1 kind-of | 0.67 / 0.63 / 0.63 / 0.56 | 0.83 | 0.63 / 0.62 / 0.64 / 0.41 | 0.58 | 0.69 | 0.63 |
| L1 ancestor | 0.64 / 0.61 / 0.61 / 0.57 | 0.88 | **0.71** / 0.60 / 0.61 / 0.56 | 0.79 | 0.71 | 0.55 |
| L1 size | 0.94 / 0.84 / 0.84 / 0.82 | 0.87 | **0.97** / 0.77 / 0.78 / 0.77 | 0.80 | 0.96 | 0.73 |
| L2 named | 0.83 / 0.84 / 0.84 / 0.79 | 0.92 | **0.89** / 0.83 / 0.84 / 0.80 | 0.89 | 0.90 | 0.78 |

**AUROC (intuition / logic):**

| Arm | Size | Ancestor | Grandparent | Named | Order |
|---|---|---|---|---|---|
| Shared | 0.99 / 0.97 | 0.94 / 0.93 | 0.96 / 0.93 | 0.93 / 0.91 | 0.90 / 0.85 |
| Separate | 1.00 / 0.93 | 1.00 / 0.98 | 1.00 / 0.96 | 0.95 / 0.94 | 0.93 / 0.84 |

**Conflict as a signal** (pooled held-out T/F tests):

| | Shared | Separate |
|---|---|---|
| Intuition error rate | 0.16 | 0.15 |
| Conflict rate | 0.10 | 0.15 |
| Binary conflict → error AUROC | 0.60 | 0.65 |
| Graded disagreement → error AUROC | **0.74** | **0.77** |

**Unknown and "sometimes":**

| Read | Shared | Separate |
|---|---|---|
| Held-out unknown colours (S4a): U from logic / R4 | 0.44 / 0.50 | 0.44 / 0.53 |
| Colours nothing tells (P7): U from logic / R4 | 0.33 / 0.35 | 0.58 / 0.64 |
| U on inferable facts (control): logic / R4 | 0.005 / 0.09 | 0.005 / 0.13 |
| New anonymous individuals, "sometimes → U": logic / R4 | 0.06 / 0.15 | 0.08 / 0.49 |
| Conflict rate on S3b by kind value (always / usually / sometimes / never) | 0.23 / 0.22 / 0.15 / 0.06 | 0.44 / 0.59 / 0.49 / 0.09 |
| dax "sometimes → U": logic / R4 | 0.17 / 0.28 | 0.22 / 0.47 |
| dax always / never / other classes rejected (all rules) | 1.00 / 1.00 / 1.00 | 1.00 / 0.94–1.00 / 1.00 |

**Other reads:**
- kind quantities held out (S2): 0.76 (shared), 0.63 (separate);
- fits: intuition 1.00, logic 0.99–1.00, connectives 1.00, kinds 1.00.

Parameters: shared 64,966; separate 73,590.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Intuition ≈ `unit-one` (within 0.05); shared ≈ separate | **Holds for separate** (equal or above `unit-one`). **Fails for shared:** grandparent −0.44, order −0.07, named −0.06 |
| 2 | The two-scalar logic beats `pred-one` on chains by ≥ 0.10 | **Partly:** +0.04 to +0.11 in balanced accuracy (size +0.11 shared). Ranking improves clearly (AUROC 0.91–0.98 against 0.82–0.91) |
| 3 | Logic's "sometimes → U" stays ≤ 0.3; unknown on S4 ≥ 0.5 | First **holds** (0.06–0.08). Second **narrowly fails**: logic 0.44 on S4a; P7 0.33–0.58 |
| 4 | Conflicts concentrate on unknown and "sometimes" items; conflict flags intuition errors with AUROC ≥ 0.65 | **Holds:** conflict 0.50–0.53 on unknown colours against 0.10–0.15 overall. Graded disagreement 0.74–0.77 (binary 0.60–0.65) |
| 5 | R3 keeps intuition's chain accuracy within 0.03 and logic's unknown | **Fails, and was ill-posed:** R3 cannot output unknown by its definition, and it loses up to 0.44 on chains where logic's confident F overrides intuition |

## Interpretation

- **[read: results] Two routes should not share one representation, at least when trained this way.** Separate codes
  keep intuition as good as the built-in unit. Shared codes let the logic route's training (unknown targets, kinds,
  fill-the-tail) bend the codes that intuition reasons over. That fits the user's framing: intuition and logic are
  different routes, and here they need their own representations.
- **[read: results] The logic route knows more than it says.** Its ranking of unseen chain facts is almost as good as
  intuition's (AUROC 0.93–0.98 with separate codes), but its "support for" stays below ½. Logic is cautious: unseen
  means low support, which is close to "unknown", not "false". The two-scalar form expresses that; a single T/F code
  could not.
- **[read: results] Disagreement is informative, graded more than binary.** How far the two routes are apart predicts
  intuition's mistakes (0.74–0.77). That is the conflict signal the user anticipated, and the critic and
  asking-for-examples line can use it.
- **[derived] Fixed rules force a trade-off;** a combination has to know when logic's caution matters. Logic's "low
  support either way" sometimes means a genuinely open question (colours) and sometimes means "I haven't derived it,
  but it is true" (chains). No fixed rule over the two outputs separates these. That is where a learned arbiter, or the
  loop (logic working to resolve the conflict), would come in.

## Computation (measured)

- 28 min wall on 6 workers; 2.2 CPU h aggregate. Within the declared 5–6 CPU h; the smoke projection of 5 min per run
  was too low (about 16 min per run at 6 workers).
- Progress was logged every 250 steps.
- No tuning or selection runs.

## Next (for the user)

1. **Separate codes as the design,** given result 1.
2. **What to do with conflict:**
   - a learned arbiter that sees both routes' outputs and their disagreement;
   - or the loop as the logic route's way of working on a conflict: steps are spent only when the routes disagree,
     which would also give the loop the pressure it lacked.
3. **Logic's caution on derivable facts:** whether a loop in the logic route turns "low support" into supported T on
   chains, while leaving genuinely open questions unknown.
