# Conflict loop v1: results

Contract: [conflict loop v1](conflict-loop-v1.md), frozen at commit 049efc9. Run 2026-11-06, seeds 450–453, all 12 runs.

- **Command:** `EV_WORKERS=6 .venv/bin/python -m experiments.conflict_loop main` (53 min wall).
- **Artifacts:** `reports/conflict-loop-v1{,-summary,-runtime}.json`; checkpoints in `runs/conflict-loop-v1/` (not
  tracked).

## Summary

1. **The learned arbiter gets both strengths, even without a loop.** The `arbiter` arm (logic one-pass):
   - keeps intuition's chain accuracy: size 0.96, grandparent 0.93 (intuition 0.97), named 0.86 (0.88);
   - beats both routes on ancestor (0.85, against intuition 0.72 and logic 0.60), order (0.83 against 0.79), symmetry
     and inverse (+0.04 to +0.05), and category from meaning (0.77 against 0.66);
   - says unknown on 0.63 of held-out unknown colours and 0.59 of colours nothing tells;
   - says unknown on only 0.013 of inferable facts.

   No fixed rule in dual-route v1 managed both.
2. **The conflict-triggered loop runs, and helps only on facts like the withheld training ones:**
   - **Withheld training facts** (never told to logic): logic's step 0 gets 0.81, and after the loop 0.89. The steps
     derive some of them.
   - **Held-out tests:** logic after the loop is within +0.04 of step 0 (size 0.65 → 0.69, ancestor 0.54 → 0.56).
   - **The produced middle thing** is rarely the true one: the named animal 0.26, the shared child for grandparent 0.04,
     the class for an anonymous individual 0.00.
   - **Expected steps** are about 1 for every chain length.
3. **Withholding half the chain facts weakens logic's one pass** (size 0.78 → 0.65, grandparent 0.71 → 0.59), and the
   loop does not make up for it on held-out tests. The arbiter recovers most of it from intuition (size 0.94,
   grandparent 0.91).
4. **The all-told loop triggers on about 3% of training rows** (about 520) and gains nothing. After the loop it is
   slightly worse, and it says unknown less on held-out unknown colours (0.36 → 0.15). That confirms the "no pressure"
   reading of multi-step reader v1.
5. **"Sometimes → unknown" for new individuals** stays low everywhere: logic 0.07–0.17, arbiter 0.13–0.21.
6. **Intuition is identical across arms** (separate codes): inheritance 0.92, grandparent 0.97, size 0.96.

## Results

Balanced accuracy (unknown is wrong on T/F tests). Mean of 4 seeds. Columns: intuition, logic at step 0, logic after
the loop, arbiter.

| Test | `arbiter`: I / L0 / L / A | `conflict-loop`: I / L0 / L / A | `conflict-loop-told`: I / L0 / L / A |
|---|---|---|---|
| P1 inheritance | 0.92 / 0.83 / 0.83 / 0.89 | 0.92 / 0.83 / 0.85 / 0.90 | 0.92 / 0.85 / 0.84 / 0.89 |
| P2 symmetry | 0.70 / 0.67 / 0.67 / **0.75** | 0.70 / 0.71 / 0.69 / **0.75** | 0.70 / 0.71 / 0.66 / 0.74 |
| P3 inverse | 0.74 / 0.68 / 0.68 / **0.76** | 0.74 / 0.70 / 0.71 / **0.77** | 0.74 / 0.70 / 0.69 / 0.75 |
| P4 order | 0.79 / 0.68 / 0.68 / **0.83** | 0.79 / 0.61 / 0.62 / 0.80 | 0.79 / 0.61 / 0.59 / 0.80 |
| P6 grandparent | 0.97 / 0.71 / 0.71 / 0.93 | 0.97 / 0.59 / 0.59 / 0.91 | 0.97 / 0.55 / 0.56 / 0.94 |
| P6 same colour | 0.57 / 0.59 / 0.59 / 0.62 | 0.57 / 0.54 / 0.54 / 0.59 | 0.57 / 0.57 / 0.51 / 0.60 |
| G2 category from meaning | 0.66 / 0.70 / 0.70 / **0.77** | 0.66 / 0.74 / 0.70 / **0.79** | 0.66 / 0.70 / 0.61 / 0.77 |
| G4 silly vs sensible | 0.54 / 0.56 / 0.56 / 0.54 | 0.54 / 0.55 / 0.57 / 0.58 | 0.54 / 0.56 / 0.56 / 0.59 |
| L1 kind-of | 0.66 / 0.64 / 0.64 / 0.59 | 0.66 / 0.49 / 0.52 / 0.57 | 0.66 / 0.61 / 0.60 / 0.59 |
| L1 ancestor | 0.72 / 0.60 / 0.60 / **0.85** | 0.72 / 0.54 / 0.56 / **0.81** | 0.72 / 0.62 / 0.54 / **0.83** |
| L1 size | 0.96 / 0.78 / 0.78 / 0.96 | 0.96 / 0.65 / 0.69 / 0.94 | 0.96 / 0.82 / 0.81 / 0.97 |
| L2 named | 0.88 / 0.79 / 0.79 / 0.86 | 0.88 / 0.80 / 0.81 / 0.87 | 0.88 / 0.82 / 0.81 / 0.85 |

**Ancestor by depth (arbiter):**

| Arm | Depth 3 | Depth 4 | Depth 5 |
|---|---|---|---|
| `arbiter` | 0.97 | 0.81 | 0.54 |
| `conflict-loop` | 0.97 | 0.71 | 0.50 |

**Size by distance (arbiter):** 0.88–1.00 in every arm.

**Unknown** (share answered U):

| Read | `arbiter`: L / A | `conflict-loop`: L0 / L / A | `conflict-loop-told`: L0 / L / A |
|---|---|---|---|
| Held-out unknown colours (S4a) | 0.55 / **0.63** | 0.51 / 0.52 / 0.53 | 0.36 / 0.15 / 0.42 |
| Colours nothing tells (P7) | 0.49 / **0.59** | 0.46 / 0.51 / 0.43 | 0.52 / 0.17 / 0.58 |
| Inferable facts (control) | 0.006 / **0.013** | 0.010 / 0.023 / 0.006 | 0.010 / 0.013 / 0.019 |
| New individuals, "sometimes → U" | 0.07 / 0.13 | 0.14 / 0.17 / 0.13 | 0.15 / 0.11 / 0.21 |

**The loop:**

| Read | `conflict-loop` | `conflict-loop-told` |
|---|---|---|
| Rows triggered in the last training step | 570 (98 of 1,558 withheld) | 520 |
| Trigger rate on held-out tests | 0.11–0.60 | 0.09–0.52 |
| Expected steps when triggered | 0.87–1.15, flat over chain length | 0.76–0.82 |
| Withheld facts: logic step 0 → after the loop | 0.81 → **0.89** (trigger rate on true ones 0.38) | — |
| Told chain facts | 0.99 → 0.99 | 1.00 → 1.00 |
| First produced = true middle (named / grandparent / S3b class) | 0.26 / 0.04 / 0.00 | 0.05 / 0.00 / 0.00 |

**Conflict flags intuition's errors:** graded AUROC 0.74–0.76 (as dual-route v1).

**The dax** (arbiter): always → T 1.00, never → F 0.83–0.86, usually → T 0.63–0.75, "sometimes → U" 0.06–0.17.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | The told loop triggers on < 5% of rows and gains nothing; `conflict-loop` triggers on most withheld T facts and fits them after the loop (≥ 0.8; step 0 ≤ 0.6) | Told: **holds** (about 3%; no gain). Withheld: fitted 0.89 after the loop, but step 0 already reaches 0.81 (logic generalizes from the told half), and only 38% of the true ones trigger. **Partly** |
| 2 | Held-out chains: logic after the loop ≥ step 0 + 0.10; first produced = middle ≥ 0.5 | **Fails:** +0.00 to +0.04; middle 0.00–0.26 |
| 3 | The arbiter keeps intuition's chains within 0.05; unknown ≥ 0.5 on S4a; ≤ 0.05 unknown on inferable facts | **Holds**, in all three arms (best in `arbiter`: S4a 0.63, control 0.013, and above intuition on ancestor and order) |
| 4 | "Sometimes → unknown" improves only through the arbiter; logic ≤ 0.3 | Logic ≤ 0.3 **holds** (0.07–0.17). The arbiter helps a little (0.13–0.21) |
| 5 | Risk: logic's inheritance drops ≥ 0.05 | **Not realized** (0.83–0.85, as dual-route v1). Intuition is unaffected |

## Interpretation

- **[read: results] Arbitration, not looping, delivers the dual-route promise.** A 7 → 16 → 3 network trained on 5%
  held-out statements combines intuition's chain accuracy with logic's unknown. On ancestor and order it beats both
  routes. The routes carry complementary information: logic's ranking (AUROC 0.95 on ancestor) plus intuition's verdict
  is more than either.
- **[read: results] The loop learns to fit what it is pressured on, not to reason.** Withheld training facts are partly
  derived by stepping (0.81 → 0.89). But the produced middle things are not the true middle entities, and the gain does
  not reach held-out facts. The loop found a way to fit its pressured set through extra computation, without
  discovering "go to the middle thing and read again". With one-hot entities and 3,000 steps, the shortcut through
  extra capacity is easier than the structured route.
- **[derived] Withholding hurts more than the loop helps.** Logic's one-pass training on half the chain facts generalizes
  worse (size 0.78 → 0.65). The arbiter makes up for it from intuition.
- **[read: results] "Sometimes → unknown" for new individuals remains unsolved** by every component. It needs the
  individual → kind → value chain, which neither one pass nor this loop provides.

## Computation (measured)

- 53 min wall on 6 workers; 5.2 CPU h (about 26 CPU min per run, within the declared 20–30).
- Training CPU per run: about 23–27 min. The arbiter's training (500 steps on about 900 statements) is negligible.
- Progress was logged every 250 steps.
- No tuning or selection runs.

## Next (for the user)

1. **Keep the arbiter.** It is the component that works; a next step could give it the conflict signal's history or
   context.
2. **The loop needs a reason to go through a middle thing,** not just to compute more. Options:
   - supervise nothing but make middle things the only route: logic sees base relations only, never the higher-order
     ones directly;
   - make the produced code be checked against an entity (a nearest-entity snap) before reading again;
   - have the loop read with base relations only.

   Each is a design choice about how much structure to give the step.
3. **Accept that chains are intuition's job** (one pass generalizes trained closures well), and keep logic for what only
   it can do: unknown, quantities, the algebra, and conflict.
