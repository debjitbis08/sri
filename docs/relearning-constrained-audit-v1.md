# Constrained update at relearning: audit and one continuation

Status: **diagnostics, 2026-11-04, seeds 300–303. Not a contract.** Tests the constrained update proposed in [relearning-loss-analysis-v1.md](relearning-loss-analysis-v1.md) (section 4), following its steps 1–3. Code: `experiments/relearning_constrained_audit.py` (audit; `continuation` for step 3), tables from `experiments/relearning_constrained_audit_summary.py`. Reports: `reports/relearning-constrained-audit-v1.json.gz` and `reports/relearning-constrained-continuation-v1.json.gz`, with runtime JSONs.

## The update tested

At every relearning update, on the batch's context episodes (each step pairs a collection with taps of the same count):
- **S** is the mean squared distance of the context collections' codes from their mean (collection spread).
- **E** is the mean squared distance between same-step collection and taps codes (cross-sense mismatch).
- **δ₀** is Adam's actual parameter displacement from all existing losses (clipped gradient, standard moment update), over all 263,914 trainable parameters: image projection head, taps encoder and core.
- **δ** is the smallest Euclidean correction of δ₀ with ∇log S·δ = 0 and ∇E·δ ≤ 0, computed as in the analysis.

Adam's moments are updated from the raw clipped gradient; only the displacement is corrected.

## Step 1–2: audit of the first 20 proposals

Two paths of 20 updates from the captured state, seeing identical batches and draws. The **audit** path applies δ₀, which is the original relearning, and measures δ. The **corrected** path applies δ. The captured state equals step 8's delay checkpoint exactly in every seed.

**Gate, declared in the script before the run.** Each criterion had to hold in every seed:
- **G1:** the median retained norm ‖δ‖/‖δ₀‖ is at least 0.5.
- **G2:** the summed first-order decrease of the sequence loss and of English matching keeps at least half of the proposal's.
- **G3:** on the corrected path, the realized change of log S on the batch is at most a quarter of the audit path's, and so is the rise of E. The held-out collection spread falls by at most a quarter of the audit path's fall.

**It passed in all four seeds:**

| | Seed 300 | 301 | 302 | 303 |
|---|---|---|---|---|
| Median ‖δ‖/‖δ₀‖ | 0.99 | 0.99 | 0.99 | 0.99 |
| Sequence-loss first-order decrease kept | 0.73 | 0.71 | 0.73 | 0.85 |
| English-matching decrease kept | 0.78 | 0.89 | 0.90 | 0.85 |
| Σ\|Δlog S\|, realized: audit / corrected | 2.49 / 0.008 | 2.58 / 0.019 | 2.17 / 0.016 | 1.62 / 0.008 |
| ΣΔE, realized: audit / corrected | 151 / 8.3 | 66 / 6.2 | 85 / 2.9 | 124 / 2.2 |
| Held-out log collection spread fall at 20: audit / corrected | 2.51 / 0.21 | 2.72 / 0.25 | 2.12 / 0.11 | 1.60 / 0.08 |

- **The two constraints are nearly independent** (cos(∇log S, ∇E) between −0.35 and +0.16).
- **The mismatch constraint was active in 16–20 of the corrected updates.**
- **The finite steps follow the first-order prediction closely:** the realized |Δlog S| per corrected step is 10⁻⁵ to 10⁻², against 0.01–0.23 for the proposal.

So the correction is cheap, keeps almost all of the update and most of its descent, and holds both statistics at the actual step size.

**The gate did not include the interpreter, and the corrected path's reads already fell by update 20.** Means over seeds at updates 10 and 20, against stage A's arms from the same state and batches:

| Measure (at 0) | Unchanged, 10 → 20 | Candidate stop-gradient, 10 → 20 | Constrained, 10 → 20 |
|---|---|---|---|
| Collection spacing (26.7) | 14.6 → 8.9 | 23.5 → 23.2 | 25.8 → **24.8** |
| Same-count c–t distance / spacing (0.28) | 0.86 → 1.46 | 0.35 → 0.37 | 0.30 → **0.33** |
| AUROC of −distance(c, t) (0.91) | 0.75 → 0.66 | 0.88 → 0.87 | 0.90 → **0.89** |
| Collections → taps, fresh (0.89) | 0.53 → 0.26 | 0.74 → **0.70** | 0.72 → 0.49 |
| Taps → collections, trained order (0.99) | 0.59 → 0.50 | 0.81 → **0.87** | 0.76 → 0.51 |
| Sequence, collections, 1 step up (0.92) | 0.71 → 0.52 | 0.88 → **0.79** | 0.85 → 0.48 |
| Composition, taps, 2 steps up (0.76) | 0.80 → 0.53 | 0.82 → **0.64** | 0.49 → 0.34 |

The geometry was held, and the codes still pair taps with collections by distance (AUROC 0.89). But the comparator's rankings, even the trained taps → collections, fell as far as unchanged, and sequence prediction and composition fell further. The interpreter was not protected over these 20 updates.

## Step 3: one constrained continuation, 200 updates

The gate passed, so the one continuation was run. The step policy, declared from the audit, takes the full corrected step with no backtracking and logs the realized changes. The references are stage A's `unchanged` and `seq-cand-stop` arms. Cells are the change from update 0: the worst transient / the level at 200, mean over seeds.

| Measure (at 0) | Unchanged | Candidate stop-gradient | Constrained |
|---|---|---|---|
| Collections → taps, fresh (0.89) | −0.67 / −0.21 | **−0.29 / −0.13** | −0.60 / −0.31 |
| Collections → taps, stored (0.94) | −0.71 / −0.21 | **−0.31 / −0.17** | −0.59 / −0.35 |
| Taps → collections (0.99) | −0.54 / −0.02 | **−0.23** / −0.04 | −0.53 / −0.07 |
| Memory A, symbol → taps (0.76) | −0.59 / −0.16 | **−0.25 / −0.10** | −0.49 / −0.28 |
| Memory A, taps → symbol (0.81) | −0.41 / −0.01 | **−0.16** / −0.03 | −0.34 / −0.03 |
| English matching, at 200 (0.15) | **+0.39** | +0.36 | +0.29 |
| Sequence, collections, 1 up (0.92) | −0.45 / **−0.06** | **−0.18** / −0.08 | −0.51 / −0.10 |
| Sequence, taps, 1 up (0.94) | −0.43 / **−0.02** | **−0.15** / −0.03 | −0.51 / −0.12 |
| Sequence, Devanagari forms, 1 up, at 200 (0.59) | **+0.18** | +0.09 | +0.06 |
| Composition, collections, 2 up (0.71) | −0.38 / **+0.06** | **−0.35** / −0.11 | −0.44 / −0.03 |
| Composition, taps, 2 up (0.76) | −0.35 / **+0.13** | **−0.27** / −0.05 | −0.44 / −0.04 |
| Collection spacing (26.7), at 200 | 8.2 | **24.0** | 10.4 |

**The constrained update fails every stage A criterion.** Its worst transient on the interpreter is close to unchanged (−0.60 against −0.67). At the deadline it is the worst of the three on the interpreter and on memory-present reads, and it is slower on English. Sequence prediction and composition are no better than unchanged.

**The batch statistics were held, but the held-out geometry was not.**
- Over updates 20–200, the realized |Δlog S| per step stayed at 1–3·10⁻³ and E fell.
- Yet the held-out collections' log spread fell from 6.08 to 4.17, and collection spacing fell to 10.4 by 200 (15.8 at 50).
- The mismatch constraint was active in only 27–46% of updates after 50.

## What this shows

1. **Holding aggregate geometry does not protect the interpreter.** In the first 20 updates, spacing, alignment and distance-based pairing were all kept. The comparator's readings and the predictor's sequence reads fell as much as unchanged or more.
   - Neither statistic depends on the comparator g or the predictor P, so those parameters were never constrained. The analysis anticipated this: a reader can change while geometry is held.
   - The codes may also have moved in ways S and E do not register, such as rotations and translations that keep distances but change what g reads from coordinates. That was not measured.
2. **Over longer training, two batch statistics do not hold the code space.**
   - S mixes within-count and between-count variance: total batch spread can be kept while the counts move closer together. The split was not measured, so this explanation is unconfirmed.
   - Whatever the route, constraining the batch's spread and mismatch did not keep the held-out collections' spacing past about 20 updates.
3. **The candidate stop-gradient still protects the interpreter most.** It changes which gradient reaches the codes rather than which displacement is accepted, and so also leaves g and P with a different training signal. The constrained update removes about 1% of each step's norm, and the step it keeps is nearly the whole of the damaging update.
4. **The declared gate measured feasibility, not protection.** It checked the constraint's own statistics and first-order descent. The audit's own reads at update 20 already showed the failure; a gate built for interpretation would have included them.

## What this does not establish

- **Other constraint sets.** For example: per-count class means, the comparator's own outputs on context pairs, or a larger set of output constraints as in orthogonal gradient descent. Each widens the method toward a retained-output constraint, and none is tested.
- **The within/between split of S, and the rotation hypothesis.** Neither was measured.
- **Whether a backtracking policy or a nonzero tolerance would change anything.** The realized deviations were already small.

## Recommendation

The constrained update in this form is closed. Its two statistics are satisfiable while the representation and its readers change. Stage A's conclusion stands:
- the candidate-side pull is the dominant identified route of the early collapse, and stopping it protects the interpreter;
- what remains open is keeping the candidates' contribution to multi-step composition.

The evidence here adds one requirement for any further remedy: it must be judged on the comparator's and predictor's reads, not on code geometry alone. No further run is proposed here; the next choice is for review.

## Computation

- **GPU** (GTX 1650), three workers, float32, deterministic algorithms.
- **Capture:** the captured state is rerun per seed, about 170–230 s.
- **Audit:** two 20-update paths per seed, with reads at 0, 10 and 20 on the corrected path. 7 min 9 s wall, 974 s aggregate worker CPU, peak RSS 3.7 GB.
- **Continuation:** one 200-update path per seed with 21 read points. 7 min 48 s wall, 1,060 s aggregate worker CPU, peak RSS 3.7 GB.
- **Per update, mean over the audit:**
  - 0.030 s for the four extra gradients (sequence, English matching, log S, E) and the whole-loss gradient;
  - 0.0018 s for the correction itself;
  - 0.24 s for the finite-step checks, which run in the audit only.
- **Complexity:** for P parameters and two constraints, the correction is O(P) time and O(P) storage beyond Adam. It needs two extra scalar backward passes per update.
- **Selection and tuning:** none. The step policy took one declared value, with no backtracking.
