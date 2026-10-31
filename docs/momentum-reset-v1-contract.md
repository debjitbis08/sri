# Momentum reset v1: contract (DRAFT, not frozen, not run)

Status: **draft 2026-10-31, revised after an external review the same day** (single primary time point, narrower equivalence reading, a held-out context-performance gate, precise reset checks, two unsupported
expectations removed). Follows [forgetting mechanism v1](forgetting-mechanism-v1-results.md), where inherited Adam state with zero new gradients lost 0.09–0.22
recall within about 25 updates and then stopped. That shows a vulnerability, not how much of the early drop in real context training it explains.

## Why this study (the larger question)

The question that matters for the architecture is: **can the dense component's arbitrary facts be preserved while the sparse component's transfer advantage is retained, at an acceptable cost?**
Evidence so far (arbitrary pairings v1, Option A; only `mixed` against `sparse48` was a prespecified contrast, the rest is descriptive): pair recall at the end of block 4 and absolute transfer (a) at block 5 —
`sparse48` 0.34 and 0.695; `dense` 0.53 and 0.613; `mixed` 0.48 and 0.625 (transfer worse than `sparse48` by 0.070); `mixed-h500` 0.47 and 0.665; `mixed-reset` 0.52 and 0.650. No arm kept the pairs through 2,000
updates of other material (0.12–0.15). Keeping the facts and keeping transfer are separate requirements, both unmet together. Remedies are compared on retention, transfer, new-learning cost, compute and storage
together ([external remedies note](forgetting-remedies-prior-work.md)). This study is the cheapest first remedy test and the mechanistic check the previous results call for; it does not answer the larger question.

## Question

During real context training from the same checkpoint, does zeroing the inherited Adam first moments reduce the early recall drop, and what does it cost in continued context performance, transfer and continued pair learning?

## Design

Common start: the end of block 4, Option A, arms `sparse48`, `dense`, `mixed`, `mixed-h125`, **fresh seeds 340–349**, N = 45, L = 2,000, 64 training renderings, GPU; code and streams as in
`experiments/forgetting_mechanism.py` (branch runner extended additively: an `adam` option, per-update loss records and reset checks; old branches unchanged).

| Branch | What changes |
|---|---|
| `ctx` | context episodes, inherited Adam state intact (reference) |
| `ctx-m0` | as `ctx`, but `exp_avg` is zeroed for **every initialized parameter state** at update 0; `exp_avg_sq` and step counters kept |
| `ctx-fresh` | as `ctx`, with the entire Adam state cleared (descriptive: a fresh state changes the effective step sizes) |
| `hold` | context + A pair items every update, inherited state intact |
| `hold-m0` | as `hold`, `exp_avg` zeroed (cost to continued pair learning) |
| `zero-grad` | replication: explicit zero gradients, **explicit decay disabled** (also in `mixed-h125`) |
| `zero-grad-m0` | zero gradients, `exp_avg` zeroed, decay disabled: must not move any weight (check) |

Context episodes and pair draws use the same streams in `ctx`, `ctx-m0` and `ctx-fresh` (and in `hold` and `hold-m0`), so batches are identical across the branches compared.
In training branches decay in `mixed-h125` is active as the arm's recipe. Evaluation points: 0, 10, 25, 50, 100, 150, 200, 250, then every 250 to 2,000.

**Reset checks (recorded as booleans).** Immediately after zeroing, every parameter, every `exp_avg_sq` and every `step` is bitwise equal to its value before zeroing, and every `exp_avg` is exactly zero. A
non-initialized state (a parameter with no Adam state) is counted and reported.

## Measures

- Accuracy change D(t) = recall(t) − recall(0), independent renderings (training renderings beside), paired over seeds, at t = 50, 250 and 2,000.
- **Continued context performance** (a fixed, held-out evaluation, measured at the end of each training branch and at update 0): the mean of three held-out accuracies from the existing evaluation — taps-to-collection matching
  (`senses/taps_to_collection`), collection comparison (`senses/comparison/collections`) and collection successor ordering (`succession_pairwise/collections`). The first two are near ceiling already (0.99, 0.98 at the end of the
  previous study), so this measures continued context performance, **not** the ability to acquire new tasks. Components and the coherence AUROC are reported beside it.
- **Training losses:** all four per-update losses (match, next, compare, transition) recorded in every training branch; mean over updates 1–50, 51–250 and 251–2,000, paired by seed because the batches are identical. Descriptive intervals;
  matching loss alone is not read as the learning cost.
- Transfer (a) (structured, absolute) at the end of each training branch, with the eight transfer measures descriptive.
- Decoders (rebuilt and frozen bank, probe) and weight movement per group, descriptive.

## Readings (prespecified; margin 0.05; the inherited rule: better/worse if |mean| > margin and the paired 95% t-interval excludes 0; equivalent if within ±margin; noninferior if the lower bound exceeds −0.05; positive difference means less loss)

1. **Replication.** `zero-grad` D(250) against 0: worse in an arm replicates the previous result in that arm.
2. **Checks.** The reset checks above; `zero-grad-m0` leaves every weight bitwise unchanged and recall constant. A failure is a runner fault.
3. **Primary: `ctx-m0` − `ctx` on D(50).** Better → zeroing the inherited first moments reduces the early drop in real training. **Persistence:** the same difference on D(250). **Longer term:** the same on D(2,000), with no prediction.
   - **Equivalent** at D(50) → resetting the inherited first moments gives no practically meaningful benefit at this checkpoint. This does not show that momentum plays no role in the dynamics.
   - A beneficial reset establishes the causal effect of that intervention; it is not an additive share of the forgetting attributable to momentum.
   - Worse → the inherited first moments were protective in these runs. Otherwise inconclusive.
4. **Cost gates** (for `ctx-m0` against `ctx` unless stated): (i) continued context performance at the end, noninferior (lower bound above −0.05); (ii) transfer (a) at the end, noninferior; (iii) `hold-m0` − `hold` on D(250) and on recall at the end of the branch, noninferior. The four training losses stay descriptive.
5. **Remedy candidate:** declared only if reading 3's D(50) is better and gates (i)–(iii) are all noninferior in the arm. It addresses the early drop only; the slow tail is not asserted to change.
6. `ctx-fresh` and everything else is descriptive.

One fixed cohort; no stopping rule; inconclusive stays inconclusive; additions are post hoc.

## Prior work (written 2026-10-31; shared survey in [prior-work.md](prior-work.md))

Adam (Kingma and Ba) is **[familiar]**. Resetting optimizer state is an established intervention: [Asadi, Fakoor and Sabach 2023](https://arxiv.org/abs/2306.17833) **[abstract]** reset the optimizer's internal state at the start of each iteration in deep RL (Rainbow on Atari) and report
improved performance, arguing carried-over moment estimates become stale when the loss changes; that setting does not establish whether a first-moment-only reset helps here. [Rodriguez-Garcia et al. 2025](https://arxiv.org/abs/2507.14056) **[abstract]**
report that momentum-SGD and Adam show stability gaps (temporary drops on earlier tasks when new ones begin); the abstract does not address resets. [Frati et al. 2025](https://arxiv.org/abs/2507.01559) **[abstract]** report that optimizer choice affects the dynamics of learning and forgetting.
A web-search summary also attributed to these and related papers statements about momentum and resets (that resets sometimes help and sometimes hurt stability gaps, that any momentum harms retention, that Adam keeps improving a finished task's loss); those are
**[search summary, not confirmed from the abstracts]** and no expectation here rests on them. Expectation (hypothesis): a first-moment reset may help, hurt or do nothing; there is no reliable prediction.
Established alternatives with a storage or compute cost (rehearsal, Dark Experience Replay, Learning without Forgetting, REMIND) are for the follow-up remedy comparison. **Prerequisite status:** the four sources above are read at abstract level; full texts are not read.

## Computation

Per arm-seed: blocks 1–4 (8,200 updates), then five training branches of 2,000 updates (`ctx`, `ctx-m0`, `ctx-fresh`, `hold`, `hold-m0`) and two zero-gradient branches (optimizer steps only): about 18,200 updates,
close to the previous study (4,174 s for 40 arm-seeds with six GPU processes), so about 1.1 h for 40 arm-seeds before the added loss records. Storage: one stored end-of-block-4 state per arm-seed (about 3P floats); loss records are 4 floats × 2,000
per training branch. Training time is O(updates · batch · P) with P the dense parameter count; a reset is O(P). To be replaced by measured costs.

## Risks

- A first-moment reset changes the optimization regime for the first dozens of updates; a result is a statement about this recipe, in Adam only.
- Identical batches make losses paired, but trajectories diverge afterwards; early trajectory changes can persist, which is why D(2,000) is measured without a prediction.
- `ctx-fresh` changes effective step sizes and is not a remedy test.
- Two of the three context accuracies are near ceiling, so the context-performance gate can miss small damage; the losses and the third component are reported to help.

## Decisions

Decided by the researchers (2026-10-31): **fresh seeds 340–349; all four arms.**

## Acceptable cost for the remedy comparison that follows (recorded here, not read in this study)

Proposed by an external review and adopted as the starting point; thresholds marked * are for the researchers to confirm before that contract is frozen.

| Criterion | Rule |
|---|---|
| Retention benefit | Same-arm remedy minus no-remedy recall at 2,000 updates of delay: meaningful improvement.* The stronger rule (interval lower bound above 0.05) is recommended for a claim of established meaningful rescue; the inherited rule (mean above 0.05 with an interval excluding zero) is the weaker alternative. |
| Preservation of acquired facts | Also compare retained recall with that arm's own pre-delay recall; distinguish partial rescue from preservation. |
| Architectural objective | Final transfer (a) noninferior to a concurrently run `sparse48` baseline, margin 0.05. |
| Learning cost | Held-out new-task performance noninferior to the same arm without the remedy, margin 0.05. |
| Resources | Prespecified budget* — provisionally at most 25% additional training time and at most one model's weight bytes of additional persistent storage; inference overhead reported separately. Replay contents, retained teachers and protection statistics count. Methods over budget are still reported on a quality–cost frontier. Declared overhead is disclosure, not an acceptance threshold. |

A **useful remedy** (substantial retention gain) and a **successful combined architecture** (also passing the `sparse48` transfer target and the learning-cost gate) are kept separate: a remedy may fail the architectural objective and still be useful progress.
