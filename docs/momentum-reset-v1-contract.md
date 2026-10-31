# Momentum reset v1: contract (DRAFT, not frozen, not run)

Status: **draft 2026-10-31**. Follows [forgetting mechanism v1](forgetting-mechanism-v1-results.md), where inherited Adam state with zero new gradients lost 0.09–0.22
recall within about 25 updates and then stopped. That shows a vulnerability, not how much of the early drop in real context training it explains.

## Why this study (the larger question)

The question that matters for the architecture is: **can the dense component's arbitrary facts be preserved while the sparse component's transfer advantage is retained, at an acceptable cost?**
Evidence so far (arbitrary pairings v1, Option A, descriptive for arms outside the prespecified contrasts): `sparse48` recall of the pairs at the end of block 4 is 0.34 with absolute
transfer (a) 0.695 at block 5; `dense` 0.53 with 0.613; `mixed` 0.48 with 0.625 (transfer worse than `sparse48` by 0.070, prespecified); `mixed-h500` 0.47 with 0.665 and `mixed-reset` 0.52 with 0.650 sit
between. No arm kept the pairs through 2,000 updates of other material (0.12–0.15). So two requirements are separate and both unmet: keeping the facts, and keeping transfer. Remedies must be
compared on retention, transfer, new-learning cost, compute and storage together ([external remedies note](forgetting-remedies-prior-work.md)). This study is the cheapest first remedy test and the
mechanistic check the previous results call for; it is not itself an answer to the larger question.

## Question

During real context training from the same checkpoint, does removing the inherited Adam first moments reduce the early recall drop, and what does it cost in context-task learning,
transfer and continued pair learning?

## Design

Common start: the end of block 4, Option A, arms `sparse48`, `dense`, `mixed`, `mixed-h125`, **fresh seeds 340–349**, N = 45, L = 2,000, 64 training renderings, GPU; code and streams as in
`experiments/forgetting_mechanism.py` (branch runner extended additively: an `adam` option and per-update loss records; old branches unchanged).

| Branch | What changes |
|---|---|
| `ctx` | context episodes, inherited Adam state intact (reference) |
| `ctx-m0` | as `ctx`, but the **first moments** (`exp_avg`) are zeroed at update 0; second moments (`exp_avg_sq`) and step counters kept |
| `ctx-fresh` | as `ctx`, with the entire Adam state cleared (descriptive: a fresh optimizer takes large first steps because of bias correction) |
| `hold` | context + A pair items every update, inherited state intact |
| `hold-m0` | as `hold`, first moments zeroed (cost to continued pair learning) |
| `zero-grad` | replication of the previous study: explicit zero gradients, no decay |
| `zero-grad-m0` | zero gradients with zeroed first moments: must not move any weight (check) |

Context episodes and pair draws use the same streams in `ctx`, `ctx-m0` and `ctx-fresh` (and in `hold` and `hold-m0`), so the batches are identical across the branches compared.
Decay in `mixed-h125` is active as the arm's recipe in all training branches. Evaluation points: 0, 10, 25, 50, 100, 150, 200, 250, then every 250 to 2,000.

## Measures

- Accuracy change D(t) = recall(t) − recall(0), independent renderings (training renderings beside), at t = 50, 250, 2,000, paired over seeds.
- **Context-task learning:** the four per-update training losses (match, next, compare, transition) recorded in every training branch; mean of the matching loss over updates 1–50, 51–250 and 251–2,000, paired by seed
  (the batches are identical across the compared branches). Descriptive intervals; no margin.
- **Transfer (a)** (structured, absolute) at the end of each training branch, with the eight transfer measures descriptive.
- Decoders (rebuilt and frozen bank, probe) and weight movement per group as in the previous study, descriptive.

## Readings (prespecified; margin 0.05; the inherited rule: better/worse if |mean| > margin and the paired 95% t-interval excludes 0; equivalent if within ±margin; noninferior if the lower bound exceeds −0.05; positive difference means less loss)

1. **Replication.** `zero-grad` D(250) against 0: worse in an arm replicates the previous result in that arm.
2. **Check.** `zero-grad-m0` leaves every weight bitwise unchanged and recall constant: a runner check.
3. **Primary.** `ctx-m0` − `ctx` on D(50) and D(250). Better → inherited first moments contribute to the early drop in real training; equivalent → they are not what drives it there, despite the zero-gradient result; worse → they were protective; otherwise inconclusive.
4. **Secondary.** The same at D(2,000) (expected small: momentum is exhausted by then).
5. **Cost.** `ctx-m0` − `ctx`: transfer (a) at the end, noninferior or not; the matching-loss differences are reported with intervals. `hold-m0` − `hold`: D(250), D(2,000) and the recall level, noninferior or not.
   A **remedy candidate** is declared only if reading 3 is better and the cost readings are noninferior. It would address the early drop only; the slow tail is not tested to change.
6. `ctx-fresh` and everything else is descriptive.

One fixed cohort; no stopping rule; inconclusive stays inconclusive; additions are post hoc.

## Prior work (written 2026-10-31; shared survey in [prior-work.md](prior-work.md))

Adam (Kingma and Ba) is **[familiar]**. A web-search summary (**[search summary]**, not read in full) reports mixed evidence on resetting optimizer state at task changes: a stability-gap study
([arXiv 2507.14056](https://arxiv.org/pdf/2507.14056)) found Adam and momentum-SGD resets sometimes reduced and sometimes increased stability gaps; the neural tangent ensemble work reports that any momentum in SGD
harms retention and that the relation becomes non-monotonic when buffers are reset between tasks; [Frati et al.](https://arxiv.org/html/2507.01559v1) report Adam keeps improving a finished task's loss for many steps, which is not explained by
momentum alone; and a Continual-Adam line treats first-moment reset as standard while keeping second moments. Expectation (hypothesis): a first-moment reset may help or hurt; there is no reliable prediction here.
Established alternatives with a storage or compute cost (rehearsal, Dark Experience Replay, Learning without Forgetting, REMIND) are the subject of the follow-up remedy comparison, not this study.
**Prerequisite not done:** these sources must be read beyond their summaries before freezing.

## Computation

Per arm-seed: blocks 1–4 (8,200 updates), then five training branches of 2,000 updates (`ctx`, `ctx-m0`, `ctx-fresh`, `hold`, `hold-m0`) plus two zero-gradient branches (optimizer steps only):
about 18,200 updates, close to the previous study (4,174 s for 40 arm-seeds with six GPU processes), so about 1.1 h for 40 arm-seeds before the added loss records. Storage: one stored end-of-block-4 state per arm-seed
(about 3P floats); per-update loss records are 4 floats × 2,000 per training branch. Training time is O(updates · batch · P) with P the dense parameter count. To be replaced by measured costs.

## Risks

- A first-moment reset changes the optimization regime for the first dozens of updates; a result is a statement about this recipe, in Adam only.
- Identical batches across compared branches make losses paired, but the trajectories diverge afterwards.
- `ctx-fresh` is confounded by large initial steps and is not a remedy test.
- Same four arms and the same checkpoint design as the previous study; no claim about other optimizers.

## Decisions for the researchers

1. Fresh seeds 340–349 (recommended, since the zero-gradient finding was seen on 330–339) or reuse 330–339.
2. Arms: the four above, or `dense` and `mixed-h125` only for half the cost.
3. What counts as an **acceptable cost** for the larger question (for the remedy comparison that follows): the draft suggests transfer (a) noninferior to `sparse48` at margin 0.05, retained recall above the no-remedy delay run by more than 0.05, and a declared storage and compute overhead; the thresholds are yours to set.
