# Relevant update v1: contract

Status: **draft, revised 2026-11-01 after external review. Nothing built or run; not frozen.** Stage 1 is specified to be frozen on its own. Stage 2 is a plan until stage 1's result is known, and stage 3 is only named. Values marked *provisional* may change once, on pilot seeds, before confirmation seeds exist, and every change is recorded under *Changes from the pilot*.

## Question

The bet: memory should update against **several relevant experiences together**, which takes a good update rule and useful experience selection. We have neither yet. So this tests them apart, before any stream:

1. **Stage 1, update rule.** Handed a known-good set of k experiences, it asks two distinct questions. **Which method acquires and preserves knowledge best,** and how much does contamination hurt it? And separately, **does processing experiences jointly help at comparable work,** with the same examples, exposures and step size?
2. **Stage 2, selection.** Given the current surprising event and a pool of past experiences, which rule chooses the set? That is, how should the gates γᵢ of the Omega rule be set?
3. **Stage 3, the pair in a stream**, against replay and recent windows, at gaps where traces can coexist.

**Why this order.** [Surprise lifecycle v1](surprise-lifecycle-v1-pilot.md) failed with retention, grouping, validation and the update tangled together. Its diagnosis found that traces could not coexist across its long gaps, that good fits rarely reached validation, and that even oracle sorting left the rank-one update damaging old knowledge. Stage 1 has no eviction, schedule or selection, so a failure there can only belong to the update or the representation.

**Scope.** Frozen keys and a shared linear or small neural reader, with structure visible in the cue. No host, drift or sparse readers, and no retention. A pass shows that an update and a selection can work here. It says nothing yet about sparse readers or a learned shared code.

## Material

As [surprise lifecycle v1](surprise-lifecycle-v1-contract.md): the frozen key encoder (839,040 bytes), one per seed; 512-symbol pool; 480 symbols per seed; cue of family, item, item (width 96, unit-norm, dot product = mean of three cosines); outcome key (width 32); `degraded` and `clean` renderings, 64 stream and 16 audit per symbol. Per seed, a fixed permutation assigns 8 known families, 36 available rule families, 4 other families, 32 outcomes, 300 stream items, 100 audit-only items.

**Seeds.** Pilot 380 and 381; confirmation 382–391, all unbuilt. Seeds 299 and 298 are not reused, since their results shaped this design. Surprise lifecycle v1's confirmation will not run, so the seeds it reserved are free.

**Old knowledge, the same start for every rule.** v1's procedure: 3,840 fit events (8 known rules × 384, 24 known bindings × 32); reader R by ridge on the rule events with λ = 1; 24 slots. MLPs by Adam at 0.001, batch 64, 10% held out, patience 20, at most 400 epochs, best epoch restored, optimizer state discarded. An architecture below 0.90 on known rules or known bindings on `clean` is rejected before selection.

## Stage 1: the update rule

No stream. A **trial** starts from old knowledge and applies one update per new rule, in order.

**New rules.** A rule is a family glyph with one fixed outcome, uniform over the 24 outcomes not used by known rules. Rules in a trial have different families. **Load** L, the rules per trial: **16**, 4, 32. Family order is permuted per trial.

**The set for one rule.** k experiences: that family, a fresh pair of stream items and renderings per experience, the outcome at a fresh rendering. k ∈ {1, 2, **4**, 8}.

**Contamination.** A share p ∈ {0, 0.25, 0.5} of the set, exactly round(p·k) positions, is replaced by one kind of contaminant, drawn fresh. Contaminated cells use k ∈ {4, 8} only.

| kind | contaminant | where it comes from |
|---|---|---|
| `unrelated` | a family not in the trial, random outcome | one-off noise |
| `same-outcome` | a family not in the trial, the rule's outcome | v1's chance residual groups |
| `same-family` | the rule's family, a random other outcome | an unreliable rule |

**Trials.** 8 per cell, per seed and condition. Draws are the same across rules for a cell, so rule contrasts are paired down to the examples. Families not in a trial remain unused and serve the breadth reading.

### Update rules

Each takes the set and the current memory. Every rule has a k = 1 form. Rules see keys only.

| rule | update, with e = o − answer | state beyond the shared start | corresponds to |
|---|---|---|---|
| `rank-one` | `R ← R + ē ĉᵀ/‖ĉ‖²` | none | **baseline**: v1's update |
| `ridge` | `R ← R + argmin_Δ Σ‖eᵢ − cᵢΔ‖² + λ‖Δ‖²` | none | least squares on a window; ATLAS Appendix G's linear case |
| `ridge-rehearse` | as `ridge`, plus a fixed buffer of r old events at weight w (below) | r × 512 bytes | experience replay |
| `ridge-interfered` | as `ridge-rehearse`, but using only the r events of a fixed buffer of B that a tentative `ridge` update damages most (below) | B × 512 bytes | inspired by maximally interfered retrieval |
| `ridge-orthogonal` | as `ridge`, then `Δ ← (I − UUᵀ)Δ`, with U a fixed basis of q old cue directions (below) | q × 384 bytes | low-rank form inspired by orthogonal weight modification |
| `expert` | keep (ĉ, ē); add `ē` to the reader's answer when `cos(c, ĉ) ≥ τ_x`, newest ties winning. R unchanged | 512 bytes per update | local gated memory, a soft slot |
| `omega` | MLP M. On the arrival of a set's j-th experience, one step: `M_t = (1 − α) M_{t−1} − η ∇ (1/j) Σ_{i≤j} ‖M(cᵢ) − oᵢ‖²`, γ = 1 | MLP fills the budget | Omega update, proceedings Eq. 5, gates off, no kernel, loss averaged rather than summed |
| `titans`, `titans-m0` | MLP M, for each experience `M_t = (1 − α) M_{t−1} + S_t`, `S_t = β S_{t−1} − η ∇‖M(c) − o‖²`; β = 0 in `-m0` | MLP fills the budget, S included | Titans Eqs. 12–14, constants for its functions |

`rank-one` and `ridge` update from the reader's residual, ignoring slots. α is decay, so 0 keeps everything. Momentum S persists across experiences and sets within a trial, as it would in a stream, and is zeroed at a trial's start. **Not included:** a polynomial kernel φ on the reader, since a degree-2 expansion of a 96-wide cue needs about 600 KB of weights, far beyond the budget; Muon, until gradient descent has been shown to be the limit; learned gates, which need outer-loop training with no sensory target defined yet. So `omega` is not a test of OmegaNet or ATLAS, and neither arm of Titans tests Titans.

**Memory construction for the protective rules.** All are built once per seed and condition from the old-knowledge fit events, are fixed for a trial and across trials, and never take in new experiences. So they protect old knowledge only. Damage among new rules is unprotected, and reported through retention.

- **Fixed buffer** (`ridge-rehearse` r events; `ridge-interfered` B events): equal numbers per known rule, drawn once from the 3,072 rule fit events. Known bindings are slots and need no rehearsal. r < 8 does not occur.
- **`ridge-rehearse`:** `Δ = argmin Σ_set ‖eᵢ − cᵢΔ‖² + w Σ_buffer ‖(oᵦ − cᵦR) − cᵦΔ‖² + λ‖Δ‖²`, under the current R, so a buffer event pulls Δ towards keeping or restoring its own answer.
- **`ridge-interfered`:** a tentative Δ₀ from `ridge`. Damage for a buffer event is `‖oᵦ − cᵦ(R + Δ₀)‖² − ‖oᵦ − cᵦR‖²`. The r largest enter the fit above in place of the fixed sample, with ties to the lower index.
- **`ridge-orthogonal`:** U is the top q eigenvectors of the uncentred second-moment matrix of the 3,072 cues, by a direct symmetric eigendecomposition of the 96 × 96 matrix, once, O(n·D_c² + D_c³). Only U is kept, never the matrix, and it is not refreshed. Since cue keys are correlated, a new rule's family direction may lie partly inside U, which would cost acquisition; rule generalization measures this.

**The two "together" comparisons.** Comparison A is the natural one, between algorithms as they would be used. Comparison B alone addresses whether joint processing helps.

- **A, natural:** a rule once on a set of k, against its k = 1 form k times on the same experiences in the same order: `ridge` against `ridge-turn`, `omega` against `titans-m0`, and likewise for the chosen rule. Exposures and work differ: `omega` makes k(k + 1)/2 exposures against k. So a gain may come from more work, and it is reported with its work.
- **B, matched: the same examples, exposures and total step size.**
  - **Neural: `omega` against `omega-seq`.** Where `omega` takes one step on the mean of j experiences at rate η, `omega-seq` takes j single-experience steps at rate η/j, on those same j experiences. So every prefix contributes the same exposures and the same total rate, with decay applied the same total amount. The only difference is whether gradients are evaluated at one state or sequentially. **[derived]** As η → 0 the two coincide, so any difference is a finite-step effect.
  - **Linear: `ridge` against `ridge-cycled`.** `ridge-cycled` passes over the set k times, each time solving each experience's single-example ridge step in turn, at the same λ. It uses k² example solves against the dual's k² inner products and k³ solve, so its work is of the same order, and it needs the set held throughout. **[derived]** With λ → 0 and enough passes this is Kaczmarz, converging to the same minimum-norm fit, so a gap measures what k passes leave undone.

Comparison B runs for these two families whichever rule is chosen, since it does not depend on the choice. `rank-one` and `expert` have no matched form and enter comparison A only.

### Budgets

Shared and outside the budget: encoder; reader or equivalent 12,288 bytes; known slots 12,288 bytes. Each rule's additional persistent state must fit in **64 KiB**, or that setting is rejected before selection. `expert` together uses L × 512 bytes, 16 KiB at L = 32. In turn it uses L × k × 512: 131,072 bytes at L = 32 with k = 8, over budget, and 65,536 bytes at L = 32 with k = 4 and at L = 16 with k = 8, which fit exactly. It has no eviction or merging, so the one over-budget cell is **infeasible and reported as such**. A `ridge-rehearse` buffer of r = 128 and `ridge-interfered` at B = 128 use exactly 64 KiB. The set itself is temporary workspace in stage 1, at most 8 × 512 bytes, reported apart; stage 2 charges it as the pool. MLP width fills 64 KiB + 24,576 bytes, counting weights and momentum; for `omega` the window also, at k = 8.

### Readings

Audit after the trial's last update, unless stated, on `clean` audit renderings with audit-only items, decoded by nearest of 32 outcome prototypes from the stream's condition. Primary cell: **`degraded`, L = 16, k = 4, p = 0.** Means over a seed's trials, then bootstrap over seeds.

**Primary.**

- **Rule generalization.** Macro over the trial's rules: 32 cues each.
- **Old knowledge.** Mean of known rules (32 cues each, audit-only items) and known bindings (8 renderings each).
- **Breadth.** Share of cues in unused families (32 per family) decoded to an outcome used by this trial's rules, minus the same share for the unchanged old reader. A rule that answers everything with a recent outcome shows up here: v1's best shadows included some that predicted a rule while fitted on none of its events.

**Descriptive.** Immediate rule generalization, right after a rule's own update, and retention to the end. Old knowledge by part. Other loads and conditions. Contamination tolerance p\*: per kind, the largest p at which rule generalization is at least 0.8 of its own p = 0 value with old knowledge within the limit, at k = 4 and 8.

**Compute.** Operations per update, measured µs per update, bytes by category.

### Settings, searched exhaustively on pilot seeds

| rule | grid |
|---|---|
| `ridge` and its variants | λ 0.01, 0.1, 1, 10 |
| `ridge-rehearse` | r 8, 32, 128; w 0.25, 1, 4 |
| `ridge-interfered` | B 128; r 8, 32; w 0.25, 1, 4 |
| `ridge-turn`, `ridge-cycled`, `omega-seq` | their partner's frozen setting, never searched |
| `ridge-orthogonal` | q 4, 8, 16, 32, 64 |
| `expert` | τ_x 0.3, 0.4, 0.5, 0.6, 0.7 |
| `omega`, `titans`, `titans-m0` | layers 1, 2; decay α 0, 0.0001, 0.001; η 0.001, 0.003, 0.01, 0.03, 0.1; `titans` β 0.5, 0.9 |

One setting per rule, frozen across k, p, L and condition. Grids may extend once if a choice lies on an edge, recorded. `rank-one` has none.

### Selection and decision rule

**Choice, pilot seeds, primary cell.** Among rule-settings with old knowledge ≥ 0.85 *(provisional)*, the highest rule generalization, ties to fewer operations. If none, the smallest shortfall on old knowledge, marked. The **chosen rule** is the single best setting across all rules other than `rank-one` and the controls `ridge-turn`, `ridge-cycled` and `omega-seq`, none of which can be chosen. Selection never looks at comparison B.

**Gain.** Confirmation seeds 382–391, fresh encoders, settings frozen. Paired by seed; percentile bootstrap over seeds, 10,000 resamples, fixed generator; 95% two-sided, bounds at 2.5% and 97.5%.

The chosen rule **beats** `rank-one` at the primary cell if its mean gain on rule generalization is at least 0.10 with lower bound above 0, its old knowledge has lower bound at least −0.02 and its breadth has upper bound at most +0.05 against `rank-one`.

The **bar** is absolute: mean rule generalization ≥ 0.70 and old knowledge ≥ 0.85 *(both provisional)*. For scale, v1's `life-oracle` read about 0.46 and 0.80, and the offline ceiling is about 0.87.

| outcome, first that applies | condition | next |
|---|---|---|
| `update-found` | chosen rule meets the bar and beats `rank-one` | stage 2 with this rule and its p\* as the target |
| `bar-reached-superiority-unestablished` | chosen rule meets the bar, without beating `rank-one` | stage 2: reaching the bar is what stage 2 needs. Use `rank-one` if it also meets the bar, being cheaper, else the chosen rule. No claim that the chosen rule is better |
| `rank-one-enough` | only `rank-one` meets the bar | stage 2 with `rank-one` |
| `improved-below-bar` | chosen rule beats `rank-one`, and neither meets the bar | report; stage 2 only with a stated reason |
| `no-rule-reaches-bar` | none of the above | perfect sets do not make these rules good enough on frozen keys. Selection cannot fix that, so stage 2 is not run; the next test is on real host codes. This does not establish a limit of all updates |

**Secondary, same bootstrap, each reading better, equivalent or inconclusive** (better: mean gain at least the margin, interval excludes 0; equivalent: interval within ±margin; margin 0.05):

- **Joint processing, comparison B, the answer to the second question:** `omega` against `omega-seq`, and `ridge` against `ridge-cycled`, at k = 4 and k = 8, on rule generalization and old knowledge. Reported with exposures and operations.
- **Natural algorithms, comparison A:** chosen rule, `ridge` and `omega`, each against its in-turn partner at k = 4 and k = 8. Reported with exposures and operations, and never read as joint processing helping by itself.
- **More evidence:** chosen rule at k = 4 against k = 1. This is expected to help and is not the together test.
- **Protection:** `ridge-rehearse` and `ridge-orthogonal` against `ridge`, on old knowledge.

**Expectations, recorded before any run.** `ridge` beats `rank-one`, since a rank-one direction with mean cue over few items keeps item noise. `expert` protects old knowledge best, and is broadest at low τ_x. In comparison A, `omega` beats `titans-m0`, largely through extra work. In comparison B, differences are small: at the small steps selection is likely to pick, `omega` and `omega-seq` nearly coincide. So a real benefit of joint processing may not be there to find, and that would be a useful result for the bet. `same-outcome` contamination hurts more than `unrelated`. `no-rule-reaches-bar` on `degraded` with only 4 examples is plausible: v1's residual fit on 8 reached about 0.55 there.

### Gates

On pilot seeds, before selection. Failure stops stage 1 until the design changes, and is recorded.

1. **Joint ceiling.** One R fit offline on old and new rules together, 384 events per rule, with known bindings as slots, audited on **both** rule generalization and old knowledge, at L = 4, 16 and 32, on both conditions. On `clean`, both must reach ≥ 0.85 at L = 16, or the study stops. A load failing that is descriptive only. On `degraded`, if the L = 16 ceiling on either reading is within 0.10 of the bar, that part of the bar becomes the ceiling minus 0.10, before any rule is compared and recorded here. Otherwise the bar could ask for what the map cannot hold.
2. **Start.** Old knowledge at the start reproduces v1's level, at least 0.85 on the mean of the pilot seeds at `degraded`, else the bar is revised before any rule is compared.
3. **Keys only.** Tests check that rules receive only keys, and that contaminant labels and outcome tables cannot reach them.
4. **Sets.** Tests check that k, p and kinds are as specified, that a contaminant never comes from a family in the trial other than for `same-family`, and that draws match across rules.
5. **Audit.** State is bitwise identical before and after an audit. Shuffled outcome keys score chance.

## Stage 2: selection (plan, frozen only after stage 1)

**Setting.** A fixed pool of K surprising events, K ∈ {64, 113, 256}, with no eviction, charged at 560 bytes each as in v1. Its contents follow a v1 stream's surprises: the target rule's n ∈ {2, 4, 8} events, other rules' events, bindings, one-off noise. An anchor, the current surprising event, is either a rule event or a noise event.

**Selectors,** each setting γᵢ ∈ {0, 1} for the top k of the pool, from keys only:

| selector | relevance to the anchor | corresponds to |
|---|---|---|
| `recent` | newest | Omega's window, γ = 1 |
| `random` | none | baseline |
| `cue` | cue similarity | nearest neighbours |
| `error` | residual similarity | v1's grouping |
| `cue-and-error` | both, by product of the clipped similarities, or both above thresholds | **the proposal** |
| `surprise` | largest error | prioritized replay |
| `gradient` | gradient overlap with the anchor's | inspired by gradient-based selection in Sleep and maximally interfered retrieval, with the opposite aim |
| `oracle` | hidden cause | ceiling |

**Coherence check.** Each selector may decline to update when the chosen set's agreement is low, so noise anchors can cause no update. Declining counts as no learning on a rule anchor.

**Readings.** Precision and recall of the set against hidden cause. Rate of updates on noise anchors. Then, with stage 1's rule, rule generalization, old knowledge and breadth, against `oracle`. **Pass:** a selector's precision reaches stage 1's tolerance (1 − p\*) at recall that leaves at least the k the rule needs, and its learning is within a margin of `oracle` while noise anchors do no measured harm. Exact numbers are set once stage 1 reports.

**Stage 3,** a stream at gaps with (m − 1)·g inside a trace's measured lifetime, against replay and windows at matched bytes, is a later contract. Long gaps return only after that, with an account of what holds evidence between events.

## Changes from external review

First review, before any build, all correct and fixed above:

- `α` was ambiguous: the equation read as retention while its grid read as decay. It is now decay, `(1 − α)`, in both rules, and momentum's persistence is defined.
- The "together" contrast mixed joint processing with extra exposures and gradient scale. It is now split into a natural comparison A and a matched comparison B, and the loss is averaged.
- A chosen rule could meet the bar and still read `no-rule-reaches-bar`. `bar-reached-superiority-unestablished` closes that, and licenses stage 2.
- The ceiling gate checked acquisition only. It now fits old and new jointly, audits both and can lower the bar.
- The protective rules' memory is now specified: fixed, built once, with damage score and basis algorithm given.
- `expert` in turn was over budget at L = 32, k = 8. That cell is now declared infeasible.
- Update costs were incomplete and are now complete.

## Changes from the pilot

None yet.

## Before freezing

- Full text, not abstracts, of orthogonal weight modification, GEM and A-GEM, and maximally interfered retrieval, and Sleep's gradient-based selection sources.
- The rest of the ATLAS proceedings on how γ is produced, which pages 5–6 and Appendix G do not specify.
- Whether the bar and the old-knowledge limit are attainable, from the gates and pilot.
- Whether the fixed buffers should also take in new rules; here they protect old knowledge only.

## Prior work

Shared survey: [prior-work.md](prior-work.md#relevant-update-several-experiences-together).

**Established names.** Update on a window with gates: Omega rule, ATLAS proceedings Eq. 5 **[read]**. One-step neural memory: Titans **[read: equations]**. Least squares on a context: ATLAS Appendix G **[read]**, attributing a recursive form to von Oswald et al.; only their abstract was read, and it does not mention one. Protecting old directions: orthogonal weight modification (Zeng et al.) **[abstract]**, GEM and A-GEM (Chaudhry et al.) **[abstract]**. Choosing what to rehearse by likely damage: maximally interfered retrieval (Aljundi et al. 2019) **[abstract]**. Selection by gradient: Sleep §3.4 **[read]**. Local gated memory: nearest neighbours with a threshold **[familiar]**.

**What follows without running [derived].**

- A ridge fit on k experiences with cue width 96 has rank at most k, so a rule's direction comes from at most k cues. With k = 4 and item noise in two thirds of the cue's weight, generalization to unseen items depends on the family part dominating the mean.
- Exact least-squares on a set is order-free, and taking the same examples in turn with exact fits is not. With small steps in turn, order hardly matters and the two approach one another, so "together" should help most when few, large steps are affordable.
- `expert` at low τ_x reduces to a broad rule, and at high τ_x to a slot. Its breadth should fall as τ_x rises, trading generalization for protection.
- The bar assumes 16 rules plus 8 known fit in a 96 × 32 linear map, which the ceiling gate checks. Their cue directions are correlated, with effective rank near 13 on 120 keys, which may limit it before L = 32.

**Known failure modes and handling.** Old-event rehearsal is uncounted memory if not charged: it is charged. Selecting the best of many rules on two seeds: one chosen rule, fresh seeds. Winning by answering everything: breadth. A win that depends on the audit's cue similarity to the fit: audit-only items. The rule visible in the cue makes selection easier than a real world: stated in scope and carried to stage 3.

**Baselines.** `rank-one`, one-step neural memory and the in-turn form of every rule are included. Stream methods wait for stage 3, since they are not comparable without a stream.

**What this adds.** A clean separation of update and selection on sensory keys, and a same-example test of together against in turn. MIR selects memories to protect them from an update; the proposal here selects experiences to form one. That difference comes from these abstract-level readings, so it is provisional. No novelty claim.

## Scaling

Per update: k experiences, D_c = 96, D_o = 32, r rehearsal events, q directions, X experts, W MLP weights.

| rule | work per update | persistent | what limits scale |
|---|---|---|---|
| `rank-one` | k·D_c·D_o residuals, k(D_c + D_o) means, D_c·D_o apply | none | none in itself; interference grows with rules |
| `ridge` | k·D_c·D_o residuals, k²D_c + k³ by the dual, k·D_c·D_o to form Δ, D_c·D_o apply | none | k cubed; small for small sets |
| `ridge-turn` | k × `ridge` at k = 1 | none | as `rank-one` |
| `ridge-cycled` | k² × `ridge` at k = 1 | none, but holds the set | passes |
| `ridge-rehearse` | as `ridge` at k + r | r(D_c + D_o) | rehearsal grows with what must be kept safe |
| `ridge-interfered` | a tentative `ridge`, 2B·D_c·D_o to rank, then `ridge` at k + r | B(D_c + D_o) | buffer |
| `ridge-orthogonal` | as `ridge` + 2q·D_c·D_o to project; once at construction n·D_c² + D_c³ | q·D_c | q; a fixed subspace does not protect later rules |
| `expert` | k·D_c·D_o residuals, k(D_c + D_o) means; X·D_c added per read | X(D_c + D_o) | **linear in rules**, like slots |
| `omega` | k(k + 1)/2 · 2W exposures | W | k² |
| `omega-seq` | the same exposures, as single steps | W | k² |
| `titans` | k · 2W | W or 2W | none in itself; capacity unknown |

- **Facts.** The reader's size is fixed, so any rule writing into R has a capacity limit in rules; `expert` trades that for linear growth. Stage 1 measures this through L = 4, 16, 32 **[probes]**.
- **What would not survive scale-up.** A family glyph handing the rule its feature; 32 outcomes decoded through an evaluator's table; 36 rule families; rules with one fixed outcome. A rule whose outcome depends on an item leaves no shared residual, so neither this test nor `cue-and-error` covers it.

## Computation plan

*Estimates, not measurements.*

- **Construction.** Encoders for 12 seeds, about 70 s each on the GTX 1650. Key pool under a minute per seed and condition. Old knowledge: ridge O(n·D_c² + D_c³) once per seed and condition; MLP O(epochs × n × W) once per architecture, seed and condition.
- **Stage 1 pilot.** 18 cells (p = 0 at four k; 12 contaminated; in turn at k = 4 and 8) × 8 trials × 16 updates at the primary load, about 2,300 updates per setting, seed and condition. 89 linear settings at about a second each and 120 neural ones at a few seconds. Expected under 30 CPU-minutes, with rejected settings counted. v1's estimate was about 3× low.
- **Stage 1 confirmation.** Chosen rule, `rank-one` and the others at frozen settings, 10 seeds, all cells, both conditions, three loads: under an hour of CPU.
- **Deterministic and timed output kept apart.** `reports/relevant-update-v1-{gates,pilot,confirm,analysis}` and separate `-runtime` files, per [computation reporting](computation-reporting.md). Peak workspace: pool 7.9 MB per seed, one set, the audit cues.
- **Persistence.** Tasks durable (`experiments/durable.py`); rerunning resumes. Pools written atomically under gitignored `runs/`.
- **Dtype.** float32. Raw images are not retained after encoding.

## Reproduction

To be written with the implementation: `src/relevant_update.py`, `experiments/relevant_update.py`, `tests/test_relevant_update.py`.
