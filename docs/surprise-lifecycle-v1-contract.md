# Surprise lifecycle v1: contract

Status: **draft, revised 2026-11-01 after two external reviews. Nothing built or run; not frozen.** Values marked *provisional* may change once, on pilot seeds 299 and 298, before confirmation seeds exist. Every such change is recorded under *Changes from the pilot*. The gates and the items under *Before freezing* must be cleared first.

## Question

Does keeping unresolved surprises, and letting later independent experience decide what they are, let a bounded memory learn more than replay or immediate updates at the same bytes, when related experiences are far apart?

The user's [accumulated-surprise proposal](continuous-sensory-memory-notes.md#accumulating-surprises-before-changing-shared-knowledge) is tested at its cheapest. Traces gain support from independent experience, drive an update, and otherwise fade. The test also examines three parts of the proposal separately: the later-experience test, grouping by shared error rather than similar cues, and a cheap familiarity sketch.

**Scope.** No host network, no encoder training, no drifting perception, no sparse reader. "Shared knowledge" here is a linear reader on frozen keys, so a passing result shows that the evidence process works on structure the existing features can express. It does **not** show that a representation should reorganize, or that the sparse-network tradeoff is solved. Those need a later experiment and are only licensed by a pass here.

## Material

- **Sensor.** The frozen key encoder of the [stable key revision](translation-memory-v1-pilot.md#stable-keys-perception-gate-passes) (base network, 32-dimensional unit-norm key, 839,040 bytes, identical in every arm), trained on background symbols only.
- **Symbols.** The task's 512-symbol pool, which no encoder trained on. Per seed, a fixed permutation assigns 48 family glyphs (8 known, 4 new-rule, 4 binding, 32 background), 32 outcome glyphs, 300 stream items and 100 audit-only items.
- **Event.** A cue and usually an outcome. The cue is always three glyphs, a family and two items, encoded separately and concatenated (width 96, cosine = mean of the three key cosines). The outcome is one glyph's key (width 32). Every event is a sensory rendering. No arm sees a word, class, cause identity, or the table of outcome keys.
- **Renderings.** 64 per symbol, encoded once per condition. Audit renderings are disjoint from stream renderings. The pool simulates the world and is not memory of any arm.

## World

Before the stream, every arm is given the same **old knowledge**: 8 known rules (a known family with any items gives that family's outcome) and 24 known bindings (a specific cue gives a specific outcome), by the procedure under *Old knowledge*, at that class's cost.

**Hidden kinds.** Labels belong to the evaluator and scheduler only.

| kind | count | what happens | occurrences |
|---|---|---|---|
| `rule` | 4 | new family glyph, different stream items each time, one fixed outcome. Audit uses audit-only items | 8, plus 2 `rule-noise` |
| `rule-noise` | 2 per rule | that rule's family, fresh items, a random other outcome, at random positions. A rule is not perfectly reliable, and the failure limit must tolerate it | 2 |
| `binding` | 12 | one fixed cue in a binding family, one fixed outcome | 8 |
| `exception` | 6 | one fixed cue in a known family, one fixed outcome other than the family's | 8 |
| `rare` | 12 | as `binding` | 2 |
| `noise-repeat` | 8 | one fixed cue in a binding family, a fresh random outcome each time. Recurs as often as a binding | 8 |
| `noise-burst` | 4 | one fixed cue in a binding family, identical outcome 6 times within 32 events; a different outcome in each later episode | 4 episodes |
| `noise-once` | background | a cue never used before or again, a uniformly random outcome | rate q |
| `query` | background | a cue with no outcome, uniform over cues already shown | 10% |
| `known` | the rest | old knowledge, correctly predicted | |

**One-off cues.** The same form as every other cue: family, item, item, sampled without replacement. The family is uniform over the families active at that moment, among known, binding and background, and never a rule family. Available cues are 44 × 300 × 299, about 3.9 million, against 65,536 needed at g = 32,768, q = 0.2, so no cue form or glyph position distinguishes noise from causes. A cue sharing only its family with another has `cos ≈ 0.5`, and family plus one item ≈ 0.75; τ must lie above those, and the construction check reports the share of one-off cues within τ of a cause. Exhausting the set is a build error, not a resample.

**Why rule families take no one-off noise.** With uniform families, a rule family would see about 1,500 contradicting events against 8 real ones at the longest gap, and the rule would simply not be true in that world. Background families play the matching role: they look like rule families but their outcomes are incoherent.

**Family activation.** Each rule family starts at a random phase in the first period. Each background family activates at an independent uniform time in the same range, so a never-seen family does not announce a rule. Among families new to the stream, 4 of 36 are rules, so a never-seen family gives a rule with posterior about 0.11 **[derived]**. That signal exists and is disclosed, not removed: a new rule does begin with a new family. No arm uses glyph novelty directly.

**Schedule.** Occurrences of a cause lie g events apart, ±10% uniform jitter, with random phase. The stream runs 8 periods plus a tail of 2g events with no scheduled causes, so its length is L = 10g. Burst episodes are spaced by g.

| factor | values |
|---|---|
| gap g | 128, 512, 2,048, 8,192, 32,768 |
| one-off surprise rate q | **0.2**, 0.05 |
| temporary budget B | 16, **64**, 256 KiB |
| condition | **`degraded`**, `clean` |

`degraded` is noise 0.06 with occluder 4, the [streaming retrieval](streaming-retrieval-v1-contract.md) level (single-frame 120-way accuracy 0.65). `clean` is independent renderings. Bold marks the primary cell. Causes occupy 368 events of L, so 29% of the stream at g = 128 and about 0.1% at g = 32,768.

**Not in this world:** a known rule changing, conjunctions needing a feature the sensor lacks, rules whose outcome varies with the item, cues that change over time, delays with no events. Gap always means intervening events; idle time is untested, since no arm has a clock.

## Common machinery

**Reader.** R, 96 × 32 linear, ridge-fit on old knowledge. Update `R ← R + η·w·(o − Rc)cᵀ/‖c‖²`.

**Answer.** If a slot's cue has cosine at least τ with c, the most similar slot's outcome; otherwise Rc.

**Match.** Everything an arm scores is observable: `hit(p, o)` means `cos(p, o) ≥ 1 − θ` for a prediction p and an observed outcome key o. An event is *surprising* if the answer misses it. θ is a single constant per arm. Only the audit decodes through the evaluator's outcome table.

**Old knowledge.** Per seed and condition, once, cached, identical for every setting. The fit set is 3,840 events from stream renderings: 8 rules × 384 events, with random items in both slots, and 24 bindings × 32 events.

- **Reader.** Ridge in closed form on the 3,072 rule events only, λ = 1, target the outcome key. The 24 known bindings are slots, written from the first event of each. Comparator and lifecycle arms all start from this R and these slots.
- **MLP.** Fit to all 3,840 events by Adam, learning rate 0.001, batch 64, mean squared error, initialization from the seed's stream. 10% of the set is held out. Stop when held-out loss has not improved for 20 epochs, or at 400 epochs, restoring the best epoch's weights. Adam's state is **discarded** before the stream, since inherited moments alone cost recall in the [forgetting study](forgetting-mechanism-v1-results.md). Each distinct architecture is fit once.
- **Target.** Gate 1 requires each class to reach at least 0.90 on known rules and known bindings on audit renderings before the stream. An MLP architecture that does not is rejected before selection.

**Bytes.** Identical outside B in every arm: the encoder, 12,288 bytes for the reader or its equivalent, and 12,288 bytes for the 24 known bindings (512 each). MLP arms get those last two as weights, so every arm has the same total. Inside B:

| item | bytes | notes |
|---|---|---|
| lifecycle trace | 560 | cue and residual 512 (the residual under the current R is stored instead of the outcome, the same bytes, since o = e + cR), trace counters and flags (pinned, consumed, familiar) 16, and one group record 32, reserved so group state can never overflow |
| shadow | 528 | ē, ĉ 512, counters 16 |
| new slot | 512 | |
| sketch entry | 8 | the projection is regenerated from the seed and not stored |
| FIFO or reservoir event | 516 | |
| MLP | 4 per weight, momentum or window float | |

**Allocation.** S shadow places are reserved first. `life-sketch` gives half of B to the sketch first. Traces and new slots share what remains; slots may take at most half of that, and past the cap the least recently used *new* slot goes. A new slot evicts the lowest-priority trace if needed. A configuration is **rejected before selection** if it holds fewer than S·m + 4m traces (a shadow's fit traces are pinned, below), or a slot cap below 12. Fixed scalars (write pointers, the reservoir count) add under 64 bytes and are reported.

| B, S = 4 | `life` traces | `life-sketch` traces, entries | slot cap | FIFO events |
|---|---|---|---|---|
| 16 KiB | 25 | 10, 1,024 | 13 | 31 |
| **64 KiB** | 113 | 54, 4,096 | 61 | 127 |
| 256 KiB | 464 | 230, 16,384 | 253 | 508 |

The slot cap shown is `life`'s; `life-sketch` has half that or less (5 at 16 KiB, 29 at 64 KiB, 125 at 256 KiB). So `life-sketch` is infeasible at 16 KiB for every setting, and that cell reports `life` only. B = 16 KiB is descriptive, so no verdict depends on it.

**Settings.** One per arm, frozen across all gaps, q and B, since a real system is not told the gap.

## Arms

**Lifecycle family.** Same trace store and eviction.

| arm | difference | role |
|---|---|---|
| `life` | the mechanism below | **candidate** |
| `life-sketch` | `life` plus a familiarity sketch | **candidate** |
| `life-cuegroup` | residual groups replaced by groups of similar cue (`cos ≥ τ_g`, below τ). Shadows, validation and reader updates unchanged | is grouping by error needed? |
| `count` | promotes on support count alone, no validation | is the later-experience test needed? |
| `count-sketch` | `count` with the sketch | does the sketch only add false matches? |
| `life-selfcount` | a query's own answer counts as a hit for eligible groups | does retrieval manufacture support? |
| `life-oracle` | hidden cause identity replaces grouping and routing; same stores, same fits | diagnostic of headroom. Its remaining fitting and budget limits mean it is not a proven upper bound. Never a comparator |

**Comparators.** Each uses the full budget.

| arm | update | corresponds to |
|---|---|---|
| `frozen` | none | floor |
| `slow` | R step on every event with an outcome, small η; no store beyond known bindings | slow learning; no traces |
| `surprise-fifo` | R step with w = 1 − cos(ô, o); FIFO store of surprising events answers lookups, newest wins | immediate surprise-weighted update |
| `fifo-replay` | FIFO of all events with outcomes; r replay steps per event; lookups | bounded recent window on the store |
| `reservoir-replay` | as above, uniform sample of all events (Vitter's Algorithm R) | ordinary bounded replay |
| `titans-m0`, `titans` | MLP M, `M_t = (1 − α)M_{t−1} + S_t`, `S_t = βS_{t−1} − θ_M∇‖M(c) − o‖²`; β = 0 in `-m0` | inspired by [Titans](https://arxiv.org/html/2501.00663v1) Eqs. 12–14 |
| `omega` | MLP M, one gradient step per event on the last c events, γ = 1 | ATLAS's Omega update ([arXiv](https://arxiv.org/pdf/2505.23735) Eq. 9, [proceedings](https://raw.githubusercontent.com/mlresearch/v306/main/assets/behrouz26b/behrouz26b.pdf) Eq. 5), with gates off and no kernel |
| `sleep-periodic` | FIFO of all events with outcomes answers lookups. Every P events, ridge-fit the store's residuals into R, penalized toward the current R, with no selection; then clear the store | scheduled, unselective consolidation, inspired by [Sleep](https://arxiv.org/abs/2606.03979) §3.2–3.3 |

MLP width is whatever fills the total budget, counting weights, momentum and window. These three are **inspired by, not tests of,** the published systems. Their α, β, θ_M and γ are constants rather than functions of the input, there is no attention or outer-loop training, and ATLAS's feature mapping and optimizer are left out. A loss here does not count against those papers. Omega's input-dependent γ can prune inside the window, which a constant γ cannot.

`sleep-periodic` keeps only Sleep's schedule and lack of selection. It has no distillation, dreaming, reinforcement learning, memory chain or growth, so it too is not a test of that system. **`sleep-grow`**, descriptive only and outside the rule, adds a new rank-r expert at each sleep instead of changing R, with the sender's store cleared and old experts frozen. Its bytes grow with sleeps and are reported, since it cannot meet a fixed budget: about 5 MB at r = 4 on the longest stream. It shows what growth alone would buy.

### The mechanism

Eligibility is decided from the cue alone, before the outcome is seen. Every eligible event is scored, so a group can never select the outcomes it is judged on. Grouping uses the outcome, but only to propose a candidate, never to validate one.

On each event with an outcome, in this order. Steps 1–5 run on every such event; 6–8 only if the event is still surprising. A query has no outcome and runs none of them, except in `life-selfcount`.

1. **Fix eligibility and answer from c, before the outcome is read.** A same-cue group is eligible if `cos(c, its cue) ≥ τ`. A shadow is eligible if `cos(c, ĉ) ≥ τ_s` and c is not within τ of one of its fit traces.
2. **Score everything, surprising or not, against the state at step 1.** Same-cue group, if eligible: *hit* or *miss* for its mean outcome. Shadow, if eligible, on the reader path alone (slots ignored): *win* if R + Δ hits and R misses, *failure* if R + Δ misses, otherwise neutral. Every shadow, eligible or not: *damage* if R hits and R + Δ misses, and *R-hit* counted if R hits. An *episode* is at least E events after a group's last counted one. Hits and wins count once per episode. **Misses, failures and damage always count.**
3. **Decide, in this order, using the counts just updated.**

| order | evidence | action |
|---|---|---|
| a | same-cue misses ≥ h_n | drop its traces (`noise-repeat`) |
| b | same-cue hits ≥ h_b, misses 0 | write a slot with the group's mean outcome, drop its traces (`binding`, `exception`) |
| c | shadow failures ≥ h_f, or damage above δ of its R-hits once it has 64 | **retire** the shadow |
| d | a remaining shadow with wins ≥ h_r and damage at most δ of its R-hits (0 R-hits counts as 0) | `R ← R + Δ`, drop its traces, delete the shadow (`rule`). At most one per event: the most wins, then the oldest |

Groups in a and b, and shadows in c, go oldest first. If a or b would drop a pinned trace, that trace leaves the same-cue group but stays until its shadow leaves.

4. **If R changed,** recompute each residual group's e from its traces under the new R. Traces R now hits leave the group. Any remaining shadow is refit on those of its fit traces that remain, topped up with the group's oldest *unconsumed* traces from distinct episodes, and repinned, with all its counts **reset to zero**, since it is now a different candidate. A shadow that cannot reach m traces retires. Slot writes do not change R, and shadow scores use only the reader path, so they cause no reset.
5. **Recompute the answer under the state after steps 3–4, and stop if it hits.** No trace is kept for what has just been absorbed.
6. **Keep the evidence.** Append a trace holding cue and outcome, not a scalar. Join a *same-cue* group if `cos(c, its cue) ≥ τ`. Join a *residual* group if `cos(e, ē_G) ≥ τ_e`, where `e = o − Rc`: experiences wrong in the same way, whatever they look like. A trace is in at most one of each, and never moves to another; it only leaves.
7. **Propose.** A residual group without a shadow, holding m **unconsumed** traces from distinct episodes, whose unit cues have mean resultant length at least κ, freezes a rank-one shadow `Δ = ē ĉᵀ/‖ĉ‖²` fitted on the oldest such m. Those fit traces are **pinned** for the shadow's life: they count toward the trace budget and cannot be evicted, so eligibility can always be checked and the exclusion needs no uncounted memory. κ rejects chance groups, which a rank-one update could not generalize from anyway. If all S places are taken, the shadow with the fewest wins, then the oldest, is displaced. At most one proposal per event, oldest group first.
8. **Evict if over budget,** by the order below, and on a slot write likewise.

**Retiring and retrying.** A shadow leaves by retiring, by being displaced, or by promotion. On retirement or displacement its fit traces unpin and are marked **consumed**. They remain ordinary members of the group, still contribute to ē, and may be evicted, but can never fit a shadow again. So a rejected candidate can return only on m **previously unused** traces: none has ever fitted a shadow in this group, though some may predate the earlier fit, since a group can already hold unused traces when it is fitted. The invariant is that a trace fits at most one shadow, step 4's refit of the same shadow excepted. Its failures cannot be cleared by reproposing it, and no fit repeats on the same evidence. A displaced shadow is treated the same, so displacement cannot launder failures either. The number of tries per group is limited only by the evidence arriving, and is reported by hidden kind, since each retry on noise is another chance of a false promotion. The reset in step 4 is the one case where counts clear without new traces. It occurs at most once per promotion, and it is disclosed because it could rescue a failing shadow.

**Eviction order.** Among unpinned traces, by the key (support, familiar, age), lowest support first, unfamiliar before familiar, oldest first. *Support* is the higher of the distinct-episode counts of the trace's two groups, 0 for a singleton. *Familiar* is the sketch's bit, below. So singletons without a sketch match leave first, in FIFO order, which is what the reach calculation assumes.

**Membership.** An evicted trace leaves both its groups, whose means and resultant are computed from the traces remaining. A group keeps its counts (hits, misses, episodes, last counted event, tries) while it holds at least one trace. They are not reduced by eviction, since they record evidence seen. A group with no traces is deleted with its counts, which is how evidence is forgotten.

There is no time-based expiry, since a clock would decide the gap by another name.

**Sketch.** A ring of 8-byte entries: 32 sign bits of a fixed projection of the cue, 32 of the residual under R at writing, so old residual bits go stale when R changes. Every surprise looks up, then writes one, so an event never matches itself. A match within Hamming radius ρ, on either half, sets the trace's **familiar** bit. This bit is held per trace, so it survives regrouping and group deletion, and it is used **only** as the eviction tie-break above. It never counts as support, an episode, a hit, a win, or toward m, κ, h_b or h_r, in `life-sketch` or `count-sketch`. Familiarity buys a trace time to meet independent confirmation and is not itself confirmation. Combining them is not tested here.

**Provenance.** Each trace comes from a sensory outcome; `query` events have none. Only `life-selfcount` breaks this rule.

### Provisional settings, in search order

| setting | grid | start |
|---|---|---|
| θ | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 | 0.3 |
| τ | 0.7, 0.8, 0.9, 0.95 | 0.8 |
| τ_e | 0.3, 0.5, 0.7, 0.9 | 0.7 |
| τ_s | 0.3, 0.4, 0.5, 0.6 | 0.5 |
| κ | 0.4, 0.5, 0.6, 0.7 | 0.5 |
| m | 2, 3, 4 | 3 |
| h_b, h_r | 1, 2, 3 | 2 |
| h_f | 2, 3, 4 | 3 |
| h_n | 2, 3 | 2 |
| S | 2, 4, 8 | 4 |
| δ | 0.02, 0.05 | 0.02 |
| E | 16, 64 | 64 |
| ρ (sketch) | 4, 8, 12 | 8 |
| τ_g (`life-cuegroup`) | 0.4, 0.5, 0.6 | 0.4 |
| η | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.03 |
| r | 1, 4 | 1 |
| P | K, K/2 | K |
| ridge λ | 0.1, 1, 10 | 1 |
| expert rank (`sleep-grow`) | 2, 4 | 4 |
| MLP | layers 1, 2; α 0, 0.0001, 0.001; β 0.5, 0.9; θ_M 0.001, 0.003, 0.01, 0.03; c 16, 64, 256 | 2, 0.0001, 0.9, 0.01, 64 |

## Readings

An audit reads a copy of the state, on audit renderings, never changing the arm. It runs at quarter points and at the end; the primary value is at the end, after the tail. Answers are decoded by the nearest of 32 evaluator-held outcome prototypes (chance 1/32), each the mean of that glyph's audit renderings **in the stream's own condition**, since an arm's predictions live in the key space it learned. Primary audits use `clean` renderings, even in a `degraded` stream, so they measure what was learned rather than one noisy read. `degraded` audits are also reported.

**Primary.**

- **Rule generalization.** `rule` families with audit-only items.
- **Binding recall.** `binding` and `exception` cues, macro-averaged over cues.
- **Noise capture.** The same behavioural reading in every arm, whichever part holds the memory, including anything absorbed into R. Target: per cue, the **last outcome** that cue was shown with. Cues: all `noise-repeat`, all `noise-burst`, and up to 128 `noise-once` cues drawn uniformly from the first 80% of the stream, the same cues for every arm of a seed and cell. Each is audited on 8 renderings. Per kind: `P(answer = target) − P(answer = target of a different cue of that kind)`, the second term being the empirical null, over a fixed derangement. Reading: the mean of the three kinds. This measures whether the arm repeats old noise as knowledge. It cannot separate a memory held as episodic from one taken as true, since no arm abstains.
- **Old knowledge.** Macro-average of known rules on audit-only items, known bindings, and each `exception` family on audit-only items.

**Descriptive.** Slots promoted per hidden kind (the arm's own routing, reported separately from noise capture), `rare` recall, occurrences to first correct, slots and traces held, shadows proposed, retired, displaced and promoted, tries per group by hidden kind, resets rescuing a shadow that later promoted, sketch true and false match rates against hidden identity, curves over the quarter points.

**Compute.** Operations per event (derived per arm), including the read every arm needs to answer every event: multiplications plus one for each 64-bit Hamming comparison. Measured µs per event on one thread, every byte by category.

## Selection and decision rule

**Search.** Pilot seeds 299 and 298, primary cell, all five gaps, coordinate descent. Lifecycle settings are searched once on `life`, in the table's order, starting from the table's values; `count`, `count-sketch` and `life-selfcount` inherit them, adding only what they alone use (h_b and h_r as support thresholds for `count`). Then ρ on `life-sketch`, τ_g on `life-cuegroup`. Each comparator is searched on its own settings. A pass visits each setting once, holding the rest. Search ends after a pass with no change, or after 3 passes, or at 1,500 scored configurations per arm, whichever is first, and the budget's use is recorded.

**Objective.** Among configurations meeting the provisional limits, noise capture ≤ 0.10 and old knowledge ≥ 0.85: the highest mean over gaps of (rule generalization + binding recall)/2, then fewer operations. If none meets them, the smallest largest relative excess over a limit, then the objective; the arm is marked and still enters the rule.

**Candidate.** Of `life` and `life-sketch`, the higher pilot objective restricted to long gaps. Only that one enters the outcomes. The other is reported through the sketch contrast.

**Comparable cost.** At most 6× the mean operations per event of `reservoir-replay` at r = 1 *(provisional)*.

**Confirmation.** Seeds 380–389, fresh encoders and permutations, settings frozen. Differences paired by seed; percentile bootstrap over seeds with 10,000 resamples and a fixed generator; 95% intervals, two-sided, bounds at 2.5% and 97.5%.

The candidate **beats** a comparator on a reading at a gap if the mean gain is at least 0.10 and the lower bound exceeds 0, and the guards hold: lower bound of the difference ≥ −0.03 on the other of rule generalization and binding recall, ≥ −0.02 on old knowledge, and upper bound ≤ +0.02 on noise capture.

The **long gaps** are 2,048, 8,192 and 32,768, each beyond the reach of every full-store arm at the primary cell (see below). Comparators are `slow`, `surprise-fifo`, `fifo-replay`, `reservoir-replay`, `titans-m0`, `titans`, `omega` and `sleep-periodic`. A reading is **won** if the candidate beats **all eight at the same gap**, at **at least two** long gaps. Requiring all comparators is conservative and needs no adjustment across them **[derived: intersection-union]**; the choice of two of three gaps is not adjusted.

| outcome, first that applies | condition | meaning |
|---|---|---|
| `earns-role` | rule generalization won, at comparable cost | unresolved surprises are a route to shared updates here; the sparse-network experiment is licensed |
| `binding-store-only` | binding recall won, rule generalization not, at comparable cost | traces work as a binding store; no route to shared updates was established |
| `higher-cost-only` | either won, above comparable cost | report the frontier |
| `no-advantage-shown` | neither | the lifecycle has no claim over ordinary means on this evidence. This is a failed superiority test, not proof that replay suffices; the intervals are reported |

**Secondary contrasts**, same bootstrap, margin 0.05 on rule generalization and binding recall, 0.02 on noise capture and old knowledge. Each reads three ways at each long gap: **better** (mean gain at least the margin, interval excludes 0), **equivalent** (interval within ±margin), otherwise **inconclusive**.

- **The later-experience test:** `life` against `count`, on noise capture and old knowledge.
- **Grouping by error:** `life` against `life-cuegroup`, on rule generalization and noise capture.
- **Sketch:** `life-sketch` against `life`, and `count-sketch` against `count` on noise capture.
- **Self-support:** `life-selfcount` against `life`, on noise capture.

**Headroom,** reported with every outcome and never overriding one: `life-oracle` minus the best comparator, on rule generalization and binding recall, at each long gap, with intervals. This shows how much room sorting had. A shortfall does not prove sorting cannot help, because of uncertainty, margins and the oracle's own limits.

Everything else (other q, B, `clean`, short gaps, `rare`, `sleep-grow`) is descriptive.

**Expectations recorded before any run:** `life` is better than `count` on noise capture. `life-cuegroup` may match `life`, because the family glyph makes a rule visible in the cue, so this world does not favour grouping by error; equivalence here would not rule it out for structure the cue hides. 8 small steps leave `slow` weak at long gaps. 32,768 defeats all but possibly the sketch. `sleep-periodic` learns `rule` better than `slow` but absorbs noise into R, since it consolidates everything. `binding-store-only` is at least as likely as `earns-role`.

## Gates

Each on pilot seeds, before selection. Failure stops the study until the design changes, recorded under *Changes from the pilot*.

1. **World.** R fit offline on all `rule` events, each new rule's 8 events weighted to match a known rule's 384, reaches rule generalization ≥ 0.85, and slots written once per cause reach binding recall ≥ 0.85, on `clean`. A residual fit on those 8 events per rule alone, as a shadow's update would make, is also reported. This shows a win is possible and sets the ceiling. `degraded` values are recorded.
2. **Separability of surprise.** With R and slots as fitted on old knowledge, AUROC of `1 − cos(ô, o)` between `known` events and first occurrences of causes must be ≥ 0.90 on `clean`. At `degraded`, if below 0.60, the primary cell cannot test the mechanism, and `clean` becomes primary before confirmation.
3. **Construction.** No one-off cue repeats or matches a reserved cue; all cues have the same form. Every configuration in the search fits its budget, by asserted byte counts. A disclosure table reports, by kind, family novelty and glyph counts at first occurrence, and the posterior of a cause given each, against the 0.11 expected.
4. **No privileged input.** Arm code receives only keys, in stream order. Tests check that a stream's outcome table, kind labels and cause ids cannot reach arm code, `life-oracle` excepted.
5. **Audit.** Bitwise-identical state before and after every audit. Shuffled outcome keys score chance; the noise-capture null is near 0 on `frozen`.
6. **Feasibility, on pilot seeds only, separately defined.** On rule generalization or binding recall, `life-oracle` has a mean gain of at least 0.10 over the best comparator at one or more long gaps. If not, the design is changed or the study stopped **before** confirmation, and that is recorded. This is a judgment on two seeds about whether a costly confirmation is worthwhile; it is not an outcome.

## Changes from external review

Before any build. The review found these; all were correct and are fixed above:

- The longest stream needed about 65,536 one-off events from 4,800 family–item cues. One-off cues now draw from ordered glyph pairs without replacement.
- Shadow scoring used "known" events and decoding, both evaluator-only. Hits are now cosines with the observed outcome, and damage is on events R itself hits.
- Validation events were selected through the outcome. Eligibility is now cue-only, all eligible failures count, and expiry and the effect of R changing are specified.
- False consolidation differed by arm and missed noise in R. It is now one behavioural reading, noise capture, with a stated target, null and sampling; slot promotion is reported apart.
- `life-samecue` froze R, so it could not test grouping. `life-cuegroup` changes only the grouping criterion.
- Budgets were infeasible at 16 KiB and left out group state. Known bindings now sit outside B in every arm, the full allocation is given, and infeasible configurations are rejected before selection.
- Selection and verdicts were left open. Search order, stopping, infeasible settings, the AUROC gate, intervals and same-gap requirement are now specified, and secondary contrasts read better, equivalent or inconclusive. `replay-suffices` became `no-advantage-shown`.
- Cost now counts Hamming comparisons, and complexity and workspace are stated.

Added after reading [Sleep](https://arxiv.org/abs/2606.03979), since it names an established alternative: `sleep-periodic` and `sleep-grow`.

Clarified after the third review: retry eligibility and fingerprint horizon against learning reach.

Added while revising: the κ requirement, since chance residual groups turned out to be common by arithmetic (see below).

Second review, also before any build, all correct and fixed above:

- The enlarged one-off pool put an item first in 95% of noise, against a family first in every cause, so a cue alone could tell them apart. Every cue now has the same three-glyph form, which also raises widths and all byte figures. The same check showed that family-first noise would have swamped rule families, so `rule` families now take only `rule-noise`, and background families and staggered activation take the rest. The remaining novelty signal is disclosed.
- The `uninformative` outcome read too much into a failed superiority test and could override a passing candidate. It is gone: headroom is reported, and a separate pilot feasibility gate decides whether to confirm.
- Shadow validity under eviction is now specified, by pinning fit traces, as is eviction's effect on group membership and counts.
- Old-knowledge fitting is now specified: the set, its size, ridge, the MLP optimizer, stopping and target.

Third review, also before any build, all correct and fixed above. The event loop is now one ordered procedure:

- A rejected shadow could be recreated unchanged with its failures cleared. Fit traces are now consumed on retirement or displacement, so a retry needs m previously unused traces.
- The early return skipped decisions on events the answer hit, so retirement depended on when an unrelated surprise came. Scoring and decisions now run on every outcome, and only keeping a trace is conditional.
- Sketch support conflicted with group support in eviction. Familiarity is now a per-trace tie-break only, and never counts as confirmation.

## Changes from the gates

Gates first run 2026-11-01, on seeds 299 and 298, before any pilot search. Gate 1 failed as first written; all three causes were procedural, and these changes came before any arm was compared:

- **Gate 1's fit.** 32 new rule events were drowned by 3,072 known ones in one ridge fit: rule generalization 0.19–0.24 on `clean`. Weighted to match it reaches 1.00, and a residual fit on those 8 events per rule alone reaches 0.90–0.98. So the world was feasible and the gate's fit was at fault.
- **Decoding.** Predictions learned on `degraded` decoded against clean prototypes read 0.41 and 0.92 on known rules (seeds 299, 298), against 0.89 and 0.97 with `degraded` prototypes on the same clean cues. Prototypes now come from the stream's condition. Otherwise the old-knowledge limit would have measured this mismatch.
- **MLP fit.** Patience 5 stopped at about 30 epochs with known bindings near 0.1. Patience 20 and at most 400 epochs give 1.00 on `clean`, on all four architecture-seed pairs tried.

**Gate readings after those changes** (`reports/surprise-lifecycle-v1-gates.json`), seeds 299 and 298:

| reading | `clean` | `degraded` |
|---|---|---|
| gate 1: offline rule generalization | **1.00, 1.00** | 0.88, 0.86 |
| gate 1: slots once per cause, binding recall | **0.94, 1.00** | 0.66, 0.76 |
| residual fit on 8 events per rule, rule generalization | 0.98, 0.90 | 0.55, 0.54 |
| gate 2: AUROC of surprise, known against first occurrence | **1.00, 1.00** | 0.97, 0.99 |
| frozen old reader, old knowledge | 1.00, 1.00 | **0.68, 0.91** |
| gate 5: noise capture on the frozen reader | 0.00, 0.04 | 0.01, 0.01 |
| gate 3: posterior of a rule given a family new to the stream | 0.10 | 0.10 |

Gates 1, 2, 3 and 5 pass. The posterior is 4 of 40 families, against the 0.11 expected, since binding families are also new to the stream, though at its start. **Open on `degraded`:** the frozen reader starts at 0.68 on seed 299, so the old-knowledge limit of 0.85 may exclude every arm at the primary cell, and a rule from 8 degraded events has a ceiling near 0.55. Whether the primary cell stays `degraded` is decided on the pilot, before confirmation, under the provisional limits rule.

## Changes from the pilot

The [pilot](surprise-lifecycle-v1-pilot.md) (2026-11-01) passes gate 6, keeps `degraded` as primary, and shows `life` learning no rules. Its post hoc diagnosis finds that all three long gaps exceed how long m traces can coexist at the primary cell, so `earns-role` is effectively unreachable there as specified. This version is kept as a failed design and is not frozen; a revision would be a new contract.

## Before freezing

- Full text of MINAS's short-term-memory and validation rules (read so far: abstract only).
- Match `sleep-periodic` to the candidate's total update allowance as well as its bytes, as the other [Sleep note](prior-work.md#sleep-consolidation-into-persistent-parameters) proposes, so a win cannot come from delaying updates alone. The pilot records each arm's reader-update operations; the matching rule must be fixed from those before freezing.
- [SEAL](https://arxiv.org/abs/2506.10943), to see what its downstream measure τ uses, since Sleep's Eq. 5 inherits it and it is the nearest published form of accepting an update only if it helps.
- Prior work on prequential or sequential testing of candidate updates in continual learning, and noisy-label replay.
- Whether the noise-capture and old-knowledge limits are attainable by anything. The streaming budgets were not at first.
- Whether 1,500 configurations per arm is enough; the budget's use is the evidence.

## Prior work

Shared survey: [prior-work.md](prior-work.md#surprise-lifecycle-unresolved-experience-as-evidence).

**Established names.** Holding unexplained examples until a group forms a new pattern: novelty detection in data streams, MINAS **[abstract]**. Deciding whether new evidence belongs to an old cause: latent-cause inference, [Gershman et al. 2017](https://elifesciences.org/articles/23763) **[read: model section]**. Scoring a model only on data it has not yet seen: prequential evaluation (Dawid 1984) **[familiar]**. Accumulating to a threshold: sequential testing (Wald) **[familiar]**, and the fixed-count rule here is a crude form of it. Uniform replay: reservoir sampling (Vitter 1985) **[familiar]**. Selective consolidation: Go-CLS ([Sun et al. 2023](https://www.nature.com/articles/s41593-023-01382-9)) **[read: excerpts]**. Ignoring unlearnable surprise: learning progress (Schmidhuber 1991; Oudeyer's IAC) **[abstract/secondary]**. Sign-bit sketch: locality-sensitive hashing **[familiar]**. Scheduled consolidation and accepting a candidate update only if it improves performance: Sleep, Eq. 5 **[read: §3.2–3.4]**.

**What follows without running [derived].**

- A singleton's chance of surviving to its second occurrence depends on intervening surprises. Under FIFO eviction the **trace reach** is K/(surprise rate), where K is the number of traces. A sketch entry survives Z/(surprise rate), its **fingerprint horizon**. That is how long a fingerprint lasts, not how far apart experiences can consolidate: familiarity only breaks eviction ties, and full traces must still survive to meet confirmation. So `life-sketch`'s effective learning reach is left to measurement, and lies somewhere from its trace reach (54 traces: ≈ 270 at q = 0.2) up to at most its horizon.

| arm at 64 KiB | capacity | horizon, q = 0.2 | horizon, q = 0.05 |
|---|---|---|---|
| `fifo-replay` | 127 events | 127 | 127 |
| `omega` | c events | c | c |
| `life`, S = 4 | 113 surprises, fewer while traces are pinned | ≈ 565 | ≈ 2,260 |
| `surprise-fifo` | 127 surprises | ≈ 635 | ≈ 2,540 |
| `life-sketch`, traces | 54 surprises | ≈ 270 | ≈ 1,080 |
| `life-sketch`, fingerprints | 4,096 entries | ≈ 20,480 | ≈ 81,920 |

 These ignore cause events and false matches, and assume each one-off is surprising.
- `reservoir-replay` keeps a share of traces equal to the share of events. Cause events are 368 of 10g, so at g = 8,192 under 1 of 127 stored events is a cause event. Uniform replay should fail at long gaps by arithmetic rather than by mechanism.
- **Chance residual groups are common.** With 113 traces and 32 outcomes, about 3–4 traces share each outcome at any moment, mostly noise. Residual similarity alone will keep proposing groups, and S is small, so a real shadow can be displaced. κ and cue-only validation are the defence. This is a hard test because of the small outcome table, and the pilot should report how often real rules hold a shadow place.
- A first occurrence cannot be told from `noise-once` from its outcome, and its cue carries only a weak base-rate signal: a never-seen family gives a rule with posterior about 0.11, and otherwise causes and noise share cue form and families. So `rare`, with one later occurrence, cannot pass h_b ≥ 2. Its loss is designed in and measured: retention by support has this cost.
- `slow` with step η moves R about 8η toward a rule's outcome in 8 occurrences, against about √(qL)·η of noise-driven random walk, so its long-gap performance degrades with stream length at a fixed η.
- A residual group can only find rules with a shared outcome, which is why this world has only those.

**Known failure modes and handling.** Our keys are correlated (effective rank 12.7 on 120 keys), so rank-one updates may leak between families: the damage count and the old-knowledge reading measure this. Sign sketches may mostly false-match: `count-sketch` and the match rates measure this. Surprise may be perception failure: `clean` and gate 2. Bindings sharing an outcome form residual groups: cue-only validation on other cues is meant to reject them, and promotion by hidden kind reports it. A large grid on few pilot seeds may overfit: one setting per arm, fresh seeds. Gradient alignment was rejected as the grouping signal, since [per-example coherence gets high even on random labels](https://arxiv.org/abs/2002.10657) **[abstract]**.

**Baselines.** Replay, recent windows, immediate updates, slow learning and the two neural-memory update rules are all included. Prioritized replay is not, since priority by error favours noise here; `count` and `surprise-fifo` cover its two main ingredients **[derived; not a test of it]**.

**What this adds.** A matched-bytes test of whether a sensory evidence process earns its keep over ordinary replay, across gaps, with the first-occurrence limit made explicit. MINAS and its kind work on feature clusters of unlabeled examples to discover classes; Sleep tests a candidate on an isolated copy against a supplied downstream measure τ, and consolidates on a schedule with no selection of what to keep. The additions here are selection by evidence accumulated across the schedule, validation on cue-eligible later sensory events rather than a supplied task, and routing to two destinations. No exact match was found in these few searches, which is not a novelty claim.

## Scaling

K traces, S shadows, Z sketch entries, W MLP weights, c window, D_c = 96, D_o = 32, F slots, n fit events.

**Per event and storage.**

| arm | operations per event | stored | what limits scale |
|---|---|---|---|
| `slow` | D_c·D_o | D_c·D_o | constant; expressive only for what R can hold |
| `surprise-fifo` | K·D_c + D_c·D_o | K(D_c + D_o) | linear in K, reach linear in K |
| `fifo-replay`, `reservoir-replay` | K·D_c + (1 + r)D_c·D_o | K(D_c + D_o) | as above; FIFO reach fixed at K events |
| `titans-m0`, `titans` | ≈ 2W | W or 2W | constant; capacity limits unknown here |
| `omega` | ≈ 2cW | W + c(D_c + D_o) | c multiplies the work |
| `sleep-periodic` | K·D_c, and every P events K·D_c² + D_c³ | K(D_c + D_o) | sleep cost; store limits what one sleep sees |
| `sleep-grow` | as above, plus each expert's r(D_c + D_o) | + r(D_c + D_o) per sleep | grows linearly with stream length |
| `life` | (K + F)·D_c + S(2D_c + 2D_o) + D_c·D_o, on a surprise K·D_o to compare residuals, on reading a group's outcome D_c·D_o per member, and on a promotion K·D_c·D_o | K(D_c + D_o) + S(D_c + D_o) + F(D_c + D_o) | linear in K |
| `life-sketch` | as `life` + 2Z Hamming comparisons | + 8Z bytes | Z linear; fingerprint horizon linear in Z; learning reach to be measured |

**Construction, fitting and search.**

| phase | time | notes |
|---|---|---|
| Encoder fit | fixed, about 70 s per seed on the GTX 1650 | once per seed |
| Rendering and encoding pool | O(symbols × 64 × conditions × encoder) | once per seed and condition |
| Old knowledge, reader | O(n·D_c² + D_c³), n = 3,072 | ridge, direct solve; once per seed and condition |
| Old knowledge, MLP | O(epochs × n × W), n = 3,456, at most 400 epochs | once per architecture, seed and condition; epochs reported |
| Shadow fit | O(m·D_c·D_o) | at most once per proposal |
| Audit | O(cues × renderings × (F·D_c + D_c·D_o)), or × W | plus one state copy |
| Search | configurations × Σ per-gap L × per-event cost | rejected configurations counted |

**Workspace, apart from persistent state.** Key pool, 480 symbols × 64 renderings × 2 conditions × 128 bytes: 7.9 MB per seed, simulating the world. Schedule arrays: about 20 bytes per event, up to 6.6 MB at the longest stream. Audit: one copy of persistent state plus batches. Peak RSS by worker in the runtime files.

- **Facts.** Trace reach in intervening surprises grows linearly with bytes, for any trace-based arm. No bounded store reaches every gap. The sketch lengthens the fingerprint horizon by about 70× per byte (560 against 8), which bounds its learning reach without establishing it **[derived]**. Slots grow with what is kept, and an index would make reads sublinear; neither is run.
- **Read cost.** Measured once before design: about 7 µs for a 640 × 64 store read, single-thread numpy.
- **Probes.** Causes ×4 (g ≥ 512; `rule` stays at 4, its families being limited) with frozen settings, to see whether thresholds survive load. Occurrences n ∈ {4, 8, 32} at g = 512, to measure how many exposures each arm needs, which is where `slow` should catch up.
- **What would not survive scale-up.** 32 outcomes and decoding through an evaluator-held table; a linear reader; a family glyph handing the rule its feature; 512 glyphs; a rule family protected from one-off noise by construction. Real causes will not announce themselves in one 32-dimensional slot. Thresholds on cosines drifted across encoders in streaming retrieval, so expect this here.

## Computation plan

*Estimates, not measurements.*

- **Construction.** Encoders: seeds 299 and 298 exist; 380–389 about 70 s each on the GTX 1650 (`/data/venvs/sri-cuda`). Key pool under a minute of CPU per seed and condition.
- **Pilot.** At most 1,500 configurations × 17 arms × 2 seeds × 5 gaps as a hard ceiling; coordinate descent should use a small part of it. Expected 1–4 CPU-hours on 11 workers, rejected settings counted; measured about 9.6.
- **Confirmation.** 17 arms × 10 seeds × 5 gaps × 2 values of q, mean stream about 87,000 events, roughly 5–30 s per run. About 10–30 minutes wall. Other B, `clean` and probes add about the same.
- **Deterministic and timed output kept apart.** `reports/surprise-lifecycle-v1-{pilot,confirm,analysis}` and separate `-runtime` files, per [computation reporting](computation-reporting.md), with executed runs, events, configurations scored, shadows proposed, and each phase above. `timing` measures one thread, warm-up, 15 repeats, median and spread.
- **Persistence.** Tasks durable (`experiments/durable.py`); rerunning resumes. Pools written atomically under gitignored `runs/`.
- **Dtype.** float32. Raw images are not retained after encoding; pools hold keys.

## Reproduction

```
.venv/bin/python -m experiments.surprise_lifecycle gates
NSY_WORKERS=11 .venv/bin/python -m experiments.surprise_lifecycle pilot
NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.surprise_lifecycle encoders
NSY_WORKERS=11 .venv/bin/python -m experiments.surprise_lifecycle confirm
.venv/bin/python -m experiments.surprise_lifecycle timing
.venv/bin/python -m experiments.surprise_lifecycle analyse
```
`confirm` refuses to run until this file's status says frozen. Tests: `.venv/bin/python -m unittest tests.test_surprise_lifecycle`. Code: `src/surprise_lifecycle.py`, `experiments/surprise_lifecycle.py`. The tests cover each state transition above, including: no trace ever fits two shadows, and a retired shadow cannot return without m previously unused traces; a shadow retires on an unsurprising event; the familiar bit never changes a count; a pinned trace cannot be evicted; deleting a group deletes its counts.
