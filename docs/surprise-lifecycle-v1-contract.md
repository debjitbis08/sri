# Surprise lifecycle v1: contract

Status: **draft, revised 2026-11-01 after external review. Nothing built or run; not frozen.** Values marked *provisional* may change once, on pilot seeds 299 and 298, before confirmation seeds exist. Every such change is recorded under *Changes from the pilot*. The gates and the items under *Before freezing* must be cleared first.

## Question

Does keeping unresolved surprises, and letting later independent experience decide what they are, let a bounded memory learn more than replay or immediate updates at the same bytes, when related experiences are far apart?

The user's [accumulated-surprise proposal](continuous-sensory-memory-notes.md#accumulating-surprises-before-changing-shared-knowledge) is tested at its cheapest. Traces gain support from independent experience, drive an update, and otherwise fade. The test also examines three parts of the proposal separately: the later-experience test, grouping by shared error rather than similar cues, and a cheap familiarity sketch.

**Scope.** No host network, no encoder training, no drifting perception, no sparse reader. "Shared knowledge" here is a linear reader on frozen keys, so a passing result shows that the evidence process works on structure the existing features can express. It does **not** show that a representation should reorganize, or that the sparse-network tradeoff is solved. Those need a later experiment and are only licensed by a pass here.

## Material

- **Sensor.** The frozen key encoder of the [stable key revision](translation-memory-v1-pilot.md#stable-keys-perception-gate-passes) (base network, 32-dimensional unit-norm key, 839,040 bytes, identical in every arm), trained on background symbols only.
- **Symbols.** The task's 512-symbol pool, which no encoder trained on. Per seed, a fixed permutation assigns 16 family glyphs (8 known, 4 new-rule, 4 binding), 32 outcome glyphs, 300 stream items and 100 audit-only items.
- **Event.** A cue and usually an outcome. The cue is two glyphs, encoded separately and concatenated (width 64, cosine = mean of the two key cosines). The outcome is one glyph's key (width 32). Every event is a sensory rendering. No arm sees a word, class, cause identity, or the table of outcome keys.
- **Renderings.** 64 per symbol, encoded once per condition. Audit renderings are disjoint from stream renderings. The pool simulates the world and is not memory of any arm.

## World

Before the stream, every arm is given the same **old knowledge**: 8 known rules (a known family with any item gives that family's outcome) and 24 known bindings (a specific family–item cue gives a specific outcome), by the fitting procedure of its model class, at that class's cost.

**Hidden kinds.** Labels belong to the evaluator and scheduler only.

| kind | count | what happens | occurrences |
|---|---|---|---|
| `rule` | 4 | new family glyph, a different stream item each time, one fixed outcome. Audit uses audit-only items | 8 |
| `binding` | 12 | one fixed family–item cue, one fixed outcome | 8 |
| `exception` | 6 | one fixed cue in a known family, one fixed outcome other than the family's | 8 |
| `rare` | 12 | as `binding` | 2 |
| `noise-repeat` | 8 | one fixed cue, a fresh random outcome each time. Recurs as often as a binding | 8 |
| `noise-burst` | 4 | one fixed cue, identical outcome 6 times within 32 events; a different outcome in each later episode | 4 episodes |
| `noise-once` | background | a cue never used before or again, a uniformly random outcome | rate q |
| `query` | background | a cue with no outcome, uniform over cues already shown | 10% |
| `known` | the rest | old knowledge, correctly predicted | |

**One-off cues.** Sampled per stream without replacement from all ordered pairs (a, b), with a from the 16 families and 300 stream items and b from the stream items, minus every cue reserved above: about 94,800 pairs, against 65,536 needed at g = 32,768, q = 0.2. An item can therefore sit in the first slot, and about 5% of one-off cues begin with a family glyph. A one-off cue shares a glyph with many others, so `cos ≈ 0.6` to them; τ must lie above that, and the construction check reports the share of one-off cues within τ of a cause. Exhausting the set is a build error, not a resample.

**Schedule.** Occurrences of a cause lie g events apart, ±10% uniform jitter, with random phase. The stream runs 8 periods plus a tail of 2g events with no scheduled causes, so its length is L = 10g. Burst episodes are spaced by g.

| factor | values |
|---|---|
| gap g | 128, 512, 2,048, 8,192, 32,768 |
| one-off surprise rate q | **0.2**, 0.05 |
| temporary budget B | 16, **64**, 256 KiB |
| condition | **`degraded`**, `clean` |

`degraded` is noise 0.06 with occluder 4, the [streaming retrieval](streaming-retrieval-v1-contract.md) level (single-frame 120-way accuracy 0.65). `clean` is independent renderings. Bold marks the primary cell. Causes occupy 360 events of L, so 28% of the stream at g = 128 and about 0.1% at g = 32,768.

**Not in this world:** a known rule changing, conjunctions needing a feature the sensor lacks, rules whose outcome varies with the item, cues that change over time, delays with no events. Gap always means intervening events; idle time is untested, since no arm has a clock.

## Common machinery

**Reader.** R, 64 × 32 linear, ridge-fit on old knowledge. Update `R ← R + η·w·(o − Rc)cᵀ/‖c‖²`.

**Answer.** If a slot's cue has cosine at least τ with c, the most similar slot's outcome; otherwise Rc.

**Match.** Everything an arm scores is observable: `hit(p, o)` means `cos(p, o) ≥ 1 − θ` for a prediction p and an observed outcome key o. An event is *surprising* if the answer misses it. θ is a single constant per arm. Only the audit decodes through the evaluator's outcome table.

**Bytes.** Identical outside B in every arm: the encoder, 8,192 bytes for the reader or its equivalent, and 9,216 bytes for the 24 known bindings (384 each). MLP arms get those last two as weights, so every arm has the same total. Inside B:

| item | bytes | notes |
|---|---|---|
| lifecycle trace | 432 | cue and outcome 384, trace counters 16, and one group record 32, reserved so group state can never overflow |
| shadow | 400 | ē, ĉ 384, counters 16 |
| new slot | 384 | |
| sketch entry | 8 | the projection is regenerated from the seed and not stored |
| FIFO or reservoir event | 388 | |
| MLP | 4 per weight, momentum or window float | |

**Allocation.** S shadow places are reserved first. `life-sketch` gives half of B to the sketch first. Traces and new slots share what remains; slots may take at most half of that, and past the cap the least recently used *new* slot goes. A new slot evicts the lowest-priority trace if needed. A configuration is **rejected before selection** if it holds fewer than 4m traces, or fewer slots than 12. Fixed scalars (write pointers, the reservoir count) add under 64 bytes and are reported.

| B, S = 4 | `life` traces | `life-sketch` traces, entries | slot cap | FIFO events |
|---|---|---|---|---|
| 16 KiB | 34 | 15, 1,024 | 19 | 42 |
| **64 KiB** | 148 | 72, 4,096 | 83 | 168 |
| 256 KiB | 603 | 299, 16,384 | 339 | 675 |

The slot cap shown is `life`'s; `life-sketch` has half that or less (8 at 16 KiB, 40 at 64 KiB, 168 at 256 KiB). So `life-sketch` is infeasible at 16 KiB for every setting, and that cell reports `life` only. B = 16 KiB is descriptive, so no verdict depends on it.

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
| `life-oracle` | hidden cause identity replaces grouping and routing; same stores, same fits | upper bound; never a comparator |

**Comparators.** Each uses the full budget.

| arm | update | corresponds to |
|---|---|---|
| `frozen` | none | floor |
| `slow` | R step on every event with an outcome, small η; no store beyond known bindings | slow learning; no traces |
| `surprise-fifo` | R step with w = 1 − cos(ô, o); FIFO store of surprising events answers lookups, newest wins | immediate surprise-weighted update |
| `fifo-replay` | FIFO of all events with outcomes; r replay steps per event; lookups | bounded recent window on the store |
| `reservoir-replay` | as above, uniform sample of all events (Vitter's Algorithm R) | ordinary bounded replay |
| `titans-m0`, `titans` | MLP M, `M_t = (1 − α)M_{t−1} + S_t`, `S_t = βS_{t−1} − θ_M∇‖M(c) − o‖²`; β = 0 in `-m0` | inspired by [Titans](https://arxiv.org/html/2501.00663v1) Eqs. 12–14 |
| `omega` | MLP M, one gradient step per event on the last c events, γ = 1 | inspired by [ATLAS](https://arxiv.org/pdf/2505.23735) Eq. 9 |

MLP width is whatever fills the total budget, counting weights, momentum and window. These three are **inspired by, not tests of,** the published systems. Their α, β, θ_M and γ are constants rather than functions of the input, there is no attention or outer-loop training, and ATLAS's feature mapping and optimizer are left out. A loss here does not count against those papers. Omega's input-dependent γ can prune inside the window, which a constant γ cannot.

### The mechanism

Eligibility is decided from the cue alone, before the outcome is seen. Every eligible event is scored, so a group can never select the outcomes it is judged on. Grouping uses the outcome, but only to propose a candidate, never to validate one.

On each event with an outcome:

1. **Fix eligibility from c.** A same-cue group is eligible if `cos(c, its cue) ≥ τ`. A shadow is eligible if `cos(c, ĉ) ≥ τ_s` and c is not within τ of a trace it was fitted on.
2. **Score every eligible group and shadow, on every eligible event, surprising or not.** Same-cue: *hit* or *miss* for its mean outcome. Shadow, on the reader path alone (slots ignored): *win* if R + Δ hits and R misses, *failure* if R + Δ misses, otherwise neutral. Separately, on every event with an outcome, a shadow *damages* if R hits and R + Δ misses. Queries have no outcome and score nothing.
3. **Stop if the answer hit.**
4. **Keep the evidence.** Append a trace holding cue and outcome, not a scalar.
5. **Group two ways.** Join a *same-cue* group if `cos(c, its cue) ≥ τ`. Join a *residual* group if `cos(e, ē_G) ≥ τ_e`, where `e = o − Rc`: experiences wrong in the same way, whatever they look like. A trace is in at most one of each.
6. **Propose.** A residual group with m traces from distinct episodes, whose unit cues have mean resultant length at least κ, freezes a rank-one shadow `Δ = ē ĉᵀ/‖ĉ‖²`, if none exists for it. κ rejects chance groups, which a rank-one update could not generalize from anyway. If all S places are taken, the shadow with the fewest wins goes, oldest first.
7. **Decide.** An *episode* is at least E events after a group's last counted one. Hits and wins count once per episode. **Misses, failures and damage always count.**

| evidence | action |
|---|---|
| same-cue hits ≥ h_b, misses 0 | write a slot with the group's mean outcome, drop its traces (`binding`, `exception`) |
| same-cue misses ≥ h_n | drop its traces (`noise-repeat`) |
| shadow wins ≥ h_r, failures ≤ h_f − 1, damage ≤ δ of events where R hit | `R ← R + Δ`, drop its traces (`rule`) |
| shadow failures ≥ h_f, or damage above δ once R has hit 64 times since it was fitted | drop the shadow, keep its traces |
| none | trace remains until evicted |

8. **After `R` changes,** recompute each residual group's e from its traces under the new R. Traces R now hits leave the group, a group below m traces loses its shadow, and any remaining shadow is refit with its win, failure and damage counts **reset to zero**. Slot writes do not change R, and shadow scores use only the reader path, so they cause no reset.
9. **Evict** the lowest distinct-episode support, oldest first. Singletons therefore leave first in FIFO order, which is what the reach calculation assumes.

There is no time-based expiry, since a clock would decide the gap by another name. A shadow leaves by failing, by being displaced, or by promotion.

**Sketch.** A ring of 8-byte entries: 32 sign bits of a fixed projection of the cue, 32 of the residual. Every surprise writes one. A new surprise matching an entry within Hamming radius ρ, on either half, enters the store with support 1, so it outlasts singletons.

**Provenance.** Each trace comes from a sensory outcome; `query` events have none. Only `life-selfcount` breaks this rule.

### Provisional settings, in search order

| setting | grid | start |
|---|---|---|
| θ | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 | 0.3 |
| τ | 0.7, 0.8, 0.9, 0.95 | 0.8 |
| τ_e | 0.3, 0.5, 0.7, 0.9 | 0.7 |
| τ_s | 0.4, 0.5, 0.6, 0.7 | 0.5 |
| κ | 0.5, 0.6, 0.7, 0.8 | 0.6 |
| m | 2, 3, 4 | 3 |
| h_b, h_r | 1, 2, 3 | 2 |
| h_f | 1, 2, 3 | 2 |
| h_n | 2, 3 | 2 |
| S | 2, 4, 8 | 4 |
| δ | 0.02, 0.05 | 0.02 |
| E | 16, 64 | 64 |
| ρ (sketch) | 4, 8, 12 | 8 |
| τ_g (`life-cuegroup`) | 0.5, 0.6, 0.7 | 0.6 |
| η | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.03 |
| r | 1, 4 | 1 |
| MLP | layers 1, 2; α 0, 0.0001, 0.001; β 0.5, 0.9; θ_M 0.001, 0.003, 0.01, 0.03; c 16, 64, 256 | 2, 0.0001, 0.9, 0.01, 64 |

## Readings

An audit reads a copy of the state, on audit renderings, never changing the arm. It runs at quarter points and at the end; the primary value is at the end, after the tail. Answers are decoded by nearest outcome key among 32 (chance 1/32). Primary audits use `clean` renderings, even in a `degraded` stream, so they measure what was learned rather than one noisy read. `degraded` audits are also reported.

**Primary.**

- **Rule generalization.** `rule` families with audit-only items.
- **Binding recall.** `binding` and `exception` cues, macro-averaged over cues.
- **Noise capture.** The same behavioural reading in every arm, whichever part holds the memory, including anything absorbed into R. Target: per cue, the **last outcome** that cue was shown with. Cues: all `noise-repeat`, all `noise-burst`, and up to 128 `noise-once` cues drawn uniformly from the first 80% of the stream, the same cues for every arm of a seed and cell. Each is audited on 8 renderings. Per kind: `P(answer = target) − P(answer = target of a different cue of that kind)`, the second term being the empirical null, over a fixed derangement. Reading: the mean of the three kinds. This measures whether the arm repeats old noise as knowledge. It cannot separate a memory held as episodic from one taken as true, since no arm abstains.
- **Old knowledge.** Macro-average of known rules on audit-only items, known bindings, and each `exception` family on audit-only items.

**Descriptive.** Slots promoted per hidden kind (the arm's own routing, reported separately from noise capture), `rare` recall, occurrences to first correct, slots and traces held, shadows proposed and promoted, sketch true and false match rates against hidden identity, curves over the quarter points.

**Compute.** Operations per event (derived per arm): multiplications plus one for each 64-bit Hamming comparison. Measured µs per event on one thread, every byte by category.

## Selection and decision rule

**Search.** Pilot seeds 299 and 298, primary cell, all five gaps, coordinate descent. Lifecycle settings are searched once on `life`, in the table's order, starting from the table's values; `count`, `count-sketch` and `life-selfcount` inherit them, adding only what they alone use (h_b and h_r as support thresholds for `count`). Then ρ on `life-sketch`, τ_g on `life-cuegroup`. Each comparator is searched on its own settings. A pass visits each setting once, holding the rest. Search ends after a pass with no change, or after 3 passes, or at 1,500 scored configurations per arm, whichever is first, and the budget's use is recorded.

**Objective.** Among configurations meeting the provisional limits, noise capture ≤ 0.10 and old knowledge ≥ 0.85: the highest mean over gaps of (rule generalization + binding recall)/2, then fewer operations. If none meets them, the smallest largest relative excess over a limit, then the objective; the arm is marked and still enters the rule.

**Candidate.** Of `life` and `life-sketch`, the higher pilot objective restricted to long gaps. Only that one enters the outcomes. The other is reported through the sketch contrast.

**Comparable cost.** At most 6× the mean operations per event of `reservoir-replay` at r = 1 *(provisional)*.

**Confirmation.** Seeds 380–389, fresh encoders and permutations, settings frozen. Differences paired by seed; percentile bootstrap over seeds with 10,000 resamples and a fixed generator; 95% intervals, two-sided, bounds at 2.5% and 97.5%.

The candidate **beats** a comparator on a reading at a gap if the mean gain is at least 0.10 and the lower bound exceeds 0, and the guards hold: lower bound of the difference ≥ −0.03 on the other of rule generalization and binding recall, ≥ −0.02 on old knowledge, and upper bound ≤ +0.02 on noise capture.

The **long gaps** are 2,048, 8,192 and 32,768, each beyond the reach of every full-store arm at the primary cell (see below). Comparators are `slow`, `surprise-fifo`, `fifo-replay`, `reservoir-replay`, `titans-m0`, `titans` and `omega`. A reading is **won** if the candidate beats **all seven at the same gap**, at **at least two** long gaps. Requiring all comparators is conservative and needs no adjustment across them **[derived: intersection-union]**; the choice of two of three gaps is not adjusted.

| outcome, first that applies | condition | meaning |
|---|---|---|
| `uninformative` | `life-oracle` fails to beat the best comparator, by this rule, on both rule generalization and binding recall | even perfect sorting would not help in this world, so the test cannot speak to the mechanism |
| `earns-role` | rule generalization won, at comparable cost | unresolved surprises are a route to shared updates here; the sparse-network experiment is licensed |
| `binding-store-only` | binding recall won, rule generalization not, at comparable cost | traces work as a binding store; no route to shared updates was established |
| `higher-cost-only` | either won, above comparable cost | report the frontier |
| `no-advantage-shown` | neither | the lifecycle has no claim over ordinary means on this evidence. This is a failed superiority test, not proof that replay suffices; the intervals are reported |

**Secondary contrasts**, same bootstrap, margin 0.05 on rule generalization and binding recall, 0.02 on noise capture and old knowledge. Each reads three ways at each long gap: **better** (mean gain at least the margin, interval excludes 0), **equivalent** (interval within ±margin), otherwise **inconclusive**.

- **The later-experience test:** `life` against `count`, on noise capture and old knowledge.
- **Grouping by error:** `life` against `life-cuegroup`, on rule generalization and noise capture.
- **Sketch:** `life-sketch` against `life`, and `count-sketch` against `count` on noise capture.
- **Self-support:** `life-selfcount` against `life`, on noise capture.

Everything else (other q, B, `clean`, short gaps, `rare`) is descriptive.

**Expectations recorded before any run:** `life` is better than `count` on noise capture. `life-cuegroup` may match `life`, because the family glyph makes a rule visible in the cue, so this world does not favour grouping by error; equivalence here would not rule it out for structure the cue hides. 8 small steps leave `slow` weak at long gaps. 32,768 defeats all but possibly the sketch. `binding-store-only` is at least as likely as `earns-role`.

## Gates

Each on pilot seeds, before selection. Failure stops the study until the design changes, recorded under *Changes from the pilot*.

1. **World.** R fit offline on all `rule` events reaches rule generalization ≥ 0.85, and slots written once per cause reach binding recall ≥ 0.85, on `clean`. This shows a win is possible and sets the ceiling. `degraded` values are recorded.
2. **Separability of surprise.** With R and slots as fitted on old knowledge, AUROC of `1 − cos(ô, o)` between `known` events and first occurrences of causes must be ≥ 0.90 on `clean`. At `degraded`, if below 0.60, the primary cell cannot test the mechanism, and `clean` becomes primary before confirmation.
3. **Construction.** No one-off cue repeats or matches a reserved cue. Every configuration in the search fits its budget, by asserted byte counts.
4. **No privileged input.** Arm code receives only keys, in stream order. Tests check that a stream's outcome table, kind labels and cause ids cannot reach arm code, `life-oracle` excepted.
5. **Audit.** Bitwise-identical state before and after every audit. Shuffled outcome keys score chance; the noise-capture null is near 0 on `frozen`.

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

Added while revising: the κ requirement and `uninformative` outcome. Chance residual groups turned out to be common by arithmetic (see below), and a world where the oracle cannot win would otherwise read as a failure of the mechanism.

## Changes from the pilot

None yet.

## Before freezing

- Full text of MINAS's short-term-memory and validation rules (read so far: abstract only).
- Prior work on prequential or sequential testing of candidate updates in continual learning, and noisy-label replay.
- Whether the noise-capture and old-knowledge limits are attainable by anything. The streaming budgets were not at first.
- Whether 1,500 configurations per arm is enough; the budget's use is the evidence.

## Prior work

Shared survey: [prior-work.md](prior-work.md#surprise-lifecycle-unresolved-experience-as-evidence).

**Established names.** Holding unexplained examples until a group forms a new pattern: novelty detection in data streams, MINAS **[abstract]**. Deciding whether new evidence belongs to an old cause: latent-cause inference, [Gershman et al. 2017](https://elifesciences.org/articles/23763) **[read: model section]**. Scoring a model only on data it has not yet seen: prequential evaluation (Dawid 1984) **[familiar]**. Accumulating to a threshold: sequential testing (Wald) **[familiar]**, and the fixed-count rule here is a crude form of it. Uniform replay: reservoir sampling (Vitter 1985) **[familiar]**. Selective consolidation: Go-CLS ([Sun et al. 2023](https://www.nature.com/articles/s41593-023-01382-9)) **[read: excerpts]**. Ignoring unlearnable surprise: learning progress (Schmidhuber 1991; Oudeyer's IAC) **[abstract/secondary]**. Sign-bit sketch: locality-sensitive hashing **[familiar]**.

**What follows without running [derived].**

- A singleton's chance of surviving to its second occurrence depends on intervening surprises. Under FIFO eviction the reach is K/(surprise rate), where K is the number of traces.

| arm at 64 KiB | capacity | reach, q = 0.2 | reach, q = 0.05 |
|---|---|---|---|
| `fifo-replay` | 168 events | 168 | 168 |
| `omega` | c events | c | c |
| `life`, S = 4 | 148 surprises | ≈ 740 | ≈ 2,960 |
| `surprise-fifo` | 168 surprises | ≈ 840 | ≈ 3,360 |
| `life-sketch` | 4,096 sketch entries | ≈ 20,480 | ≈ 81,920 |

 These ignore cause events and false matches, and assume each one-off is surprising.
- `reservoir-replay` keeps a share of traces equal to the share of events. Cause events are 360 of 10g, so at g = 8,192 under 1 of 168 stored events is a cause event. Uniform replay should fail at long gaps by arithmetic rather than by mechanism.
- **Chance residual groups are common.** With 148 traces and 32 outcomes, about 4–5 traces share each outcome at any moment, mostly noise. Residual similarity alone will keep proposing groups, and S is small, so a real shadow can be displaced. κ and cue-only validation are the defence. This is a hard test because of the small outcome table, and the pilot should report how often real rules hold a shadow place.
- A first occurrence cannot be told from `noise-once` by any online method. So `rare`, with one later occurrence, cannot pass h_b ≥ 2. Its loss is designed in and measured: retention by support has this cost.
- `slow` with step η moves R about 8η toward a rule's outcome in 8 occurrences, against about √(qL)·η of noise-driven random walk, so its long-gap performance degrades with stream length at a fixed η.
- A residual group can only find rules with a shared outcome, which is why this world has only those.

**Known failure modes and handling.** Our keys are correlated (effective rank 12.7 on 120 keys), so rank-one updates may leak between families: the damage count and the old-knowledge reading measure this. Sign sketches may mostly false-match: `count-sketch` and the match rates measure this. Surprise may be perception failure: `clean` and gate 2. Bindings sharing an outcome form residual groups: cue-only validation on other cues is meant to reject them, and promotion by hidden kind reports it. A large grid on few pilot seeds may overfit: one setting per arm, fresh seeds. Gradient alignment was rejected as the grouping signal, since [per-example coherence gets high even on random labels](https://arxiv.org/abs/2002.10657) **[abstract]**.

**Baselines.** Replay, recent windows, immediate updates, slow learning and the two neural-memory update rules are all included. Prioritized replay is not, since priority by error favours noise here; `count` and `surprise-fifo` cover its two main ingredients **[derived; not a test of it]**.

**What this adds.** A matched-bytes test of whether a sensory evidence process earns its keep over ordinary replay, across gaps, with the first-occurrence limit made explicit. MINAS and its kind work on feature clusters of unlabeled examples to discover classes; the additions here are shadow updates validated on cue-eligible later events, and routing to two destinations. No match was found in these few searches, which is not a novelty claim.

## Scaling

K traces, S shadows, Z sketch entries, W MLP weights, c window, D_c = 64, D_o = 32, F slots, n fit events.

**Per event and storage.**

| arm | operations per event | stored | what limits scale |
|---|---|---|---|
| `slow` | D_c·D_o | D_c·D_o | constant; expressive only for what R can hold |
| `surprise-fifo` | K·D_c + D_c·D_o | K(D_c + D_o) | linear in K, reach linear in K |
| `fifo-replay`, `reservoir-replay` | K·D_c + (1 + r)D_c·D_o | K(D_c + D_o) | as above; FIFO reach fixed at K events |
| `titans-m0`, `titans` | ≈ 2W | W or 2W | constant; capacity limits unknown here |
| `omega` | ≈ 2cW | W + c(D_c + D_o) | c multiplies the work |
| `life` | (K + F)·D_c + K·D_o + S(2D_c + 2D_o) + D_c·D_o, and on a promotion K·D_c·D_o | K(D_c + D_o) + S(D_c + D_o) + F(D_c + D_o) | linear in K |
| `life-sketch` | as `life` + 2Z Hamming comparisons | + 8Z bytes | Z linear; reach linear in Z |

**Construction, fitting and search.**

| phase | time | notes |
|---|---|---|
| Encoder fit | fixed, about 70 s per seed on the GTX 1650 | once per seed |
| Rendering and encoding pool | O(symbols × 64 × conditions × encoder) | once per seed and condition |
| Old knowledge, reader | O(n·D_c² + D_c³) | ridge, direct solve |
| Old knowledge, MLP | O(epochs × n × W) | counted to a common fit target |
| Shadow fit | O(m·D_c·D_o) | at most once per proposal |
| Audit | O(cues × renderings × (F·D_c + D_c·D_o)), or × W | plus one state copy |
| Search | configurations × Σ per-gap L × per-event cost | rejected configurations counted |

**Workspace, apart from persistent state.** Key pool, 448 symbols × 64 renderings × 2 conditions × 128 bytes: 7.3 MB per seed, simulating the world. Schedule arrays: about 20 bytes per event, up to 6.6 MB at the longest stream. Audit: one copy of persistent state plus batches. Peak RSS by worker in the runtime files.

- **Facts.** Reach in intervening surprises grows linearly with bytes, for any trace-based arm. No bounded store reaches every gap, and the sketch only changes the constant, by about 50× in bytes per entry (432 against 8) **[derived]**. Slots grow with what is kept, and an index would make reads sublinear; neither is run.
- **Read cost.** Measured once before design: about 7 µs for a 640 × 64 store read, single-thread numpy.
- **Probes.** Causes ×4 (g ≥ 512) with frozen settings, to see whether thresholds survive load. Occurrences n ∈ {4, 8, 32} at g = 512, to measure how many exposures each arm needs, which is where `slow` should catch up.
- **What would not survive scale-up.** 32 outcomes and decoding through an evaluator-held table; a linear reader; a family glyph handing the rule its feature; 512 glyphs. Real causes will not announce themselves in one 32-dimensional slot. Thresholds on cosines drifted across encoders in streaming retrieval, so expect this here.

## Computation plan

*Estimates, not measurements.*

- **Construction.** Encoders: seeds 299 and 298 exist; 380–389 about 70 s each on the GTX 1650 (`/data/venvs/sri-cuda`). Key pool under a minute of CPU per seed and condition.
- **Pilot.** At most 1,500 configurations × 15 arms × 2 seeds × 5 gaps as a hard ceiling; coordinate descent should use a small part of it. Expect 1–4 CPU-hours on 11 workers, rejected settings counted.
- **Confirmation.** 15 arms × 10 seeds × 5 gaps × 2 values of q, mean stream about 87,000 events, roughly 5–30 s per run. About 10–30 minutes wall. Other B, `clean` and probes add about the same.
- **Deterministic and timed output kept apart.** `reports/surprise-lifecycle-v1-{pilot,confirm,analysis}` and separate `-runtime` files, per [computation reporting](computation-reporting.md), with executed runs, events, configurations scored, shadows proposed, and each phase above. `timing` measures one thread, warm-up, 15 repeats, median and spread.
- **Persistence.** Tasks durable (`experiments/durable.py`); rerunning resumes. Pools written atomically under gitignored `runs/`.
- **Dtype.** float32. Raw images are not retained after encoding; pools hold keys.

## Reproduction

To be written with the implementation: `src/surprise_lifecycle.py`, `experiments/surprise_lifecycle.py`, `tests/test_surprise_lifecycle.py`.
