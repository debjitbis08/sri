# Surprise lifecycle v1: contract

Status: **draft, 2026-11-01. Nothing built or run; not frozen.** Values marked *provisional* may change once, on pilot seeds 299 and 298, before confirmation seeds exist. Every such change is recorded under *Changes from the pilot*. The world gate and the items under *Before freezing* must be cleared first.

## Question

Does keeping unresolved surprises, and letting later independent experience decide what they are, let a bounded memory learn more than replay or immediate updates at the same bytes, when related experiences are far apart?

The user's [accumulated-surprise proposal](continuous-sensory-memory-notes.md#accumulating-surprises-before-changing-shared-knowledge) is tested at its cheapest. Traces gain support from independent experience, drive an update, and otherwise fade. The test also decides three parts of the proposal separately: whether the later-experience test is needed, whether grouping by shared error finds structure that similar cues do not, and how far a cheap familiarity sketch extends the reachable gap.

**Scope.** No host network, no encoder training, no drifting perception, no sparse reader. "Shared knowledge" here is a linear reader on frozen keys, so a passing result shows that the evidence process works on structure the existing features can express. It does **not** show that a representation should reorganize, or that the sparse-network tradeoff is solved. Those need a later experiment and are only licensed by a pass here.

## Material

- **Sensor.** The frozen key encoder of the [stable key revision](translation-memory-v1-pilot.md#stable-keys-perception-gate-passes) (base network, 32-dimensional unit-norm key, 839,040 bytes, identical in every arm), trained on background symbols only.
- **Symbols.** The task's 512-symbol pool, which no encoder trained on. Per seed, a fixed permutation assigns 16 family glyphs (8 known, 4 new-rule, 4 binding), 32 outcome glyphs, 300 stream items and 100 audit-only items.
- **Event.** A cue and usually an outcome. The cue is two glyphs, a family and an item, encoded separately and concatenated (width 64, cosine = mean of the two key cosines). The outcome is one glyph's key (width 32). Every event is a sensory rendering. No arm sees a word, class or cause identity.
- **Renderings.** 64 per symbol, encoded once per condition. Audit renderings are disjoint from stream renderings.

## World

Before the stream, every arm is given the same **old knowledge**: 8 known rules (a known family with any item gives that family's outcome) and 24 known bindings (a specific cue gives a specific outcome), by the fitting procedure of its model class, at that class's cost.

**Hidden kinds.** Labels belong to the evaluator and scheduler only.

| kind | count | what happens | occurrences |
|---|---|---|---|
| `rule` | 4 | new family glyph, a different stream item each time, one fixed outcome. Audit uses unseen items | 8 |
| `binding` | 12 | one fixed cue, one fixed outcome | 8 |
| `exception` | 6 | one fixed cue in a known family, one fixed outcome other than the family's | 8 |
| `rare` | 12 | as `binding` | 2 |
| `noise-repeat` | 8 | one fixed cue, a fresh random outcome each time. Recurs as often as a binding | 8 |
| `noise-burst` | 4 | one fixed cue, identical outcome 6 times within 32 events; a different outcome in each later episode | 4 episodes |
| `noise-once` | background | random family and item, random outcome, never scheduled again | rate q |
| `query` | background | a cue with no outcome, drawn from all cues above | 10% |
| `known` | the rest | old knowledge, correctly predicted | |

**Schedule.** Occurrences of a cause lie g events apart, ±10% uniform jitter, with random phase. The stream runs 8 periods plus a tail of 2g background events, so its length is 10g. Bursts are one episode each and spaced by g.

| factor | values |
|---|---|
| gap g | 128, 512, 2,048, 8,192, 32,768 |
| one-off surprise rate q | **0.2**, 0.05 |
| temporary budget B | 16, **64**, 256 KiB |
| condition | **`degraded`**, `clean` |

`degraded` is noise 0.06 with occluder 4, the [streaming retrieval](streaming-retrieval-v1-contract.md) level (single-frame 120-way accuracy 0.65). `clean` is independent renderings. Bold marks the primary cell.

**Not in this world:** a known rule changing, conjunctions needing a feature the sensor lacks, cues that change over time, delays with no events. Gap here always means intervening events; elapsed idle time is untested by construction, since no arm has a clock.

## Common machinery

**Reader.** R, 64 × 32 linear, 8,192 bytes, ridge-fit on old knowledge. Update `R ← R + η·w·(o − Rc)cᵀ/‖c‖²`.

**Answer.** If a stored slot's cue has cosine at least τ with c, its outcome; otherwise Rc. Audits decode by nearest outcome key; the evaluator holds those keys and no arm sees them. Chance is 1/32.

**Surprise.** `s = 1 − cos(ô, o)`, surprising if `s ≥ θ`. The residual is `e = o − ô`.

**Bytes.** B covers every temporary and durable stored item: traces (cue, outcome and 16 bytes of counters: 400 bytes), slots, the 24 known bindings, shadows, and sketch. The encoder and R sit outside B in every arm. At 64 KiB, 163 such items fit; the 24 known bindings take 24, leaving 139 for traces. Unused budget is reported, not reassigned.

**Settings.** One setting per arm is frozen across all gaps, q and B, since a real system is not told the gap.

## Arms

**Lifecycle family.** Same trace store and eviction.

| arm | difference | role |
|---|---|---|
| `life` | the mechanism below | **candidate** |
| `life-sketch` | `life` plus a familiarity sketch, half of B | **candidate** |
| `life-samecue` | no residual groups; R frozen | is grouping by error needed? |
| `count` | promotes on support count alone, no later-experience test | is the test needed? |
| `count-sketch` | `count` with the sketch | does the sketch only add false matches? |
| `life-selfcount` | query events count as a confirming hit | does retrieval manufacture support? |
| `life-oracle` | hidden cause identity replaces grouping and sorting | upper bound; never a comparator |

**Comparators.** Each uses the full budget.

| arm | update | corresponds to |
|---|---|---|
| `frozen` | none | floor |
| `slow` | R step on every event with an outcome, small η; no store beyond known bindings | slow learning; no traces |
| `surprise-fifo` | R step with w = s; FIFO store of surprising events answers lookups, newest wins | immediate surprise-weighted update |
| `fifo-replay` | FIFO of all events with outcomes; r replay steps per event; lookups | bounded recent window on the store |
| `reservoir-replay` | as above, uniform sample of all events (Vitter's Algorithm R) | ordinary bounded replay |
| `titans-m0`, `titans` | MLP M, `M_t = (1 − α)M_{t−1} + S_t`, `S_t = βS_{t−1} − θ∇‖M(c) − o‖²`; β = 0 in `-m0` | inspired by [Titans](https://arxiv.org/html/2501.00663v1) Eqs. 12–14 |
| `omega` | MLP M, one gradient step per event on the last c events, γ = 1 | inspired by [ATLAS](https://arxiv.org/pdf/2505.23735) Eq. 9 |

MLP width is whatever fills B + 8,192 bytes, counting weights, momentum and window. These three are **inspired by, not tests of,** the published systems. Their α, β, θ and γ are constants rather than functions of the input, there is no attention or outer-loop training, and ATLAS's feature mapping and optimizer are left out. A loss here does not count against those papers. Omega's input-dependent γ can prune inside the window, which a constant γ cannot.

### The mechanism

On each event with an outcome:

1. **Score first.** Before anything learns from this event, every live group that matches it records whether it predicted the outcome correctly. This is prequential: a group is only ever scored on events that did not fit it.
2. **Stop if unsurprising,** after recording shadow damage (step 5) on a known event.
3. **Keep the evidence.** Append a trace holding cue and outcome, not a scalar.
4. **Group two ways.** A *same-cue* group if cue cosine ≥ τ. Otherwise a *residual* group if `cos(e, ē_G) ≥ τ_e`: experiences wrong in the same way, whatever they look like. This is the proposed answer to the grouping problem.
5. **Propose.** A residual group with m traces from distinct episodes freezes a rank-one shadow update `Δ = ē ĉᵀ` (96 floats). At most S shadows live at once. A shadow *wins* when it decodes correctly where R does not, on a cue outside its fit. It *damages* when it makes a known event wrong.
6. **Decide.** Thresholds are on prequential scores, counted over distinct episodes, where an episode needs at least E events since the group's last counted one.

| evidence | action |
|---|---|
| same-cue hits ≥ h_b, no misses | write a slot, drop traces (`binding`, `exception`) |
| same-cue misses ≥ h_n | drop traces (`noise-repeat`) |
| shadow wins ≥ h_r, damage rate ≤ δ | `R ← R + Δ`, drop traces (`rule`) |
| none | trace remains until evicted |

7. **Evict** the lowest distinct-episode support, oldest first. Singletons therefore leave first in FIFO order, which is what the reach calculation assumes. Slots may take at most half of B.

**Sketch.** A ring of 8-byte entries: 32 sign bits of a fixed projection of the cue, 32 of the residual. Every surprise writes one. A new surprise that matches an entry within Hamming radius ρ enters the store with support 1, so it outlasts singletons. With half of B, `life-sketch` holds 57 traces beside the known bindings and 4,096 entries at 64 KiB.

**Provenance.** Each trace comes from a sensory outcome; `query` events have none. Only `life-selfcount` breaks this rule.

### Provisional settings

| setting | grid |
|---|---|
| η | 0.003, 0.01, 0.03, 0.1, 0.3 |
| r | 1, 4 |
| θ | 0.1–0.9 by quantile of s on known events in the pilot |
| τ, τ_e | 0.5–0.95, steps of 0.05 |
| m | 2, 3, 4 |
| h_b, h_r | 1, 2, 3 |
| h_n | 2 |
| S | 2, 4, 8 |
| δ | 0.02, 0.05 |
| E | 16, 64 |
| ρ | 4, 8, 12 |
| MLP | layers 1 or 2; α 0, 0.0001, 0.001; β 0.5, 0.9; θ 0.001–0.1, four values; c 16, 64, 256 |

## Readings

An audit reads a copy of the state, on audit renderings, never changing the arm. It runs at quarter points and at the end; the primary value is at the end, after the tail. Primary audits use `clean` renderings, even in a `degraded` stream, so they measure what was learned rather than one noisy read. `degraded` audits are also reported.

**Primary.**

- **Rule generalization.** `rule` families with audit-only items.
- **Binding recall.** `binding` and `exception` cues.
- **False consolidation.** Share of noise cues (`noise-repeat`, `noise-burst`, a fixed sample of `noise-once`) answered from a slot, or in reader-only and MLP arms, answered with a noise event's outcome above chance.
- **Old knowledge.** Known rules on audit-only items, known bindings, and for `exception` the same family on other items.

**Descriptive.** `rare` recall, occurrences to first correct, kinds promoted against hidden kinds, slots and traces held, sketch true and false match rates against hidden identity, curves over the quarter points.

**Compute.** Multiplications per event (derived, per arm), measured µs per event on one thread, every byte by category.

## Selection and decision rule

**Selection.** Pilot seeds 299 and 298, primary cell, one setting per arm: among settings meeting the provisional limits, false consolidation ≤ 0.10 and old knowledge ≥ 0.85, the highest mean over gaps of (rule generalization + binding recall)/2; ties to fewer multiplications. An arm with none in limit reports its closest and is marked. `frozen` has no settings. `count`, `count-sketch`, `life-samecue` and `life-selfcount` use `life`'s shared settings where they apply, so ablations differ only by the ablated part.

**Comparable cost.** A candidate is comparable if its mean multiplications per event are at most 6× those of `reservoir-replay` at r = 1 *(provisional)*.

**Confirmation.** Seeds 380–389, fresh encoders and permutations, settings frozen. Paired by seed, bootstrap over seeds, 10,000 resamples.

A candidate **beats** a comparator on a reading at a gap if:

1. the mean gain is at least 0.10 and the 95% interval excludes 0;
2. on the other primary readings the bound is no worse than: rule generalization and binding recall −0.03, old knowledge −0.02, false consolidation +0.02.

The **long gaps** are 2,048, 8,192 and 32,768: every one exceeds the full-store reach at the primary cell (see below). Outcomes are decided at the primary cell, against `slow`, `surprise-fifo`, `fifo-replay`, `reservoir-replay`, `titans-m0`, `titans` and `omega`, for the better of `life` and `life-sketch`, chosen on the pilot.

| outcome | condition | meaning |
|---|---|---|
| `earns-role` | beats every comparator on rule generalization at ≥ 2 long gaps, at comparable cost | unresolved surprises are a route to shared updates; the sparse-network experiment is licensed |
| `binding-store-only` | beats every comparator on binding recall at ≥ 2 long gaps, but not `earns-role` | traces are a binding store, not a route to reorganization. The proposal's central role is not supported here |
| `higher-cost-only` | either condition met only above comparable cost | report the frontier |
| `replay-suffices` | neither | no case for the lifecycle over ordinary means, on this evidence |

**Secondary contrasts**, same bootstrap, same margins, each with its own one-line result:

- **The later-experience test:** `life` against `count`, false consolidation and old knowledge. No gain means support counting suffices.
- **Grouping by error:** `life` against `life-samecue`, rule generalization. No gain means similar cues suffice or neither works.
- **Sketch:** `life-sketch` against `life` at long gaps, and `count-sketch` against `count` on false consolidation.
- **Self-support:** `life-selfcount` against `life` on false consolidation, with an expected harm.

Everything else (other q, B, `clean`, short gaps, `rare`) is descriptive.

**Expectations recorded before any run:** `life` beats `count` on false consolidation; `rule` needs residual grouping; `slow` holds up at 8 occurrences less well than complementary-systems reasoning would suggest, because 8 small steps are few; 32,768 defeats all but possibly the sketch; `binding-store-only` is at least as likely as `earns-role`.

## Gates

Each on pilot seeds, before selection. Failure stops the study until the design changes.

1. **World.** R fit offline on all `rule` events reaches rule generalization ≥ 0.85, and slots written once per cause reach binding recall ≥ 0.85, on `clean`. This shows a win is possible and sets the ceiling. `degraded` values are recorded.
2. **Separability of surprise.** With oracle features, known events and each hidden kind differ on s at `clean`. Report the AUROC at `degraded`; if it is near chance there, the primary cell cannot test the mechanism and `clean` becomes primary, recorded before confirmation.
3. **Audit.** Bitwise-identical state before and after every audit. Shuffled outcome keys score chance.

## Changes from the pilot

None yet.

## Before freezing

- Full text of MINAS's short-term-memory and validation rules (read so far: abstract only).
- Prior work on prequential or sequential testing of candidate updates in continual learning, and noisy-label replay.
- Whether the old-knowledge and false-consolidation limits are attainable by anything. The streaming budgets were not at first.

## Prior work

Shared survey: [prior-work.md](prior-work.md#surprise-lifecycle-unresolved-experience-as-evidence).

**Established names.** Holding unexplained examples until a group forms a new pattern: novelty detection in data streams, MINAS **[abstract]**. Deciding whether new evidence belongs to an old cause: latent-cause inference, [Gershman et al. 2017](https://elifesciences.org/articles/23763) **[read: model section]**. Scoring a model only on data it has not yet seen: prequential evaluation (Dawid 1984) **[familiar]**. Accumulating to a threshold: sequential testing (Wald) **[familiar]**, and the fixed-count rule here is a crude form of it. Uniform replay: reservoir sampling (Vitter 1985) **[familiar]**. Selective consolidation: Go-CLS ([Sun et al. 2023](https://www.nature.com/articles/s41593-023-01382-9)) **[read: excerpts]**. Ignoring unlearnable surprise: learning progress (Schmidhuber 1991; Oudeyer's IAC) **[abstract/secondary]**. Sign-bit sketch: locality-sensitive hashing **[familiar]**.

**What follows without running [derived].**

- A singleton's chance of surviving to its second occurrence depends on intervening surprises. Under FIFO eviction the reach is K/(surprise rate), where K is the number of traces.

| arm at 64 KiB | capacity | reach, q = 0.2 | reach, q = 0.05 |
|---|---|---|---|
| `fifo-replay` | 139 events | 139 | 139 |
| `omega` | c events | c | c |
| `surprise-fifo`, `life` | 139 surprises | ≈ 695 | ≈ 2,780 |
| `life-sketch` | 4,096 sketch entries | ≈ 20,480 | ≈ 81,920 |

 These ignore cause events (46/g of the stream, not small at g = 128) and false matches.
- `reservoir-replay` keeps a share of traces equal to the share of events. Cause events are 360 of 10g, so at g = 8,192 under 1 of 139 traces is a cause event. Uniform replay should fail at long gaps by arithmetic rather than by mechanism.
- A first occurrence cannot be told from `noise-once` by any online method. So `rare`, with one prequential chance, cannot pass h_b ≥ 2. Its loss is designed in and measured: retention by support has this cost.
- `slow` with step η moves R about 8η toward a rule's outcome in 8 occurrences, against √(qL)·η of noise-driven random walk, so its long-gap performance degrades with stream length at a fixed η.
- A residual group can only find rules with a shared outcome. A rule whose outcome varies with the item leaves no shared residual, so a pass here does not cover it.

**Known failure modes and handling.** Our keys are correlated (effective rank 12.7 on these keys), so rank-one updates may leak between families: the shadow's damage test and the old-knowledge reading measure this. Sign sketches may mostly false-match: `count-sketch` and the match rates measure this. Surprise may be perception failure: `clean` and gate 2. Bindings that share an outcome will falsely form residual groups: the different-cue wins requirement is meant to reject them, and the kind confusion table reports it. A large grid on few pilot seeds may overfit: one setting per arm, fresh seeds. Gradient alignment was rejected as the grouping signal, since [per-example coherence gets high even on random labels](https://arxiv.org/abs/2002.10657) **[abstract]**.

**Baselines.** Replay, recent windows, immediate updates, slow learning and the two neural-memory update rules are all included. Prioritized replay is not, since priority by error favours noise here; `count` and `surprise-fifo` cover its two main ingredients **[derived; not a test of it]**.

**What this adds.** A matched-bytes test of whether a sensory evidence process earns its keep over ordinary replay, across gaps, with the first-occurrence limit made explicit. MINAS and its kind work on feature clusters of unlabeled examples to discover classes; the additions here are residual grouping, shadow updates scored prequentially, and routing to two destinations. No match was found in these few searches, which is not a novelty claim.

## Scaling

Per event: K traces, cue width D_c = 64, outcome width D_o = 32, S shadows, Z sketch entries, W MLP weights, c window.

| arm | work per event | stored | what limits scale |
|---|---|---|---|
| `slow` | D_c·D_o | D_c·D_o | constant; expressive only for what R can hold |
| `surprise-fifo` | K·D_c + D_c·D_o | K(D_c + D_o) | linear in K, reach linear in K |
| `fifo-replay`, `reservoir-replay` | K·D_c + (1 + r)D_c·D_o | K(D_c + D_o) | as above; reach of the FIFO fixed at K events |
| `titans-m0`, `titans` | ≈ 2W | W or 2W | constant; capacity limits unknown here |
| `omega` | ≈ 2cW | W + c(D_c + D_o) | c multiplies the work |
| `life` | K·D_c + K·D_o + S(D_c + D_o) + D_c·D_o | K(D_c + D_o) + S(D_c + D_o) | linear in K |
| `life-sketch` | as `life` + Z (Hamming) | + 8Z bytes | Z linear; reach linear in Z |

- **Facts.** Reach in intervening surprises grows linearly with bytes, for any trace-based arm **[derived]**. No bounded store reaches every gap, and the sketch only changes the constant, by about 50× in bytes per entry (400 against 8). Slots grow with what is kept, and an index would make the read sublinear; neither is run.
- **Read cost.** Measured once before design: about 7 µs for a 640 × 64 store read, single-thread numpy.
- **Probes.** Causes ×4 (g ≥ 512) with frozen settings, to see whether thresholds survive load. Occurrences n ∈ {4, 8, 32} at g = 512, to measure how many exposures each arm needs, which is where `slow` should catch up.
- **What would not survive scale-up.** 32 outcomes and decoding through an evaluator-held table; a linear reader; a family glyph handing the rule its feature; a keyspace of 512 glyphs. Real causes will not announce themselves in one 32-dimensional slot. Thresholds on cosines drifted across encoders in streaming retrieval, so expect this here.

## Computation plan

*Estimates, not measurements.*

- **Construction.** Encoders: seeds 299 and 298 exist; 380–389 about 70 s each on the GTX 1650 (`/data/venvs/sri-cuda`). Rendering pool: 512 symbols × 64 renderings × 2 conditions per seed, under a minute of CPU each.
- **Pilot.** The grid above, pruned by coordinate search on shared settings, 2 seeds, primary cell, all gaps. Expect 1–4 CPU-hours on 11 workers, rejected settings counted.
- **Confirmation.** 15 arms × 10 seeds × 5 gaps × (2 values of q): mean stream about 87,000 events, roughly 5–30 s per run. About 10–30 minutes wall. Other B, `clean` and probes add about the same.
- **Deterministic and timed output kept apart.** `reports/surprise-lifecycle-v1-{pilot,confirm,analysis}` and separate `-runtime` files, per [computation reporting](computation-reporting.md), with executed runs, events, shadows proposed and candidates scored. `timing` measures one thread, warm-up, 15 repeats, median and spread.
- **Persistence.** Tasks durable (`experiments/durable.py`); rerunning resumes. Pools written atomically under gitignored `runs/`.
- **Dtype.** float32. Raw images are not retained after encoding; pools hold keys.

## Reproduction

To be written with the implementation: `src/surprise_lifecycle.py`, `experiments/surprise_lifecycle.py`, `tests/test_surprise_lifecycle.py`.
