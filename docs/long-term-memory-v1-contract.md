# Long-term memory v1: contract

Status: **draft, revised 2026-11-01 after external review. Nothing built or run; not frozen.** Values marked *provisional* may change once, on pilot seeds, before confirmation seeds exist. Every change is recorded under *Changes from the pilot*.

## Question

Step 1 of the user's target architecture: `sparse48`, the best earlier architecture for relations, with a shared core receiving all senses and sparse readers, plus memory networks combined with it in the forward pass. This study adds one of them.

**Can a long-term memory network, combined with `sparse48` in the forward pass and unable to change the core, retain knowledge that does not fit relations, without costing relational transfer?**

Earlier results set the bar. `sparse48` has the highest transfer, but every arm fell to 0.12–0.15 on arbitrary pairs after one more block (chance 0.11), and adding dense units inside the host cost transfer ([arbitrary pairings](arbitrary-pairings-v1-results.md)). An explicit store kept 0.94 of old words in [translation memory](translation-memory-v1-pilot.md), while the host kept 0.00.

**Scope.** No recent-window memory, surprise, gists or gates, which are later steps. No attention or sleep. Learned key and value projections are included, as the user asked, since the host's training provides their outer loop. A pass shows that retention is possible beside sparse relations, with material chosen by the curriculum. It does not show how a system would select what to remember.

## Architecture

**Host.** `sparse48` exactly as in arbitrary pairings Option A: image and sequence encoders, shared 16-dimensional code, sparse readers `g` and `P` with RigL, Adam, the same losses and streams. RigL ends at update 4,500, so connectivity is fixed during pairing.

**Combining in the forward pass.** For a query code q and candidate code c, the matching head scores:

`score(q, c) = reader(q, c) + s · ⟨M(k(q)), v(c)⟩`

where M is the memory network, k and v its key and value, and s a learned scale starting at 0, so memory begins silent. The readers' head is the existing `relation_logits` head 1. Successor and comparison heads are unchanged. Answers need no router: memory adds to the readers' evidence.

**Two rules keep memory from eroding the core.**

1. **Nothing flows back.** Keys and values are computed from detached inputs. No memory gradient reaches an encoder, projection, core parameter or reader.
2. **Readers keep their pressure.** The host trains on its original loss with memory absent, so memory cannot answer for the readers and relieve them. **Memory trains on the readers' errors:** on each batch's matching items, cross-entropy over the 9 candidates of `reader(q, c).detach() + s⟨M(k(q)), v(c)⟩`.

So the host's weights are **bitwise identical** to plain `sparse48` on the same seed, checked by gate 1. The core cannot be eroded, by construction. What can still go wrong is the combined answer on relational tasks, so that is read directly.

**One host run per seed, replayed.** Since the host never depends on memory, each host run writes a **trajectory cache**: for every update's matching items and every evaluation point, the readers' logits, the live and `slow` codes of query and candidates, and the image indices for `frozen` keys. Memory variants then train offline, on CPU, from the cache, in any order. So adaptive selection needs no host reruns, every variant sees exactly the same batches, and variants are paired exactly. The hook captures under no-grad and draws no random numbers. Its storage and time are charged.

**Which experiences train memory.** Every matching item in a batch: the curriculum's matching plus, in pair blocks, the 16 pair items. No selection. The residual loss is near zero where readers already answer, so relational items should write little. Whether that holds is read through the combined answer's transfer.

## Memory variants

All have the same outer form: `k = W_K · key(x)`, `v = W_V · value(x)`, both d = 32, then M from R^32 to R^32.

**Network M.** State of the art family, simplified.

| form | M | corresponds to |
|---|---|---|
| `linear` | one matrix | linear attention; delta rule; ATLAS Appendix G's linear case |
| `mlp` | 2-layer tanh MLP, width filling the budget | Titans and OmegaNet's memory, with no kernel |
| `k-sparse` | as `mlp`, but keeping the top p = 5% of hidden activations per input, others zero | local activation to limit interference; related to the sparse experts of Hope and Sleep |

**Update rule.** Inspired by Titans Eqs. 13–14 and Hope Eq. 71, for all of M, W_K, W_V and s:

every C updates, with ḡ the mean gradient over those C updates, clipped to norm 5 as the host clips:

`S_t = β^C S_{t−C} − η C ḡ`, `θ_t = θ_{t−C} + S_t`.

`η C ḡ` is the sum in Hope's Eq. 71. With this scaling and β^C, total learning per update and momentum's duration in updates stay the same as C changes, so **C changes frequency only, as far as that can be separated**; the noise in ḡ still differs, and that is part of the effect. α = 0, so there is no decay: this is long-term memory. No Adam, whose inherited state alone damaged recall in the [forgetting study](forgetting-mechanism-v1-results.md).

**Projections.** `learned`: W_K, W_V train as above. `fixed`: seeded Gaussian, never trained, which is the cheap baseline.

**Key source.** The host's code drifts: old keys fell to 0.20 recognition or below in translation memory.

| source | key(x) and value(x) | stable? | cost, outside the memory budget |
|---|---|---|---|
| `live` | the host's current code, detached | no | none |
| `slow` | a slow-moving copy of the image encoder and projection, updated towards the host every update with half-life 2,000 updates; its 16-dimensional code | mostly; moves with the host | one encoder copy |
| `frozen` | the [frozen key encoder](translation-memory-v1-pilot.md#stable-keys-perception-gate-passes), 32 dimensions, for queries; values as gate 2 decides | fully | 839,040 bytes |

**Memory acts only when query and candidate are both images,** for every key source, `live` included. A sequence (taps) has no frozen or slow key or value, so memory's term is 0 for taps queries and taps candidates. Applying this to all key sources keeps key-source comparisons on the same tasks. **The cost:** memory cannot help cross-sense facts in this study. A sequence pathway is a later question.

**References, not arms of the system:** `sparse48` alone. `flat-store`: an explicit diagnostic that writes each pair's frozen key and target count once and answers by nearest key, for what perfect storage scores on these keys. `mixed` from the same protocol, for a transfer cost inside the host.

## Material and schedule

As arbitrary pairings Option A, with a third phase:

| phase | updates | trains |
|---|---|---|
| blocks 1–3 | 6,200 in total | curriculum, no rehearsal |
| block 4 | 2,000 | context plus pair set **A** |
| block 5 | 2,000 | context plus pair set **B** |
| block 6 | 2,000 | context only, the delay |

N = 45 pairs per set, 64 training renderings, 5 evaluation renderings per symbol, balanced over nine counts. Symbols are from the 512-symbol pool, which no frozen encoder trained on. Other pair-set sizes: 18 and 90, as a load probe.

**Seeds.** Pilot 392 and 393. Confirmation 394–403. All unbuilt, and neither the host runs nor the encoders are reused.

## Budget

Memory's persistent state, in float32: W_K, W_V, M, s, momentum S and the accumulated gradient: **64 KiB** *(provisional)*. **Width is set once per form,** charging three copies of the weights (weights, S, accumulator) for every setting of β, C and projections, whether used or not. So a setting cannot change capacity. The unused state of a setting is reported, not reassigned. Outside it and reported: the host and any key pathway. Maximum bytes per stored pair is reported at each load. `flat-store`'s 90 pairs of 64 floats take about 23 KB.

## Readings

On independent renderings, with memory's state unchanged. Nine candidates in seeded positions, as in earlier studies. Chance 0.111 for recall.

**Primary, at the end of block 6.**

- **Retained A:** combined recall of set A, after B and the delay.
- **Retained B:** combined recall of set B, after the delay.
- **Answer transfer:** the only inherited transfer measures memory can affect, combined: the mean of matching on withheld counts for Chinese and for Devanagari digits, `chinese_withheld_matching` and `devanagari_withheld_matching`.

**Invariance checks, not endpoints.** Coordinate transfer (a) reads host codes, and never uses matching scores, so it cannot see memory. The comparison measures and the taps measures do not pass through memory's term. All must equal plain `sparse48` exactly, and a difference is a bug.

**Reported beside them.** `sparse48` alone on all three. Combined matching on paired and withheld counts for all three scripts. Recall of A and B at the end of blocks 4 and 5, and curves. Memory alone, readers silenced. **Unknown symbols:** on a third set never taught, the share of queries where memory's term changes the readers' choice, a measure of confident error. Combined recall of A by batch age and by load. Cost: bytes, operations, GPU time per update.

**Checked once per host run.** Host weights' hash at each block's end, recorded in the cache, which all variants share.

## Gates

On pilot seeds, before selection. Failure stops the study until the design changes, and is recorded.

1. **The host is untouched.** Host weights, with the cache hook attached, are bitwise identical at each block's end to `sparse48` trained without it. Reader logits replayed from the cache equal those recomputed from a checkpoint. If GPU nondeterminism prevents bitwise equality with the same code path, the check is a same-process rerun, and any gap is reported. Memory never runs in the host process, so its gradients cannot reach host parameters.
2. **Key quality, for `frozen`.** The frozen encoder is trained on symbols, and may not separate collections of dots. On evaluation renderings: 45-way symbol identity, and 9-way count identity of collections by nearest prototype. If count identity is under 0.80, `frozen` uses `slow` for values, and the study reports it as frozen keys only.
3. **Silence.** At s = 0, combined scores equal reader scores exactly.
4. **Storage ceiling.** `flat-store` reaches retained A ≥ 0.80 at N = 45, else the keys cannot support the bar, and the bar is revised before any variant is compared.
5. **Audit.** Reading a variant leaves its state bitwise unchanged, and shuffled targets score chance.

## Selection and decision rule

**Search,** pilot seeds, N = 45, offline from the trajectory cache, as a **sequential sweep, not a Cartesian grid.** Each stage holds everything at the previous stage's winner, others at their start values:

1. Form, on `frozen` keys, the best case for storage, with `learned` projections.
2. `learned` against `fixed`.
3. Key source.
4. Each setting below, once, in the table's order.

That is 14 configurations per seed, known only as stages finish.

| setting | values | start |
|---|---|---|
| η | 0.001, 0.003, 0.01, 0.03 | 0.01 |
| β | 0, 0.9 | 0.9 |
| C | 1, 16, 64 | 16 |
| p, `k-sparse` | 2%, 5%, 10% | 5% |

**Objective** *(provisional)*: among variants **eligible** by answer transfer at least `sparse48`'s minus 0.03, the highest mean of retained A and retained B. Ties to fewer operations. If none is eligible, the smallest transfer shortfall, marked. One variant is chosen, the **candidate**.

**Key-source preference.** `frozen` may not win by default. At stage 3, among eligible key sources, `live` wins if its objective is within 0.03 of the best; otherwise `slow` if within 0.03; otherwise the best.

**Confirmation.** Seeds 394–403, settings frozen. Paired by seed; percentile bootstrap, 10,000 resamples; 95% two-sided.

A candidate **retains** if, against `sparse48` alone, retained A and retained B each gain at least 0.15 with lower bound above 0.

On answer transfer against `sparse48`, margin 0.05, the audit margin of earlier studies: **holds** if the lower bound is at least −0.05; **lost** if the mean is at most −0.05 and the upper bound below 0; otherwise **inconclusive**.

**Near storage** if the lower bound of retained A, candidate minus `flat-store`, is at least −0.15. A point estimate is not enough.

| outcome, first that applies | condition | meaning |
|---|---|---|
| `long-term-works` | retains, transfer holds, near storage | a network can hold specifics beside sparse relations. Step 2 follows |
| `retains-store-gap` | retains, transfer holds, near storage not established | it works; closeness to perfect storage is not shown, and the gap's interval is reported. Report what limits it before step 2 |
| `retains-at-transfer-cost` | retains, transfer lost | the combined answer disturbs relations, though the core is untouched |
| `retains-transfer-inconclusive` | retains, transfer inconclusive | retention is shown. Preservation is neither shown nor refuted, and step 2 waits on more seeds |
| `no-retention-shown` | none of the above | this network did not retain, on this evidence. Failed superiority, not proof that no network can; intervals reported |

**Secondary: ablations, each changing one thing from the candidate, all else frozen at its settings.** Each runs on all 10 confirmation seeds, on retained A, retained B and answer transfer, reading better, equivalent or inconclusive at margin 0.05, unadjusted:

| ablation | changes | question |
|---|---|---|
| other projection | `learned` or `fixed`, whichever the candidate is not | do learned projections earn their place? |
| other key sources | each of `live`, `slow`, `frozen` the candidate is not | the size of the drift problem |
| other forms | each of `linear`, `mlp`, `k-sparse` the candidate is not, width by the budget rule | does the form matter? |
| other update frequency | C = 1 if the candidate's C > 1, otherwise C = 16 | Hope's update frequency |

Six variants. Candidate against `mixed` on retained A and, since `mixed` has no memory term, on the same answer-transfer measures, is reported beside them. Pilot-stage comparisons are exploratory, as their winners differ in other settings.

**Expectations, recorded before any run.** `frozen` keys retain best and `live` keys go stale by block 6. `k-sparse` beats `mlp`, since interference among shared weights has been the main loss so far. Learned projections beat fixed, modestly. Answer transfer holds, since memory should write little where readers are right. `retains-store-gap` is the most likely outcome. The largest risk: Neural memories forgot quickly in the [relevant update pilot](relevant-update-v1-pilot.md), though those were small and fast, with no slow schedule.

## Before freezing

- Hope's sections beyond the continuum memory, and its forgetting results if any; Sleep's expansion rule, since growing capacity is excluded here.
- Sparse activation for continual learning, which is recalled from memory and unread.
- Whether "memory acts only on image queries" costs readings that use taps; the transfer measures make this visible.
- Attainability of the objective's transfer limit and the 0.15 gains, from gates and pilot.

## Changes from external review

First review, before any build, all correct and fixed above:

- Coordinate transfer (a) reads host codes and never matching scores, so it could not see damage from memory. Answer transfer replaces it as the endpoint, and it stays as an invariance check.
- Failing to establish preservation read as demonstrated harm. Transfer now reads holds, lost or inconclusive, with its own outcome, and nearness to `flat-store` is an interval bound.
- Sequence candidates had no value pathway under `slow` and `frozen`. Memory now acts only on image-image pairs for every key source.
- C changed optimization and capacity as well as frequency. Updates are normalized, clipping is defined, and width is fixed per form.
- Secondary comparisons had no runs. They are now six enumerated ablations on all confirmation seeds, and pilot comparisons are exploratory.
- Adaptive selection could not run on a single live host. A trajectory cache replays it offline, the search is a defined sequential sweep, and the key-source preference names its metric and keeps eligibility.

## Changes from the pilot

None yet.

## Prior work

Shared survey: [prior-work.md](prior-work.md#long-term-memory-a-network-beside-sparse-relations).

**Established names.** Neural memory with online updates: Titans **[read: Eqs. 11–15]**, ATLAS **[read: pp. 5–6, Appendix G]**. Multi-frequency memory: Hope's continuum memory system, a chain of blocks with chunked updates, Eq. 71 **[read through a reader: §7]**. Combining a slow learner with a fast memory: complementary learning systems **[abstract, earlier survey]**. Training on another model's errors: boosting and residual learning **[familiar]**. Slow copy of an encoder: momentum encoders (MoCo) **[familiar; read before use]**. Local activation limiting interference: sparse distributed representations and k-winners **[familiar; read before use]**.

**What this differs from.** Hope's blocks share gradient flow with the model and a Transformer task. Here memory is detached and residual. Titans' memory is reset per context; here it persists across the whole schedule, which those papers do not claim: ATLAS says no persistent learning typically carries over once the memory is cleared.

**What follows without running [derived].**

- With rule 1 and rule 2, the core and readers are unchanged. So coordinate transfer (a), comparison and taps measures equal `sparse48`'s, and any cost must appear in the combined matching.
- A network with no decay and no replay can still lose A while learning B, by interference. Only update frequency and local activation stand against that here. So step 1 tests whether storage alone holds; if not, step 2's recent memory and replay may be needed for long-term storage, not only for acquisition.
- With s starting at 0 and a residual loss, memory's write is proportional to the readers' error. For items readers know, it writes little. Items they get wrong but later learn will have been written, so the two may overlap with no harm.
- Memory's values depend on collection codes under `live` and `slow`. If count stays decodable as the forgetting study found, values drift more slowly than symbol keys, but that was measured with labels supplied.

**Known failure modes and handling.** Silent memory: s stuck at 0 reads as no retention; memory-alone recall separates it. Confident errors on unknown symbols: read directly. A combined answer dominated by memory: readers-alone beside it. GPU nondeterminism masking host identity: gate 1's fallback. Selection among many variants on two seeds: stages, one candidate, fresh seeds. The frozen encoder is a scoped baseline, and the user does not want perception frozen in the end; the `slow` comparison is there for this.

**Baselines.** `sparse48`, `flat-store` and `mixed`. Retention remedies' `rehearse-M` read 0.47 by 2,000 updates, on marked pairs, with a replay buffer; this study has no buffer, so it is context only.

**What this adds.** A memory network beside sparse relations that cannot erode them by construction, trained on residuals and read with and without its contribution. No match was found in these few searches, which is not a novelty claim.

## Scaling

P weights in M, d = 32, code width 16, B batch matching items, 9 candidates, F pairs.

| part | per update | persistent | what limits scale |
|---|---|---|---|
| Host | as `sparse48` | as `sparse48` | unchanged |
| Trajectory cache | one code per query and candidate | workspace only, not part of the system | disk, during the study |
| Key pathway, `live` | none extra | none | stale keys |
| Key pathway, `slow` | one encoder forward per image, and an encoder-sized update | one encoder | model size |
| Key pathway, `frozen` | one encoder forward per image | 839,040 bytes | fixed; independent of F |
| M forward and gradient | B·9·P | P + P (S) + P (accumulator) | bytes bound capacity |
| Projections | B·9·(d·16 or d·32) | up to 2·16·d floats | fixed |

- **Facts.** Memory's size is fixed, so capacity in pairs is a measured limit, and failure may be abrupt, as clipped associative memory was. Read work does not grow with F, unlike slots, which is the one scaling argument for a network over `flat-store`. Whether it holds depends on the load probe at N = 18, 45, 90.
- **What would not survive scale-up.** 9-way matching among candidates shown. 16-dimensional code. 45 symbols. A frozen encoder trained on 2,000 background symbols. Curriculum-chosen material.

## Computation plan

*Estimates, not measurements.*

- **Construction.** Encoders 12 × about 70 s on the GTX 1650. Symbol material as arbitrary pairings, under a minute per seed.
- **Theoretical.** Host O(updates × batch × active weights). Each variant O(updates × B × 9 × P) added, O(P) stored. Readings O(N × E × 9) forward passes per point.
- **Host and cache.** 2 pilot and 10 confirmation host runs, 12,200 updates each; 321 s alone per run for a comparable arm in the forgetting study, so about 6 minutes each. References: `mixed` as one further host run per seed. **Cache,** workspace: about 1,300 bytes per matching item, 16 live and 16 `slow` floats for the query and each of 9 candidates, plus 9 logits. Taken at about 40 items per update, an estimate to be measured, that is about 640 MB per seed, under gitignored `runs/`.
- **Selection.** 14 configurations × 2 seeds, each 12,200 cached updates on CPU, with readings at block ends, on 11 workers. About 1–2 CPU-hours.
- **Confirmation.** Candidate and 6 ablations, 10 seeds: 70 replays, under 6 CPU-hours. `flat-store` is cheap. Load probes at N = 18 and 90, candidate and `flat-store`, on 3 seeds, need 6 more host runs, and are exploratory.
- **Deterministic and timed output kept apart.** `reports/long-term-memory-v1-{gates,pilot,confirm,analysis}` and separate `-runtime` files, per [computation reporting](computation-reporting.md).
- **Persistence.** Durable tasks; checkpoints written atomically at block ends, since these runs are long.
- **Device.** GPU, as the study this compares with. Not comparable with CPU runs.

## Reproduction

To be written with the implementation: `src/long_term_memory.py`, `experiments/long_term_memory.py`, `tests/test_long_term_memory.py`.
