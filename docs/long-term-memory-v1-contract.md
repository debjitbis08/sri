# Long-term memory v1: contract

Status: **draft, 2026-11-01. Nothing built or run; not frozen.** Values marked *provisional* may change once, on pilot seeds, before confirmation seeds exist. Every change is recorded under *Changes from the pilot*.

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

**This allows one host run per seed to feed every memory variant at once.** Variants differ only in memory, see the same batches, and are paired exactly.

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

`S_t = β S_{t−1} − η Σ_{chunk} ∇L`, `θ_t = θ_{t−1} + S_t`, applied every C updates, the gradients accumulated between.

α = 0, so there is no decay: this is long-term memory. No Adam, whose inherited state alone damaged recall in the [forgetting study](forgetting-mechanism-v1-results.md). Gradient clipping at norm 5, as the host. `C` is a low update frequency in the sense of Hope's continuum memory.

**Projections.** `learned`: W_K, W_V train as above. `fixed`: seeded Gaussian, never trained, which is the cheap baseline.

**Key source.** The host's code drifts: old keys fell to 0.20 recognition or below in translation memory.

| source | key(x) and value(x) | stable? | cost, outside the memory budget |
|---|---|---|---|
| `live` | the host's current code, detached | no | none |
| `slow` | a slow-moving copy of the image encoder and projection, updated towards the host every update with half-life 2,000 updates; its 16-dimensional code | mostly; moves with the host | one encoder copy |
| `frozen` | the [frozen key encoder](translation-memory-v1-pilot.md#stable-keys-perception-gate-passes), 32 dimensions, for queries; values as gate 2 decides | fully | 839,040 bytes |

Memory acts only where the query is an image. A sequence (taps) query has no frozen or slow key, so memory's term is 0 there. For `live` it acts on all queries.

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

Memory's persistent state, in float32: W_K, W_V, M, s, momentum S and the accumulated gradient: **64 KiB** *(provisional)*. Width fills it. So for `mlp` at C > 1 with β > 0, three copies of the weights are charged. Outside it and reported: the host and any key pathway. Maximum bytes per stored pair is reported at each load. `flat-store`'s 90 pairs of 64 floats take about 23 KB.

## Readings

On independent renderings, with memory's state unchanged. Nine candidates in seeded positions, as in earlier studies. Chance 0.111 for recall.

**Primary, at the end of block 6.**

- **Retained A:** combined recall of set A, after B and the delay.
- **Retained B:** combined recall of set B, after the delay.
- **Transfer (a), combined:** average-coordinate transfer through the combined answer, the [earlier measure](limited-memory-v1-results.md).

**Reported beside them.** `sparse48` alone on all three. Each of the eight transfer measures, combined. Recall of A and B at the end of blocks 4 and 5, and curves. Memory alone, readers silenced. **Unknown symbols:** on a third set never taught, the share of queries where memory's term changes the readers' choice, a measure of confident error. Combined recall of A by batch age and by load. Cost: bytes, operations, GPU time per update.

**Reads checked once per host run.** Host weights' hash at each block's end. Identical for all variants.

## Gates

On pilot seeds, before selection. Failure stops the study until the design changes, and is recorded.

1. **The host is untouched.** Host weights, after training with all variants attached, are bitwise identical at each block's end to `sparse48` trained alone, with independent random streams. If GPU nondeterminism prevents bitwise equality with the same code path, the check is a same-process rerun, and any gap is reported. Gradients to host parameters from memory's loss are asserted None.
2. **Key quality, for `frozen`.** The frozen encoder is trained on symbols, and may not separate collections of dots. On evaluation renderings: 45-way symbol identity, and 9-way count identity of collections by nearest prototype. If count identity is under 0.80, `frozen` uses `slow` for values, and the study reports it as frozen keys only.
3. **Silence.** At s = 0, combined scores equal reader scores exactly.
4. **Storage ceiling.** `flat-store` reaches retained A ≥ 0.80 at N = 45, else the keys cannot support the bar, and the bar is revised before any variant is compared.
5. **Audit.** Reading a variant leaves its state bitwise unchanged, and shuffled targets score chance.

## Selection and decision rule

**Search,** pilot seeds, N = 45. In order, each from the previous winner: form on `frozen` keys, the best case for storage, at `learned` and starting settings; then `fixed` against `learned`; then key source; then the settings below.

| setting | values | start |
|---|---|---|
| η | 0.001, 0.003, 0.01, 0.03 | 0.01 |
| β | 0, 0.9 | 0.9 |
| C | 1, 16, 64 | 16 |
| p, `k-sparse` | 2%, 5%, 10% | 5% |

**Objective** *(provisional)*: among variants with combined transfer (a) at least `sparse48`'s minus 0.03, the highest mean of retained A and retained B. Ties to fewer bytes, then fewer operations. If none qualifies, the smallest transfer shortfall, marked. One variant is chosen, the **candidate**. `frozen` keys may not win by default: if `slow` or `live` is within 0.03 of it, the less costly and less frozen wins.

**Confirmation.** Seeds 394–403, settings frozen. Paired by seed; percentile bootstrap, 10,000 resamples; 95% two-sided.

A candidate **retains** if, against `sparse48` alone, retained A and retained B each gain at least 0.15 with lower bound above 0. **Transfer holds** if the combined (a) difference from `sparse48` has lower bound at least −0.05. That margin is the audit margin used in earlier studies.

| outcome, first that applies | condition | meaning |
|---|---|---|
| `long-term-works` | retains, transfer holds, and retained A within 0.15 of `flat-store` | a network can hold specifics beside sparse relations. Step 2 follows |
| `retains-below-store` | retains, transfer holds, not within 0.15 of `flat-store` | it works, but well below perfect storage; report what limits it before step 2 |
| `retains-at-transfer-cost` | retains, transfer does not hold | the combined answer disturbs relations, though the core is untouched |
| `no-retention-shown` | none of the above | this network did not retain, on this evidence. Failed superiority, not proof that no network can; intervals reported |

**Secondary, each better, equivalent or inconclusive, margin 0.05:** `learned` against `fixed` projections. `live` against `slow` and `frozen` keys, the size of the drift problem. `mlp` against `k-sparse` and `linear`. C = 1 against the chosen C, Hope's update frequency. Candidate against `mixed` on transfer and retained A.

**Expectations, recorded before any run.** `frozen` keys retain best and `live` keys go stale by block 6. `k-sparse` beats `mlp`, since interference among shared weights has been the main loss so far. Learned projections beat fixed, modestly. Transfer holds, since memory should write little where readers are right. `retains-below-store` is the most likely outcome. The largest risk: Neural memories forgot quickly in the [relevant update pilot](relevant-update-v1-pilot.md), though those were small and fast, with no slow schedule.

## Before freezing

- Hope's sections beyond the continuum memory, and its forgetting results if any; Sleep's expansion rule, since growing capacity is excluded here.
- Sparse activation for continual learning, which is recalled from memory and unread.
- Whether "memory acts only on image queries" costs readings that use taps; the transfer measures make this visible.
- Attainability of the objective's transfer limit and the 0.15 gains, from gates and pilot.

## Changes from the pilot

None yet.

## Prior work

Shared survey: [prior-work.md](prior-work.md#long-term-memory-a-network-beside-sparse-relations).

**Established names.** Neural memory with online updates: Titans **[read: Eqs. 11–15]**, ATLAS **[read: pp. 5–6, Appendix G]**. Multi-frequency memory: Hope's continuum memory system, a chain of blocks with chunked updates, Eq. 71 **[read through a reader: §7]**. Combining a slow learner with a fast memory: complementary learning systems **[abstract, earlier survey]**. Training on another model's errors: boosting and residual learning **[familiar]**. Slow copy of an encoder: momentum encoders (MoCo) **[familiar; read before use]**. Local activation limiting interference: sparse distributed representations and k-winners **[familiar; read before use]**.

**What this differs from.** Hope's blocks share gradient flow with the model and a Transformer task. Here memory is detached and residual. Titans' memory is reset per context; here it persists across the whole schedule, which those papers do not claim: ATLAS says no persistent learning typically carries over once the memory is cleared.

**What follows without running [derived].**

- With rule 1 and rule 2, the core and readers are unchanged. So the readers-alone transfer equals `sparse48`'s, and any cost must be in the combined answer.
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
- **Selection.** About 40 variants per host run in stages, 12,200 updates, 2 seeds. An arm-seed took 321 s alone in the forgetting study. Variants are small, but 40 run per update, so expect 1–3 hours per seed on the GPU, single process.
- **Confirmation.** Candidate and references, 10 seeds, and N = 18 and 90 on 3 seeds. Under a day of GPU with 6 processes.
- **Deterministic and timed output kept apart.** `reports/long-term-memory-v1-{gates,pilot,confirm,analysis}` and separate `-runtime` files, per [computation reporting](computation-reporting.md).
- **Persistence.** Durable tasks; checkpoints written atomically at block ends, since these runs are long.
- **Device.** GPU, as the study this compares with. Not comparable with CPU runs.

## Reproduction

To be written with the implementation: `src/long_term_memory.py`, `experiments/long_term_memory.py`, `tests/test_long_term_memory.py`.
