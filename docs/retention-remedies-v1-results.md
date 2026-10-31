# Retention remedies v1: results

Frozen contract: [retention-remedies-v1-contract.md](retention-remedies-v1-contract.md) (2026-10-31); [pilot](retention-remedies-v1-pilot.md) on seed 299. Seeds 350–359, arms `sparse48`, `mixed`,
`mixed-h125`, N = 45 pairs of which 18 are marked (M, two per count) and 27 unmarked (U), 2,000 context updates from the end of block 4, Option A, GPU. Deterministic results:
`reports/retention-remedies-v1-analysis.json` from `reports/retention-remedies-v1-results.json.gz`; derived relations: `reports/retention-remedies-v1-derived.json`. No arm, seed, branch,
parameter, endpoint or reading was changed after results were seen. Items marked **post hoc** were added afterwards.

Departures from the contract's wording, stated before the results: (a) the static branches were computed in every arm, although the contract lists the cue variants for `mixed` only; their other-arm values are descriptive. (b) The derived-relations reading was not recorded by the main runner; it was computed in a separate deterministic pass for `mixed`, checked against the main records (below). (c) The contract's "not beneficial" is read as "a meaningful benefit is excluded" (upper bound at most +0.05). (d) The analysis labels every benefit over `ctx` exploratory, as the contract says.

All recall values are nine-candidate accuracy, chance 0.111; differences are paired over ten seeds with 95% t-intervals; "established meaningful" (the stronger rule) means the lower bound exceeds +0.05.
Recall on M at the end of block 4, original matching protocol (canonical candidates in brackets): `sparse48` 0.357 (0.342), `mixed` 0.502 (0.468), `mixed-h125` 0.464 (0.417).

## Reading 1: checks
All pass in every arm: destination parameters unchanged after distillation (bitwise hash), the frozen image-to-code path unchanged in `stable-path-replay`, the buffer holds only M (four renderings per pair),
M balanced at two per count. `ctx` declines on M as in earlier studies (−0.23, −0.28, −0.35 by 2,000 updates).

## Reading 2: primary contrast (mixed)
**Destination (`consolidate-cache`, cues discarded) minus `frozen-snapshot` on M: −0.043 [−0.089, +0.002]. Noninferiority at margin 0.05 is not established** (nor is the destination shown worse).
Compression: the destination's persistent storage is **0.941** of the snapshot's (866,404 against 920,616 bytes); the frozen image-to-code path (859,712 bytes) dominates both.

**Post hoc decomposition.** The two sides were scored differently: the destination nine-way on canonical candidates, the snapshot by the original matching protocol. Splitting the difference:
evaluation format (teacher on canonical candidates minus teacher by original matching) −0.034 [−0.084, +0.015]; distillation (destination minus teacher, both on canonical candidates) −0.009 [−0.050, +0.032].
Most of the gap is the scoring format, not lost information; this was a flaw in how the primary was specified and does not change its frozen outcome.

## Reading 2 (exploratory): retention benefit over `ctx` on M at 2,000 updates
`ctx` recall on M at 2,000: `sparse48` 0.130, `mixed` 0.222, `mixed-h125` 0.116. E = established meaningful; i = inconclusive; nb = not beneficial.

| Branch | sparse48 | mixed | mixed-h125 | mixed absolute |
|---|---|---|---|---|
| `ctx-m0` | — | −0.028 nb | +0.007 nb | 0.194 |
| `hold-M` (unbounded) | +0.387 E | +0.388 E | +0.497 E | 0.610 |
| `stable-path-replay` | — | +0.299 [+0.250, +0.348] E | — | 0.521 |
| `der-M` (weight 0.01) | +0.220 E | +0.262 E | +0.279 E | 0.484 |
| `der-M-strong` (0.1) | +0.214 E | +0.249 E | +0.273 E | 0.471 |
| `rehearse-M` | +0.211 E | +0.249 [+0.196, +0.302] E | +0.303 E | 0.471 |
| `rehearse-M-m0` | — | +0.223 E | — | 0.446 |
| `frozen-snapshot` | +0.227 E | +0.280 E | +0.349 E | 0.502 |
| `consolidate-cache` | +0.219 E | +0.237 [+0.148, +0.326] E | +0.297 E | 0.459 |
| `consolidate-cache1` | +0.197 E | +0.216 E | +0.283 E | 0.438 |
| `consolidate-other` | +0.184 E | +0.200 E | +0.242 E | 0.422 |
| `consolidate-random` | +0.063 i | +0.051 i | +0.034 i | 0.273 |
| `consolidate-cache-anchor` | +0.123 E | +0.129 i | +0.201 E | 0.351 |
| `consolidate-cache1-anchor` | +0.081 i | +0.069 i | +0.164 E | 0.291 |

Every bounded replay remedy and every cue-carrying destination keeps far more of M than `ctx`. The destination comparisons against `ctx` favor protected stores by construction (contract), so they are reported, not counted as evidence of a mechanism.

## Reading 3: preservation (M at 2,000 minus the branch's own start)
- **Preserved** (noninferiority established): `hold-M` in all arms (+0.11 to +0.16, it exceeds its start); `stable-path-replay` (+0.019 [−0.010, +0.048]); `consolidate-cache` in all arms (`mixed` −0.009 [−0.050, +0.032], the lower bound at the margin); `der-M` in `sparse48` only.
- **Partial rescue** (meaningful benefit, preservation not established): `rehearse-M`, `rehearse-M-m0`, `der-M`, `der-M-strong` (`mixed` −0.018 to −0.057), `consolidate-cache1`, `consolidate-other`.
- **Inconclusive:** `consolidate-random` and the anchored heads in `mixed`.

## Reading 4: selectivity (M minus U at 2,000; `mixed`)
`hold-M` +0.33, `rehearse-M` +0.17, `der-M` +0.13, `consolidate-cache` +0.10, `stable-path-replay` +0.09, `consolidate-other` +0.005, `ctx` +0.04. The cache destination answers unmarked symbols at 0.357
(teacher 0.455 on U): it learned part of the teacher's general code-to-count map, not only the marked facts. Other-symbol cues give a destination with no preference for M at all.

## Reading 5: cue sources (`mixed`; other arms descriptive and similar)
- Cache minus random: **+0.186 [+0.083, +0.288], established meaningful.** Random codes give little access to the teacher's useful behavior.
- Cache minus other symbols: +0.037 [+0.002, +0.072], inconclusive under the stronger rule (the interval excludes 0, but not a meaningful gap). Unrelated real symbols recover most of what the cache gives, with no per-fact storage.
- Cache (4 cues) minus one cue: +0.021 [−0.014, +0.056], inconclusive.

## Reading 6: cost gates (noninferior: lower bound > −0.05)
- **(i) Transfer against concurrent `sparse48` `ctx`.** Fails for every `mixed` branch, including `ctx` itself (`ctx` −0.028 [−0.067, …]): the memory component's transfer cost, which no remedy changes. Passes for every `sparse48` branch and for `mixed-h125` except `hold-M` (−0.020, lower bound −0.056).
- **(ii) Learning set B after 250 updates, against the same arm's `ctx`.** Passes everywhere except `stable-path-replay` (−0.066 [−0.117, …]): freezing the image-to-code path costs new learning.
- **(iii) Context performance.** Passes everywhere except `mixed-h125` `rehearse-M` (lower bound −0.053, marginal).
For the consolidation branches the host is `ctx` (consolidation never touches it), so (ii) and (iii) are zero by construction and (i) is `ctx`'s.

## Reading 7: resources (budget: one model, 1,489,192 bytes; 25% of the `ctx` phase's time)
Persistent bytes: `rehearse-M` 225,792; `der-M` 256,608; `stable-path-replay` 4,608 (stored codes; it pins the host's own path); destinations 864,000–866,404; `frozen-snapshot` 920,616; `hold-M` 3,612,672 (the full rendering pool; over budget). Temporary consolidation storage 7,776 bytes (cache) or 2,376 (one cue), deleted after distillation.
Extra time over `ctx` (`mixed`; timing in `reports/retention-remedies-v1-analysis-runtime.json`): replay branches within ±3% (the replay items are a small addition and timing noise is of that size); distillation about 2% (nine-way) and 5% (anchored). All within the time budget.

## Reading 8: verdicts
- **Useful remedy (exploratory):** bounded rehearsal (`rehearse-M`), DER at both weights and `stable-path-replay` are established meaningful in every arm where they ran, as are the cue-carrying destinations.
- **Protected-store candidate meeting component-level gates:** `consolidate-cache` in `mixed-h125` and in `sparse48`; **not in `mixed`**, only because the host fails transfer gate (i). This shows complementary capabilities in separate components, not a system that uses them together (routing untested); it is not a successful combined architecture.
- **First-moment reset:** `ctx-m0` replicates the early benefit in `mixed` (D(50) +0.067 [+0.003, +0.130]; previously +0.058) but not in `mixed-h125` here (+0.028 [−0.037, +0.093]); added to rehearsal it does not help (−0.026 [−0.076, +0.025]). No long-term effect (reading 2).

## Derived relations (supplement, `mixed`)
Which of two marked symbols has the larger count (288 pairs of 18 symbols per seed, first evaluation rendering; chance 0.5). The recomputed setup matched the main records exactly in all ten seeds.
- Through the host's comparison head on symbol codes: teacher 0.484, host after 2,000 `ctx` updates 0.467 (at chance: the comparison head does not apply to symbol codes).
- Nine-way destination's answers compared as numbers: 0.818. Anchored destination through the comparison head: 0.797; its own nearest-canonical answers compared: 0.789.
- **Connection benefit (anchored through the comparison head minus its own answers): +0.008 [−0.011, +0.028]: none detected.** Anchored minus nine-way: −0.021 [−0.065, +0.024]. The pilot's single-seed advantage for the anchored head (0.83 against 0.74) did not replicate.
- Both destinations make the relation usable at about 0.8 while their exact recall is about 0.35–0.46: their errors are mostly near the right count, so ordinal information survives distillation.

## Reading 9: descriptive
- DER's stored scores add nothing over bounded rehearsal at either weight (`mixed` 0.484 and 0.471 against 0.471), plausibly because the stored scores are those of a teacher that is right about half the time.
- The best bounded remedy on retention in `mixed` is `stable-path-replay` (0.521, preserved), which pays in new learning; the cache destination (0.459) keeps its teacher's answers (agreement 0.79) at no host cost but needs a copy of the image path.
- `hold-M` exceeds the pre-delay level in every arm: rehearsing with the full pool keeps teaching.
- `mixed-h125`, with explicit decay, loses the most under `ctx` and gains the most from each remedy.

## Interpretation (bounded)
Arbitrary marked facts can be kept through 2,000 updates of other material by every established remedy tested, at small compute and storage costs; the new consolidation route keeps them in a protected head about as faithfully as its teacher answers, cue storage can be dropped (other symbols serve nearly as well) but random cues cannot replace it, and the stored information is the teacher's general code-to-count map rather than separate facts. Anchoring the destination in the host's count codes gave no relational benefit beyond its answers. Compression against keeping the old pathway is about 6%, because the frozen image path dominates. The `mixed` architecture's transfer cost remains the binding constraint for the combined objective; `mixed-h125` meets it here. Durability of the destination when new facts are added, routing, and a destination that does not need its own frozen path are the open questions (see the contract's internet-scale assessment).

## Computation (measured costs only)
Theoretical costs are in the contract. Measured on a GTX 1650 with six workers: 3,681 s wall (runs 3,647 s; data 28 s; rendering 3 s), 20,708 s total worker CPU, peak worker RSS 2.70 GB (parent 1.78 GB); 30 arm-seeds.
Selection and tuning: three seed-299 pilot runs, about 1,660 s wall in all (462, about 600 and 601 s; the selections come from the last). Analysis 0.4 s. Derived-relations pass 654 s wall (five workers, ten seeds).
Timing is kept in the runtime files; the results JSON holds only deterministic content.
