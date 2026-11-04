# Plan: senses from established models, ready to integrate

Status: **plan, 2026-11-03. Nothing here has run unless marked done.** Not a contract.

## Direction

User, 2026-11-03:

- **A frozen sense is acceptable,** as long as advanced models can be used cheaply.
- **Senses are not this project's differentiator.** Where other projects have solved perception, reuse their solution.
- **Licences must be evaluated** for every model reused.
- **The architecture must accommodate other established models for senses,** so that a sense can be added or replaced without redesign.

This replaces the goal of unfreezing the senses (2026-11-02). It also lowers the priority of training our own general encoder in [the general frozen encoder study](general-frozen-encoder.md): step 3, the diet test, is dropped, and step 2 survives only as the reduced check in step 8 below.

## Where sight stands

From the [general frozen encoder study](general-frozen-encoder.md), pilot seeds only:

- Frozen DINOv2 ViT-S/14 joined with the frozen TrOCR-small stage 1 encoder, using two domain centres and a label-free learned query gate, reads 0.67–0.71 matched on mild views. That is inside the range of the copies (one frozen encoder per stretch of time), with fewer false answers.
- On strong views it falls short: 0.32–0.38 against 0.35–0.40, and 0.21–0.23 against 0.43–0.46 at τ.
- Stored keys never move. The gate weights only the query.
- The centres and the gate saw every kind in advance, and the domain centres were told the domain.

## When sight counts as ready

Sight is ready to integrate when a frozen set of established models, behind one interface, meets all of the following:

- it holds memory within sight as well as the copies do, on mild and strong views, with no labels anywhere;
- it keeps doing so as new kinds arrive;
- it gives codes the core and readers can use;
- it is within a cost budget and under licences we can use.

## Steps

Recommended order: 1 and 2, then 3 to 5 (one experiment), 7, 8, 9, 10. Step 11 is measured throughout. Step 6 is off the path to integration (user, 2026-11-03): good to have, not important now, so only lightweight experiments, run when convenient.

1. **One sense interface.** Every sense model sits behind the same wrapper, which declares:
   - its name, version and weights hash;
   - its licence;
   - its input preparation and code width;
   - its cost per input;
   - the layer read.

   Memory keeps **each model's code as a separate part of a key.** It reads with per-model query weights (the joined read, which is w·cos_a + (1−w)·cos_b, generalised to n models), with a centre per domain for each model. DINOv2, TrOCR and our small encoders are wrapped first.

   *Done when:* the wrapped version reproduces the `pilot mix` results exactly.
   **Done, 2026-11-03.** The interface is `src/senses.py`, tested in `tests/test_senses.py`. It has four parts:
   - `SenseModel`, which describes a model and its licence;
   - a registry holding `dinov2-s` and `trocr-s`;
   - `Encoder` and `Sense`, which encode inputs and count their cost;
   - `Joint`, which keeps a key part and centres per model (a missing model gives a zero part), with `join` for n models and `Gate`, the learned query gate, for n models.

   `pilot mix` now encodes through it. Run with `GE_TAG=-senses`, it reproduces the earlier report exactly: 1,056 of 1,056 reads, and the gate diagnostics in all four runs (`reports/general-encoder-v1-senses-mix.json.gz`).

   Adding a model means writing its module (`load`, `layers`, `WEIGHTS`, `SHA256`) and registering a `SenseModel`. Our own small encoders are not wrapped yet, since nothing on the plan's path uses them.

   Cost measured in that run, on a GTX 1650 shared by two workers:

   | Model | Parameters | Seconds per input | Key part |
   |---|---|---|---|
   | DINOv2-S | 22.1 M | 0.020 | 1,536 bytes |
   | TrOCR-small encoder | 21.8 M | 0.054 | 1,536 bytes |

   TrOCR costs 2.7 times as much as DINOv2, because its input is 384 × 384. The joined key is 3,072 bytes, and the gate has 770 parameters.

   Run compute: 6,551 s wall, 13,043 s aggregate CPU, peak RSS 2.6 GB.
2. **Licence evaluation.** Record the code and weights licence of each model in the wrapper. Check the model card and the repository, and note any use restrictions. This must be finished before any distribution or any decision to depend on a model.

   Current models, **not yet verified**: DINOv2 (code and weights believed to be Apache 2.0 since 2023) and TrOCR-small stage 1 (Microsoft unilm, believed MIT; the Hugging Face card should be checked).
   **Done, 2026-11-03.** I read the sources and recorded them in the registry (`src.senses.licences()`). This is not legal advice. Only the weights' licence binds us, because both networks are written here and no upstream code is run or copied.

   | Model | Weights | Code | Status | Source |
   |---|---|---|---|---|
   | DINOv2-S | Apache-2.0 | Apache-2.0 | clear | The DINOv2 README: "DINOv2 code and model weights are released under the Apache License 2.0." |
   | TrOCR-small stage 1 | none stated | MIT (unilm root LICENSE) | unclear | The Hugging Face card (revision d94c404) has no licence field or section, and the unilm TrOCR README gives no terms for checkpoints. |

   - **DINOv2.** Fine for research and for products. Redistributing the weights would carry Apache-2.0 notice duties. We fetch them outside the repository, so we don't redistribute them. Other models in the same repository differ: XRay-DINO and Cell-DINO weights are under the FAIR Noncommercial Research License, so a model from there needs its own check before it is registered.
   - **TrOCR.** Usable for this research, on the assumption that MIT covers the checkpoints. That assumption is not stated anywhere. Before any distribution or product use, either get a statement from Microsoft or replace TrOCR with a glyph model whose weights carry a stated permissive licence. TrOCR is also the more expensive model (step 1), so replacing it may pay twice.
   - **Rule from now on.** No model is registered without a licence status, the sources read and the date checked. The tests enforce this.

3. **No labels anywhere.** Run the joined arms with label-free domain centres (2-means or the stability rule) instead of centres told the domain.
4. **New kinds after setup.** Fit the gate and centres on some kinds, then let others arrive in the stream, including at least one new domain. This step needs a rule for opening a new domain centre. Earlier, label-free centres merged kinds, so that rule is the open part.
5. **Stability under adaptation.** When the gate refits or a centre splits, earlier facts must still read as before. Teaching B must not forget A, in both orders.
   **Steps 3 to 5, pilot, 2026-11-03** ([results](sense-adapt-v1-pilot.md)).
   - **Step 3 holds.** Domain centres found by 2-means read exactly like centres told the domain.
   - **Step 4 holds on final reads.** Centres opened online by a batch-shift rule, about one per kind with no labels, plus a gate refitted from stored codes, read 0.67–0.76 mild and 0.35–0.40 strong at matched false answers. That is inside the copies' range on both. They also answer fewer unseen inputs: 0.11–0.13 at τ, against 0.24–0.29 for the reference.
   - **Step 5 does not fully hold.** Single kinds lose up to 0.16 as later kinds arrive, because queries move to later, similar centres and gate refits change their weights. Next: keep each unit's centre with it, and freeze query weights per centre.

   **Drift fix, pilot, 2026-11-04** ([results](sense-stable-v1-pilot.md)). Each fact keeps the centre it was written under, for key and value. Units are partitioned by centre, every centre is searched, and each centre keeps its own frozen gate. Together these remove drift exactly: 0 of 5,400 query reads changed in score, unit, value decoding or acceptance when centres opened or the gate refitted. Mild reads no longer degrade at all. What remains is interference from new units: a different unit wins in 48 of 5,400 reads on strong views, with 6 value decodings and 2 acceptances flipped, up to −0.04 on a single kind. The invariant covers unchanged units; a write merging into an old unit was not exercised. The costs are slightly more unseen inputs answered (0.13–0.15 against 0.11–0.13 mild) and batch-one retrieval of 1.36–1.50 ms against 1.09–1.16 ms. Occupied units take 8.5–8.9 MB in both designs, and allocation is 108–126 MB against 72 MB. Kept gate codes add 3 MB per centre. External review (2026-11-04) accepted the fix and recommended moving to step 8. The invariant is now a regression test (`tests/test_kept_memory.py`, `src.senses.KeptMemory`).

   **When to open a centre, pilot, 2026-11-04** ([results](sense-growth-v1-pilot.md)). No rule tested matches one centre per kind on all four stress streams. That count is a diagnostic, not the objective: choosing a rule needs memory quality against state and read cost, which was not measured.
   - **A windowed shift rule** suits kinds that arrive in runs: no false centre in 51,200 familiar inputs. It opens centres for blends on interleaved and gradual streams.
   - **A pile-up trigger** handles gradual change, but opens extra pure centres for overlapping kinds. It also keeps every centre's forming codes, so its state grows with the number of centres.

   Occupied centres are never merged by averaging, since that would move the coordinates their units were written in. Per external review (2026-11-04), general online clustering is not a prerequisite. Step 10 uses `shift-window` under its stated assumption that kinds arrive in runs.

6. **Upgrading a sense in production (lightweight, not on the path to integration).** When a sense model is replaced or added while memory holds facts written through the old one, old keys cannot be re-encoded, because no images are stored. Candidate designs, cheapest first:
   - **run both models for a while:** the old model stays as one more part of the key, new facts are written through the new model, and the old model is retired once few facts depend on it;
   - **re-write facts when they are recalled:** when an input matches an old fact, the live input is encoded through the new model and that code is added to the fact's key, so facts in use move over and unused ones fade;
   - **learn a map from new codes to old,** fitted on inputs arriving at upgrade time (backward-compatible embeddings, Shen et al. 2020; model stitching). The [image-free pilot](sense-image-free-v1-pilot.md) found linear maps weak, but on one drifting small encoder, not two pretrained models;
   - keeping a small trace per fact for re-encoding is the usual production answer, and it is ruled out by the no-image rule.

   Lightweight test: replace DINOv2-S with DINOv2-B partway through the stream, with both models running plus re-writing on recall against a learned map. Measure old memory lost, how long the old model must keep running, and the extra compute. Use pilot seeds only, and spend little time on it.

7. **Strong views by a stronger model, not by tuning.** Close the strong-view gap by swapping in larger or different established models through step 1's interface. This checks the interface and the gap at once.
8. **Codes the core can use.** Two checks:
   - on categories never seen (step 2 of the general encoder study, reduced), answers should land on near categories;
   - on the stimuli the core and readers train on (counts and symbols), the joined code should read at least as well as the encoder they use now.
   **Step 8, pilot, 2026-11-04** ([results](sense-core-v1-pilot.md)). The core, `sparse48`, was trained and read exactly as in the limited-memory study, with frozen DINOv2-S + TrOCR-small blocks feeding a small trained MLP in place of its own image encoder. Compared with that encoder, it reads clearly better on:
   - the script being taught, Devanagari: paired 0.97 against 0.54, withheld 0.95 against 0.32;
   - recall across senses on withheld counts: taps → Devanagari 0.99 against 0.73;
   - composition of dot collections: 0.95 against 0.49.

   Comparison is equal or better. The `existing` arm reproduces the study's `sparse48` exactly, 1,540 of 1,540 measures. These are four pilot seeds only.

   The first run fitted standardization statistics on every cached image, which is transductive. A rerun fitted them on block 1's training images only, then froze them (`joined-mlp-calibrated`). It keeps every block-3 result: Devanagari 0.97 paired, 1.00 withheld; taps → Devanagari 0.99; composition 0.94. The statistics and their provenance are now saved in the projection's checkpoint.

   Forgetting of earlier scripts is unchanged in every arm. The trained projection can forget as well as the core, so this locates forgetting in projection, core and readers together. Recovery after 200 relearning updates is better, which is not yet a measure of relearning speed.

9. **Contract.** Freeze the bar from "When sight counts as ready" after the prior-work review in `literature-review.md`: multi-expert late fusion, normalisation (CSLS, whitening), test-time domain discovery and model-swap compatibility (backward-compatible embeddings). Then run fresh seeds.
10. **Integration.** Feed the joined sight code to the sparse48 core and readers, and check three things:
    - a relation learned through the new sight, against the current encoder;
    - recall across senses, with the sense-only control;
    - relation reads with memory removed, so memory does not wear down the core.

    **Design draft, 2026-11-04** ([draft](sense-memory-core-v1-design.md), nothing run). Stage 1: facts are new symbols seen with dot collections. Kept-centre memory on frozen sight stores each collection's raw frozen code as its value. At read time, the value goes through the current calibrated projection and core, which answers taps, dots and comparison: a sensory trace with no image stored. It reads through the step 8 checkpoints, so no core is retrained, with stale-copy, sense-only, core-only and oracle arms. Stage 2, latent replay from memory (rehearsal, by user decision), waits for stage 1.

    **Stage 1, pilot, 2026-11-04** ([results](sense-memory-core-v1-pilot.md), seeds 300–303, both orders). Through the core, a recalled dot value answers taps, a different sense: 0.95 on the stored value and 0.94 on correctly retrieved queries (paired on the same queries: −0.010), at or above the core's reading of fresh dots (0.91). The core alone is at chance, and sight alone cannot answer taps. Averaging stored values costs nothing.
    - Retrieval at τ 0.85 limits end-to-end reads to about 0.65, since 30% of queries abstain. At 0.70, an exploratory threshold, they reach 0.90–0.96, with 0.13–0.20 of unseen symbols answered.
    - Traces follow the core as it improves, while copies go stale (0.12 at block 3).
    - Interference is negligible (≤ 0.004).
    - The moving-host bar fails at relearning (taps 0.96 to 0.72), because the core itself reads fresh dots worse there (0.91 to 0.66). Fixed values do not give fixed answers.

    **Stage 2 design draft, 2026-11-04** ([draft](sense-memory-replay-v1-design.md), nothing run). Latent replay from memory, which is rehearsal, during the core's training. The primary outcomes are read with memory removed: acquisition of the new symbol-to-count associations, and retention of the core's cross-sense abilities. Arms: no replay, memory replay, and a reservoir buffer of feature pairs at memory's bytes and the same replay budget. Replayed pairs are labelled by the core's own reading of the stored collection, with a true-count diagnostic arm.

    **Stage 2, pilot, 2026-11-04** ([results](sense-memory-replay-v1-pilot.md), seeds 300–303). With memory removed, replay teaches the core the new associations: taps from a new symbol 0.72–0.79, against 0.11 without replay. About a quarter is lost again at relearning, even with true counts. Memory replay is no better than a reservoir buffer at the same bytes on taps and dots, and 0.07–0.19 worse on comparison between new symbols; its uniform-over-units exposures are uneven. Every replay arm costs withheld script transfers, by up to −0.28 on Chinese at relearning (memory replay the most), while collections, taps and their comparison are untouched. `no-replay` reproduces step 8 exactly. Storage binds at 32 writes (0.25–0.30 units per write).

    **Judged as the combined system** (user, 2026-11-04: memory and core together is the goal; reads with memory removed are diagnostics; consolidation and sleep are future work). With memory present, replay does not help and partly harms. Symbol → taps is 0.04–0.20 lower at block 3 and the delay, and no better at relearning. Taps → symbol is unchanged. The loss is in interpretation: replay lowers the core's own collections → taps reading (−0.06 to −0.21), while retrieval is unchanged. Collections → taps falls at relearning in every arm (0.66–0.69); taps → collections stays at 0.92–0.98. Memory gives no support to withheld script tasks. **Decision:** keep stage 1's retrieval without replay as the working design, and record replay's acquisition benefit and its interference for future consolidation work. The uniform-replay pilot is closed.

    **Collections → taps at relearning** ([diagnosis](collections-taps-diagnosis-v1.md), existing checkpoints, no training). The decline is confirmed under identical evaluation, fresh 0.84–0.94 to 0.49–0.81 and stored values 0.88–0.99 to 0.52–0.85, while taps → collections holds at 0.92–0.99.
    - Collections → taps is never trained: matching always calls the order-sensitive `same` comparator as g(form or taps, collection).
    - Relearning English moves the shared image projection most (0.26–0.40) and contracts the space between counts. Only the trained order stays calibrated; scoring the same pairs in that order recovers most of the loss.
    - Failures sit at 9 dots, plus a hub at 9 in one seed.
    - [Same-relation diagnosis](same-relation-diagnosis-v1.md): the learned `same` comparator does not enforce symmetry. The reverse validity that emerged is fragile and associated with the learned alignment of collections and taps; the violation grows with code distance. How it arose and why it broke are not yet established.
    - Relearning shrinks the code space threefold, collections more than taps, and rotates the senses' count lines apart. The trained use, taps → collection, survives at 0.94–0.98.
    - The evaluated read fails through per-candidate offsets (hubs; centring lifts it from 0.49–0.82 to 0.72–0.90) and a smaller loss of pair discrimination (AUROC 0.97–1.00 to 0.90–0.97).
    - [Relearning branches](relearning-branches-v1.md), from one captured pre-relearning state (reproductions exact). Reintroducing the form-relearning material (English plus rehearsal of earlier scripts) triggers the decline in these branches; continuing the delay's material keeps every property, including memory-present reads.
    - It is an abrupt disruption in the first 20 updates, as forgotten forms give confidently wrong losses (matching about 12, sequence 60–94). The code space shrinks threefold and even the trained taps → collections falls to 0.29–0.60.
    - The trained direction recovers by 50 updates; collections → taps recovers partly, to 0.67 by 200 and still rising.
    - Freezing the shared image projection does not prevent it (0.61), and slows English reacquisition.
    - [Relearning dynamics](relearning-dynamics-v1.md), measurement first. The first relearning updates' gradients are two orders of magnitude above the delay's (total norm 229–2,027, clipped by 0.01). Before clipping, the form sequence-prediction loss dominates the shared projections, and they move 4–7 times more than from delay material.
    - Removing the form sequence-prediction rows keeps the code space and the senses' alignment, with collections → taps at 0.86 and stored values at 0.90 at 200, and English still relearns (0.54). Removing form matching does not prevent the disruption, and slows English.
    - Over 1,000 updates the reverse read peaks around 300 (0.78) and settles at 0.69, below its earlier 0.89, unrelated to English reacquisition per seed.
    - Adam's history is not tested. Explicit reverse training is deferred.
    - [Mechanism](relearning-mechanism-v1.md): forgotten forms' sequence predictions land far from their target collections, so the squared-distance loss pulls each target straight at its prediction. The pull points inward, and through the shared `proj['image']` it sums to a largely uniform contraction of the image codes (contraction locally lowers the loss, shrink derivative +169; radial cosine 0.23–0.33, so it is not the steepest direction). Collections contract to 0.13–0.35 of their scale while taps, never candidates there, keep 0.79–0.96, so the senses separate.
    - Stopping only that candidate-side gradient, the dominant identified route of the early collapse, keeps most of the code space and lifts the reverse read (0.69 against 0.42 at 50 updates, still below 0.89) with English unchanged. Stopping the input side, or freezing the temperature, does not.
    - Lesson: a correct objective can still destroy a representation other skills use, and memory cannot compensate when its interpreter moves.
    - [Remedies plan](relearning-remedies-plan.md) (draft for review, nothing run): batch-relative distance logits, a target-side stop-gradient, bounded row influence, a per-sense scale anchor, and a memory-anchored constraint, each with its prior work. A screen from the captured states, then the full curriculum for the best.
    - [Stage A](relearning-remedies-a-v1.md) (four sequence-loss arms). No arm passes. Stopping the gradient into sequence prediction's candidates halves the interpreter's worst transient (−0.29 against −0.67) and keeps the geometry, but weakens multi-step composition (−0.11 against +0.06). Batch-relative logits expand the codes instead and degrade. A tenth of the weight only slows the collapse. Stage B not started. Proposed next, for review: a weaker or slower candidate path.
    - [Constrained update](relearning-constrained-audit-v1.md) (from the loss analysis): keeping batch collection spread and collection–taps mismatch fixed to first order holds those statistics but not the interpreter. Reads fall as with unchanged, held-out spacing still collapses by 200, and the deadline is worse than unchanged. Closed; remedies must be judged on the comparator's and predictor's reads, not geometry alone.
11. **Cost and scaling.** Measure seconds per input and bytes per key for each model, and their sum. Check them against `cpu-efficiency-design-constraint.md`.

    Scaling: n models cost n forward passes per input. Keys grow by the sum of code widths. Centres grow with domains × models, and the gate with models. All of this is linear and independent of time, unlike copies.

## What this does not decide

- Which models to use in the end. Step 1 makes that a choice, not a redesign.
- Senses other than sight. The same interface is meant to hold them, which is untested.
