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
- it keeps doing so as new kinds arrive and as models are added or replaced;
- it gives codes the core and readers can use;
- it is within a cost budget and under licences we can use.

## Steps

Recommended order: 1 and 2, then 3 to 6 (one experiment), 7, 8, 9, 10. Step 11 is measured throughout.

1. **One sense interface.** Every sense model sits behind the same wrapper, which declares:
   - its name, version and weights hash;
   - its licence;
   - its input preparation and code width;
   - its cost per input;
   - the layer read.

   Memory keeps **each model's code as a separate part of a key.** It reads with per-model query weights (the joined read, which is w·cos_a + (1−w)·cos_b, generalised to n models), with a centre per domain for each model. DINOv2, TrOCR and our small encoders are wrapped first.

   *Done when:* the wrapped version reproduces the `pilot mix` results exactly.
2. **Licence evaluation.** Record the code and weights licence of each model in the wrapper. Check the model card and the repository, and note any use restrictions. This must be finished before any distribution or any decision to depend on a model.

   Current models, **not yet verified**: DINOv2 (code and weights believed to be Apache 2.0 since 2023) and TrOCR-small stage 1 (Microsoft unilm, believed MIT; the Hugging Face card should be checked).
3. **No labels anywhere.** Run the joined arms with label-free domain centres (2-means or the stability rule) instead of centres told the domain.
4. **New kinds after setup.** Fit the gate and centres on some kinds, then let others arrive in the stream, including at least one new domain. This step needs a rule for opening a new domain centre. Earlier, label-free centres merged kinds, so that rule is the open part.
5. **Stability under adaptation.** When the gate refits or a centre splits, earlier facts must still read as before. Teaching B must not forget A, in both orders.
6. **Adding or replacing a model mid-stream.** This is the test of flexibility. Keys written before a model existed have no part for it, and keys written through a replaced model hold codes that the new model cannot match. No images are stored, so old keys cannot be re-encoded. Candidate designs:
   - weight a missing part at zero for old keys;
   - keep a retired model running for old keys until they fade (copies again, but only on replacement);
   - learn a map between the old and new codes. The earlier [image-free pilot](sense-image-free-v1-pilot.md) found linear maps weak.

   Test by adding a third model (for example a larger DINOv2 or a CLIP-style image encoder) mid-stream, and by replacing one.
7. **Strong views by a stronger model, not by tuning.** Close the strong-view gap by swapping in larger or different established models through step 1's interface. This checks the interface and the gap at once.
8. **Codes the core can use.** Two checks:
   - on categories never seen (step 2 of the general encoder study, reduced), answers should land on near categories;
   - on the stimuli the core and readers train on (counts and symbols), the joined code should read at least as well as the encoder they use now.
9. **Contract.** Freeze the bar from "When sight counts as ready" after the prior-work review in `literature-review.md`: multi-expert late fusion, normalisation (CSLS, whitening), test-time domain discovery and model-swap compatibility (backward-compatible embeddings). Then run fresh seeds.
10. **Integration.** Feed the joined sight code to the sparse48 core and readers, and check three things:
    - a relation learned through the new sight, against the current encoder;
    - recall across senses, with the sense-only control;
    - relation reads with memory removed, so memory does not wear down the core.
11. **Cost and scaling.** Measure seconds per input and bytes per key for each model, and their sum. Check them against `cpu-efficiency-design-constraint.md`.

    Scaling: n models cost n forward passes per input. Keys grow by the sum of code widths. Centres grow with domains × models, and the gate with models. All of this is linear and independent of time, unlike copies.

## What this does not decide

- Which models to use in the end. Step 1 makes that a choice, not a redesign.
- Senses other than sight. The same interface is meant to hold them, which is untested.
