# Step 10: memory and core together through frozen sight (design draft)

Status: **design for the stage 1 pilot, revised 2026-11-04 after the user's decisions and an external review (*Decisions* and *Changes from review*, below). Not a contract. Stage 1 has run: [results](sense-memory-core-v1-pilot.md).** This is step 10 of the [sense integration plan](sense-integration-plan.md). The external review of the step 8 pilot recommended a small integration pilot next, using the current growth rule under its stated stream assumption.

**The claim stage 1 can support is narrow:** memory on frozen sight preserves arbitrary symbol-to-collection associations that the current core can interpret, across senses and through a changing core. It does not establish general retention of arbitrary sensory values, and it cannot establish whether training through memory preserves or improves the core.

## Question

Can memory on frozen sight and the `sparse48` core answer together what neither answers alone? Each fact is a new symbol seen with a collection of dots. Asked later, by a fresh rendering of the symbol:
- **How many taps?** This is a question across senses, which sight memory cannot answer alone.
- **Is it more than another symbol's count?**

Memory must keep its facts while the core keeps learning, and the core's own reads must not change.

## What earlier results predict

- **Sense-only memory holds arbitrary pairs; the core in the address hurts** ([stage 1 pilot](shared-core-sparse-memory-v1-stage1-pilot.md)). The core places symbols on a number line, so as an address it pulls unrelated symbols together, and no address reached across senses. That pilot pointed to reading memory through the reader's head, where the link across senses is. This design does that, and addresses by sight alone.
- **Values coded by the host go stale as the host learns; sensory traces re-encoded at read time hold** ([moving host](shared-core-sparse-memory-v1-moving-host-pilot.md), [traces](shared-core-sparse-memory-v1-traces-pilot.md) pilots). Those traces were stored images, which are now ruled out ([no image storage](sense-image-free-v1-pilot.md)).
- **With frozen sight, a trace needs no image.** The core's image path now starts from frozen codes ([step 8](sense-core-v1-pilot.md)), so a stored frozen code is exactly the input the current projection and core take. Re-encoding at read time costs one pass through the projection and core, not the frozen models.
- **Frozen values beat drifting ones** ([long-term memory pilot](long-term-memory-v1-pilot.md) diagnosis). Here values are frozen by construction.
- **Kept centres and frozen gates make old reads exactly invariant to growth** ([stability pilot](sense-stable-v1-pilot.md)). Interference from new units remains on strong views. Raw values are in the frozen models' own coordinates, so they are invariant too.
- **The core reads dots and taps throughout training.** Collections and taps are in every block's context episodes, and the comparison head is trained on collections and taps only. Scripts taught earlier are forgotten, so values must not be forms of a script.
- **Fixed values do not guarantee fixed answers.** The interpreter changes, so a value read the same way at block 3 and after relearning can give different answers. This is measured, not assumed away.

## Design

### Facts

- **Set A and set B:** 45 symbols each from the 512-symbol pool (`experiments.arbitrary_pairings` `POOL`), disjoint, balanced over counts 1–9 (5 symbols per count), as in the [long-term memory contract](long-term-memory-v1-contract.md).
- **A write is an experience:** one training rendering of a symbol, seen with one dot collection of the symbol's count, a different collection each time from `P.DOTS['train']`. Each symbol gets 8 writes, shuffled within its set.
- **Queries:** 5 evaluation renderings per symbol, none of them written.
- **Unseen symbols:** 45 more pool symbols, never written, to measure answers given to nothing.

The count is never given to memory. It is only in the dots.

### Memory

`src.senses.KeptMemory`, kept centres with a frozen gate per centre, on DINOv2-S + TrOCR-small, at the stability pilot's settings (`sense_adapt` τ, 2,048 units per centre):
- **Keys:** the symbol rendering's joined, centred code under its centre, as now.
- **Values:** the dot collection's raw frozen blocks (768 float32), averaged when writes merge into one unit. Today values are centred codes, which cannot be passed back to the core, so `KeptMemory` gets `values='raw'`, with the invariant tests extended to it (`tests/test_kept_memory.py`).
- **Averaging control.** A projection of an average need not read like the collections averaged. So each unit also keeps, as a diagnostic, the first value written to it: one actual collection's code (`trace-single`).
- **Centres:** opened by `shift-window` on runs of 1,024 passing renderings of other pool symbols, one run before each set, each arriving as a run, which is the rule's stated assumption. The gate refits from mild and strong views when a centre opens, as in the stability pilot.

### Hosts

The step 8 checkpoints, seeds 300–303, at block 1, block 3, after the delay and after relearning:
- `joined-mlp-calibrated`: frozen sight with the calibrated projection, whose statistics are carried in the checkpoint;
- `existing`: the core's own image encoder, for the stale-copy reference.

Memory's contents do not depend on the host, so no core is retrained for this pilot. A "write time" only decides which host codes a copy.

### Schedule

The first set is written after block 1, the second after block 3. Order 1 writes A first, order 2 writes B first. Memory has two states: `first` (the first set only) and `both`.

The two effects are separated:
- **Interference:** the first set read on `first` and on `both`, through the same checkpoint (block 3, the delay, relearning). Only memory changes.
- **Host change:** the first set read on `first` through block 1, block 3, the delay and relearning. Only the host changes.
- **Answer dependence on the interpreter:** for the same stored value and the same candidates, the share of answers that change between checkpoints.

This follows the rule that teaching B must not forget A (user, 2026-11-02): read A after B, in both orders, with a moving host.

### Arms

| Arm | Value stored | How a read answers |
|---|---|---|
| **`trace`** (the candidate) | raw frozen code of the dots | through the current calibrated projection and core |
| `copy` | the core's code of the dots, computed by the host at write time | as stored, against current candidate codes |
| `copy-existing` | as `copy`, through the `existing` host | as stored |
| `trace-single` | the first collection's raw frozen code, one per unit | as `trace`: the averaging control |
| `sense-only` | raw frozen code of the dots | in frozen space only: cosine against candidate collections, no core |
| `core-only` | none | the symbol rendering through the current host, no memory |
| `fresh-dots` | none | a fresh collection of the right count through the current host: a reference for the host, not a ceiling for the stored values |

**Paired diagnostics.** Retrieval and the host's reading need not be independent, and a fresh collection is not the stored average. So for `trace` and `trace-single`, each query is also read on its own fact's stored value, bypassing address selection. The fact's own unit is the one holding most of its writes. Three numbers are reported:
- end to end;
- end to end among queries whose retrieval was correct;
- on the correct stored value (`bypass`).

`existing` cannot read a frozen trace, and storing images is ruled out. So `copy-existing` is its only memory-plus-core arm. It is a reference, not a matched comparison.

### Reads

Every read is scored per query; a query whose best score is below τ abstains and counts as wrong. τ is 0.85, the stability pilot's threshold, for writes and the main reads. A secondary read abstains only below 0.70, the earlier pilots' second threshold. It was added after the seed-300 smoke run showed 38% of seen queries below 0.85, and it is reported beside the main read, never instead of it.

| Read | Candidates (9, one per count) | What answers | Chance |
|---|---|---|---|
| **R0 retrieval** | the units | right fact's unit wins | |
| **R1 dots, sight only** | dot collections | `sense-only`: frozen cosine | 1/9 |
| **R2 dots, through the core** | dot collections | the reader's pairing head (`G.pair`), as the core's matching read | 1/9 |
| **R3 taps, across senses** | tap sequences | the pairing head; `sense-only` cannot answer | 1/9 |
| **R3′ taps → symbol** | 9 written symbols, one per count | each candidate's recalled value against the taps, pairing head | 1/9 |
| **R4 comparison** | pairs of written symbols with different counts | the comparison head (`relation_logits`, relation 2) on both recalled values | 1/2 |
| **R5 unseen** | | share of unseen symbols answered above τ | |
| **R6 core unchanged** | | a hash of every host's parameters and buffers before and after all memory reads | |

R6 is a side-effect check only. Memory has no path to the core's weights, so it cannot show that memory spares the core; that needs stage 2.

## What would count as working

Pilot bars, proposed for a later contract and not tested here:
- **Across senses:** `trace` R3 on the correct stored value (`bypass`) within 0.10 of `fresh-dots`, and end to end among correctly retrieved queries within 0.05 of `bypass`. `sense-only` is at chance on R3 by construction.
- **Averaging:** `trace` within 0.05 of `trace-single` on `bypass`. Otherwise a failure could come from averaging, not integration.
- **Moving host:** `trace` holds R2–R4 from block 3 to relearn within 0.05 on `first`, while `copy` may fall. A host that has moved must not make old facts unreadable.
- **Interference:** the first set's R0–R4 loss from `first` to `both`, through the same checkpoint, is at most 0.05, in both orders.
- **Unseen:** R5 no higher than in the stability pilot at the same τ.
- **R6:** identical reads.

If `trace` R3 is near `fresh-dots`, the core earns its place by reading memory, not by addressing it.

## Stage 2: latent replay from memory (sketched only)

Memory then trains the core, the role the user set for memory on 2026-11-01: a buffer that matches experiences to train the sparse network. Replaying stored frozen codes stores no image, but **it is rehearsal**: stored feature vectors are retained experience. Stage 2 is latent replay from memory.

Its read is the third check of step 10: relation reads with memory removed. Stage 2 must specify, before it runs:
- **What a replayed item is.** A dot value alone teaches nothing about a symbol. The learning signal is the stored pair: the symbol's raw frozen cue features with its collection's raw frozen value, trained with the core's matching objective, as a pairing item. The current address keys are centred, averaged and gated, so they are not raw cue features the core's projection can take. Memory must therefore also keep raw cue features per unit (a mean, or one actual cue), and those bytes are counted.
- **Comparisons at matched resources.** No replay, and a plain buffer of frozen cue/value feature pairs with no memory mechanism, at matched retained bytes, replay exposures and update cost. That identifies what memory adds over storing features.

Stage 2 waits for stage 1's result.

## Decisions (user, 2026-11-04)

1. **Stage 1 first:** yes. R6 is a side-effect check only.
2. **Frozen-code replay in stage 2:** it fits the no-image-storage rule, and it counts as rehearsal: latent replay from memory, accounted for in bytes, exposures and update cost.
3. **Dot values:** yes, for a deliberately limited first test, with the narrow claim stated at the top. Taps could be stored too, through a frozen sequence encoder. That is an extension, not a prerequisite.

## Changes from review (2026-11-04)

- **Averaging control:** `trace-single` keeps one actual value per unit.
- **The R0 × oracle ceiling is replaced** by the paired diagnostics: bypass, and success conditional on retrieval. Fresh dots are kept as a separate reference.
- **Interference and host change are separated:** `first` against `both` through the same checkpoint, and `first` across checkpoints.
- **Answer dependence on the interpreter** is measured.
- **Stage 2:** named latent replay from memory, which is rehearsal. It needs a matched frozen-feature buffer and the stored cue/value pair with its objective.

## Prior work

These are familiar methods, not re-read for this draft. The shared file `docs/prior-work.md` has uncommitted edits by the user, so entries go there once those land.
- **Hippocampal indexing and complementary learning systems** (Teyler and DiScenna 1986; McClelland, McNaughton and O'Reilly 1995). An index stores where an experience was, and the cortex reinstates its content at recall. Here memory stores the frozen sensory code, and the current core interprets it. CLS also predicts stage 2: interleaved replay from the fast store trains the slow one without catastrophic interference.
- **Retrieval-augmented and memory-augmented networks** (kNN-LM, Khandelwal et al. 2020; Memorizing Transformers, Wu et al. 2022; RETRO, Borgeaud et al. 2022). These use frozen keys and values with a learning reader. The known failure is stale keys when the encoder changes, which frozen sight avoids. The cost is that keys cannot improve.
- **Generative and latent replay** (van de Ven et al. 2020, replay of internal representations). Latent replay of frozen features is the established form of stage 2, and it is known to work well when the feature extractor is fixed.
- **Errors compound:** a wrong retrieval passes a wrong count to the core. The bypass and conditional reads separate the two losses.

## Scaling

- **Write:** one pass through each frozen model per experience, already paid by sight, plus O(k d) to code the cue on k centres.
- **Read:** O(k d + h d) to retrieve among h units, then one pass through the projection and core per recalled value, O(1) in the number of facts. R3′ needs one recall per candidate.
- **State:** per unit, the address sum and address (2 × 768 float32), the raw value sum (768 float32) and a count, about 9 KB per unit; `trace-single` adds 3 KB per unit as a diagnostic. That is linear in units and independent of how long the host trains. `copy` stores the core's code instead, smaller but stale.
- **No re-encoding of stored inputs ever happens.** Host changes cost nothing to memory.

## Computation plan

To be measured and reported as in `computation-reporting.md`:
- frozen codes for the pool renderings, about 90 × 13 + 45 × 5 images: one pass each through DINOv2-S and TrOCR-small;
- memory writes and reads on CPU;
- reads through 4 checkpoints × 4 seeds × 2 hosts, with batch-one recall latency;
- state in bytes per arm.

No core is trained.
