# Step 10: memory and core together through frozen sight (design draft)

Status: **draft, 2026-11-04. Nothing run, not frozen.** This is step 10 of the [sense integration plan](sense-integration-plan.md), drafted at the user's request after the external review of the step 8 pilot. That review recommended a small integration pilot next, using the current growth rule under its stated stream assumption. The decisions this draft needs from the user are listed under *Decisions for the user*.

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

## Design

### Facts

- **Set A and set B:** 45 symbols each from the 512-symbol pool (`experiments.arbitrary_pairings` `POOL`), disjoint, balanced over counts 1–9 (5 symbols per count), as in the [long-term memory contract](long-term-memory-v1-contract.md).
- **A write is an experience:** one training rendering of a symbol, seen with one dot collection of the symbol's count, a different collection each time from `P.DOTS['train']`. Each symbol gets 8 writes.
- **Queries:** 5 evaluation renderings per symbol, none of them written.
- **Unseen symbols:** 45 more pool symbols, never written, to measure answers given to nothing.

The count is never given to memory. It is only in the dots.

### Memory

`src.senses.KeptMemory`, kept centres with a frozen gate per centre, on DINOv2-S + TrOCR-small, at the stability pilot's settings (`sense_adapt` τ, 2,048 units per centre):
- **Keys:** the symbol rendering's joined, centred code under its centre, as now.
- **Values:** the dot collection's raw frozen blocks (768 float32), averaged when writes merge into one unit. This is new: today values are centred codes, which cannot be passed back to the core. Raw values need a `raw_values` option on `KeptMemory`, with the invariant test extended to them.
- **Centres:** opened by `shift-window` on the write stream. A and B each arrive as a run, which is the rule's stated assumption.

### Hosts

The step 8 checkpoints, seeds 300–303, at block 1, block 3, after the delay and after relearning:
- `joined-mlp-calibrated`: frozen sight with the calibrated projection, whose statistics are carried in the checkpoint;
- `existing`: the core's own image encoder, for the stale-copy reference.

Memory's contents do not depend on the host, so no core is retrained for this pilot. A "write time" only decides which host codes a copy.

### Schedule

| Order | Written after block 1 | Written after block 3 | Read at |
|---|---|---|---|
| 1 | A | B | block 3, delay, relearn (A and B); block 1 (A only) |
| 2 | B | A | the same, with roles swapped |

Hosts keep learning between writes and reads, so every read of the first set is through a host that has moved. This follows the rule that teaching B must not forget A (user, 2026-11-02): read A after B, in both orders, with a moving host.

### Arms

| Arm | Value stored | How a read answers |
|---|---|---|
| **`trace`** (the candidate) | raw frozen code of the dots | through the current calibrated projection and core |
| `copy` | the core's code of the dots, computed by the host at write time | as stored, against current candidate codes |
| `copy-existing` | as `copy`, through the `existing` host | as stored |
| `sense-only` | raw frozen code of the dots | in frozen space only: cosine against candidate collections, no core |
| `core-only` | none | the symbol rendering through the current host, no memory |
| `oracle` | none | a fresh collection of the right count through the current host: the host's own ceiling |

`existing` cannot read a frozen trace, and storing images is ruled out. So `copy-existing` is its only memory-plus-core arm. It is a reference, not a matched comparison.

### Reads

Every read is scored per query; a query whose best score is below τ abstains and counts as wrong.

| Read | Candidates (9, one per count) | What answers | Chance |
|---|---|---|---|
| **R0 retrieval** | the units | right fact's unit wins | |
| **R1 dots, sight only** | dot collections | `sense-only`: frozen cosine | 1/9 |
| **R2 dots, through the core** | dot collections | the reader's pairing head (`G.pair`), as the core's matching read | 1/9 |
| **R3 taps, across senses** | tap sequences | the pairing head; `sense-only` cannot answer | 1/9 |
| **R3′ taps → symbol** | 9 written symbols, one per count | each candidate's recalled value against the taps, pairing head | 1/9 |
| **R4 comparison** | pairs of written symbols with different counts | the comparison head (`relation_logits`, relation 2) on both recalled values | 1/2 |
| **R5 unseen** | | share of unseen symbols answered above τ | |
| **R6 core unchanged** | | the step 8 snapshot reads, with memory present and absent | |

R6 is identical by construction in this stage, because memory has no path to the core's weights. It is recorded as a check that reads have no side effects, not as evidence that memory spares the core. That evidence needs stage 2.

## What would count as working

Pilot bars, proposed for a later contract and not tested here:
- **Across senses:** `trace` R3 at block 3 within 0.10 of R0 × `oracle` R3 (retrieval and the host's own ceiling are the only losses), and well above `sense-only`, which is at chance there by construction.
- **Moving host:** `trace` holds R2–R4 from block 3 to relearn within 0.05, while `copy` may fall. A host that has moved must not make old facts unreadable.
- **Teaching B must not forget A:** A's R0–R4 loss from before B to after B is at most 0.05, in both orders.
- **Unseen:** R5 no higher than in the stability pilot at the same τ.
- **R6:** identical reads.

If `trace` R3 is near `oracle`, the core earns its place by reading memory, not by addressing it.

## Stage 2, sketched only

Memory then trains the core, the role the user set for memory on 2026-11-01, a buffer that matches experiences to train the sparse network: recalled traces are replayed as inputs during the core's training. Its read is the third check of step 10: relation reads with memory removed, against the same core trained without replay. Replaying frozen codes stores no image, but it is replay from memory rather than rehearsal from data. Whether that is allowed is a decision for the user, below. Stage 2 waits for stage 1's result.

## Decisions for the user

1. **Stage 1 first, with no training through memory.** Memory answers through the core but never changes it, so check (c) of step 10 is deferred to stage 2. Recommended: yes, because it is cheap (no core retraining) and isolates reading from training.
2. **Replay of stored frozen codes in stage 2.** Does replaying frozen sight codes, not images, fit the no-image-storage rule, or does it count as rehearsal?
3. **Values are dots.** Facts pair symbols with collections, never with forms of a taught script, since those are forgotten. Taps cannot be stored without storing an input, so they appear only as candidates.

## Prior work

These are familiar methods, not re-read for this draft. The shared file `docs/prior-work.md` has uncommitted edits by the user, so entries go there once those land.
- **Hippocampal indexing and complementary learning systems** (Teyler and DiScenna 1986; McClelland, McNaughton and O'Reilly 1995). An index stores where an experience was, and the cortex reinstates its content at recall. Here memory stores the frozen sensory code, and the current core interprets it. CLS also predicts stage 2: interleaved replay from the fast store trains the slow one without catastrophic interference.
- **Retrieval-augmented and memory-augmented networks** (kNN-LM, Khandelwal et al. 2020; Memorizing Transformers, Wu et al. 2022; RETRO, Borgeaud et al. 2022). These use frozen keys and values with a learning reader. The known failure is stale keys when the encoder changes, which frozen sight avoids. The cost is that keys cannot improve.
- **Generative and latent replay** (van de Ven et al. 2020, replay of internal representations). Latent replay of frozen features is the established form of stage 2, and it is known to work well when the feature extractor is fixed.
- **Errors compound:** a wrong retrieval passes a wrong count to the core. R0 and `oracle` separate the two losses.

## Scaling

- **Write:** one pass through each frozen model per experience, already paid by sight, plus O(k d) to code the cue on k centres.
- **Read:** O(k d + h d) to retrieve among h units, then one pass through the projection and core per recalled value, O(1) in the number of facts. R3′ needs one recall per candidate.
- **State:** per unit, the key (2 × 384 float32) and the raw value (768 float32), about 6 KB per unit. That is linear in units and independent of how long the host trains. `copy` stores the core's code instead, smaller but stale.
- **No re-encoding of stored inputs ever happens.** Host changes cost nothing to memory.

## Computation plan

To be measured and reported as in `computation-reporting.md`:
- frozen codes for the pool renderings, about 90 × 13 + 45 × 5 images: one pass each through DINOv2-S and TrOCR-small;
- memory writes and reads on CPU;
- reads through 4 checkpoints × 4 seeds × 2 hosts, with batch-one recall latency;
- state in bytes per arm.

No core is trained.
