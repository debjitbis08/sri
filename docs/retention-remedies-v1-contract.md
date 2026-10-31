# Retention remedies v1: contract (DRAFT, not frozen)

Status: **draft 2026-10-31 (revised after an external review)**, for the researchers' review. Nothing is implemented. Arms, seeds, branches, thresholds and readings change only before freezing; items marked **[decide]** need a decision.
Follows [momentum reset v1](momentum-reset-v1-results.md) (the reset delays the early drop and does not change the long decline; kept as an optional component) and the
[acceptable-cost table](momentum-reset-v1-contract.md) recorded earlier. Decided 2026-10-31: **the stronger rule for retention benefit; both the established baselines and the consolidation design are in this study.**

## Question

Can arbitrary facts selected for preservation survive 2,000 updates of other material while the sparse transfer advantage and the ability to learn new material are kept, and at what compute and storage cost?
This is the project's larger question (dense facts plus sparse transfer, at acceptable cost) stated as a test of remedies.

## Design

Common start: end of block 4, Option A, **fresh seeds 350–359**, N = 45 pairs, L = 2,000 updates, 64 training renderings per symbol, GPU; code and streams as in the forgetting-mechanism and momentum-reset runners (`experiments/retention_remedies.py`, additive).
**Marked set M**: 18 of the 45 pairs (two per count), drawn by a seeded stream identical across arms and branches; the other 27 are the unmarked set U. Marking is an explicit given signal; a learned selector is out of scope.
Every remedy sees only M. Recall is measured separately on M and U (nine candidates, chance 0.111, the existing matching protocol applied to each subset's evaluation renderings).

Arms: `mixed` (host), `sparse48` (the concurrent transfer baseline and a descriptive host), `mixed-h125` (descriptive host; its decay interacts with replay). `dense` is not added (reviewer's recommendation).

Branches (all start from the same frozen state; the same context episodes in every branch):

| Branch | What changes | Applies to |
|---|---|---|
| `ctx` | context episodes only | all arms |
| `ctx-m0` | first-moment reset ([momentum reset v1](momentum-reset-v1-results.md)) | mixed, mixed-h125 |
| `hold-M` | context + M pair items every update, drawn from the full rendering pool (upper reference; storage not bounded) | all arms |
| `rehearse-M` | bounded rehearsal: a buffer of K = 4 stored renderings per M pair; 16 buffer items added to every 10th update | all arms |
| `rehearse-M-m0` | `rehearse-M` with the first-moment reset (is the reset additive to replay?) | mixed |
| `der-M` | `rehearse-M` plus stored matching scores: each buffered rendering keeps the model's nine candidate scores (against nine stored canonical collection images) at storage time and a penalty keeps current scores near them (Dark Experience Replay form) | all arms |
| `stable-path-replay` | the **entire image-to-code path** frozen (image encoder and `model.proj['image']`, as computed by `codes_fn`; freezing the encoder alone leaves the projection moving, so stored codes would go stale); buffer of stored item codes (K = 4 per M pair) trains the rest of the reader only (REMIND form) | mixed |
| `frozen-snapshot` | reference, no training: keep the end-of-block-4 image-to-answer pathway (image-to-code path plus the pair-scoring readout) as a frozen copy and answer M queries with it; preserves the pre-delay behavior exactly, at the storage of the copy | all arms |
| `consolidate-cache` | protected destination (below) trained in a window before the 2,000 updates, cues = K = 4 stored item codes per M pair, teacher responses from the end-of-block-4 network; **cues and stored teacher outputs are deleted after distillation** and the branch is evaluated from destination weights alone | all arms |
| `consolidate-cache1` | as above with K = 1 (a one-cue cache; not an address mechanism) | mixed |
| `consolidate-random` | no stored cues: R random vectors in code space (per-dimension mean and spread measured on context-task codes, not on pair symbols); same distillation budget and discard rule | mixed |
| `consolidate-other` | no stored cues: renderings of symbols that belong to neither A nor B; same budget | mixed |

Deferred, not in this study: an interleaved (changing-teacher) consolidation branch. It confounds teacher state, timing, total distillation work and the compatibility of old cached codes with the current teacher, and it would contradict the destination-unchanged check; a later timing study should hold teacher state and distillation budget fixed first.

**Destination**: a frozen copy of the **whole end-of-block-4 image-to-code path** (image encoder and projection, the stable input interface) plus a small trained head (one hidden layer) that outputs nine count scores; trained only by distillation (KL to the teacher's softmax over the nine counts, the teacher scoring each cue against the nine canonical collection images); it receives no context-task gradients (checked bitwise).
The destination answers by arg-max of its nine scores; the endpoint is **destination accuracy** (nine-way, chance 0.111), evaluated on independent renderings, with no per-example selection between the destination and the host. **How the system would choose between the host and the destination (routing) is not tested**; destination accuracy is a different quantity from the host's matching recall, and this caveat is carried into every cross-family comparison.
**Temporary and persistent storage are reported separately.** Temporary: cues, teacher outputs, canonical candidates held during consolidation. Persistent: what remains for deployment (the destination's head and its frozen path; for replay branches, the buffer; for `frozen-snapshot`, the copy). Whether cues must be kept for later consolidation rounds is a separate requirement, not tested here.

After 2,000 updates every branch continues with 250 updates teaching the never-seen set B (as in the forgetting-mechanism relearning), for the new-learning reading.
Hyperparameters not fixed here (head width, distillation steps and learning rate, DER penalty weight, buffer cadence, R) are fixed by a seed-299 pilot before freezing; that pilot and its cost are recorded, and no scored seed is used for tuning.

## Measures

Per branch and evaluation point (0, 50, 250, 500, 1,000, 2,000): recall on M and on U (host recall; destination recall for consolidation branches); **teacher accuracy, destination accuracy and teacher–destination agreement, reported separately** (the teacher is measured both with its original matching evaluation and with the same nine canonical candidates used for distillation); transfer (a) at 2,000 as in the earlier studies; held-out context performance; accuracy on B after the 250 updates; the four training losses; destination parameters unchanged (bitwise); stored bytes by component; extra training updates and wall time over `ctx`; inference cost.

## Readings (prespecified; margin 0.05; positive differences mean better; **the stronger rule: an effect is "established meaningful" only if the paired 95% t-interval's lower bound exceeds +0.05**; noninferior if the lower bound exceeds −0.05; ten seeds)

1. **Checks.** Destination bitwise unchanged in the context phase; the frozen path bitwise unchanged; cues and teacher outputs absent when the destination is evaluated; buffer and cue contents exclude U; M and U identical across branches; `ctx` reproduces the earlier recall trajectory form. A failed check is a runner fault, reported as such.
2. **Retention benefit.** Per remedy and host arm: recall on M at 2,000 minus the same arm's `ctx` recall on M. Outcome: established meaningful (lower bound > 0.05), inconclusive, or not beneficial. **Primary contrast (named before running): `consolidate-cache` destination accuracy on M at 2,000 minus `ctx` host recall on M, `mixed` host.** Its compression comparison is the persistent storage of `consolidate-cache` against `frozen-snapshot`. Every other contrast is exploratory and no multiplicity adjustment is made; none is to be presented as confirmatory.
3. **Preservation.** Recall on M at 2,000 minus the branch's own recall at 0 (for consolidation branches, the teacher's recall at 0 on the canonical candidates). Three outcomes: **preserved** (noninferiority to the pre-delay reference established); **partial rescue** (meaningful benefit over `ctx` established, preservation not established); **inconclusive** (neither established). Failure to establish preservation is not partial rescue or no rescue.
4. **Selectivity.** Recall on M minus recall on U at 2,000 per branch (for `ctx` about 0). Restricting training to M does not guarantee a gap (generalization from M to U is possible), so it is measured; for stored-cue remedies a gap is expected by design and is not counted as a result.
5. **Cue sources** (mixed). `consolidate-cache` minus `consolidate-random` and minus `consolidate-other` on destination accuracy on M at 2,000 (does the cue source give access to useful teacher behavior? label-free cues and labeled rehearsal do not carry identical supervision, and random or other-symbol cues need not preserve M selectively); `consolidate-cache` minus `consolidate-cache1` (value of more cues).
6. **Cost gates** (per remedy and host arm), each noninferior at margin 0.05: (i) final transfer (a) against the concurrent `sparse48` `ctx` (the architectural objective); (ii) accuracy on B after 250 updates against the same arm's `ctx` (learning cost; a loss of ability to learn new pairs); (iii) held-out context performance against the same arm's `ctx`.
7. **Resources**, measured: extra persistent bytes (buffer, frozen paths, head; the full frozen inference path counts) and temporary bytes, against the provisional budget of one model's weight bytes (about 1.53 MB; 382,694 parameters at 4 bytes: image encoder 213,888, sequence encoder 142,144, model 26,662); extra training time against the provisional 25%, whose denominator is the wall time of the same arm's `ctx` 2,000-update phase and whose numerator includes consolidation and teacher queries; inference overhead (including the frozen path's forward passes) reported separately. Over-budget methods are still reported on a quality–cost frontier.
8. **Verdicts, kept separate.** *Useful remedy*: the primary contrast, or an exploratory one labeled as such, established meaningful. *Protected-store candidate meeting component-level gates*: a consolidation branch whose destination accuracy meets preservation, whose host meets gates (i)–(iii), and whose resources are within budget; this shows complementary capabilities in separate components, not a system that uses them together, and is **not** a "successful combined architecture" (that requires routing, deferred). *First-moment reset*: `rehearse-M-m0` minus `rehearse-M` (additive?) and `ctx-m0` replicated; no claim beyond those.
9. **Descriptive.** All trajectories; `hold-M` against bounded variants; M against U; losses; `mixed-h125` and `sparse48` host comparisons; consolidated recall against teacher recall.

## Prior work (written 2026-10-31; [prior-work.md](prior-work.md), section "Retention remedies and consolidation")

- Components correspond to: rehearsal and pseudorehearsal (Robins 1995) [abstract]; Dark Experience Replay [read: intro excerpt]; REMIND-style stable features plus feature replay [read: method excerpt]; distillation (Hinton et al. 2015) [abstract]; generative replay (Shin et al. 2017) [abstract]; complementary learning systems [abstract]. Cue cache = stored-feature replay; random latents = pseudorehearsal control; destination = distillation into a protected store.
- Known failure modes: pseudorehearsal and generative replay degrade when cues do not cover the old distribution (the reason for the random and other-symbol controls); replay protects only what is replayed (M only, by design); DER-style logit matching needs stored outputs; a frozen interface limits new learning (gate ii); unrouted stores are not a usable system (routing untested).
- What follows from theory without running: a destination protected from context gradients cannot lose the pairs through them; a destination trained by exact imitation on the same inputs matches the teacher; approximate distillation on other inputs can make it better or worse against ground truth (students outperforming teachers are documented, though Born Again Networks also uses labels), so teacher accuracy, destination accuracy and agreement are reported separately. Those are checks, not findings. Stored-label replay and cue-cache distillation draw on the same stored information; the added content of consolidation is the teacher's soft answers, the protection, and the discard of the cues.
- What the study adds: a comparison of these remedies on matched arbitrary pairs against retention, new learning, transfer, compute and storage in this architecture. Otherwise nothing new.
- Cheaper alternatives included: `hold-M`, `rehearse-M`, `ctx-m0` and `frozen-snapshot` (keep the old pathway; the direct storage baseline for the destination). Not included: weight-importance regularizers (EWC, Synaptic Intelligence) and Learning without Forgetting (cue sets on context inputs do not cover arbitrary pairs; no prediction made).

## Scaling (required: the small design must be a scalable one)

Notation: F facts kept (|M|), K cues per fact, d code dimension, A answer-space size (9 here), P model parameters, B frozen-copy parameters.

| Component | Storage | Extra compute | Survives scale-up? |
|---|---|---|---|
| `rehearse-M` | O(F·K·input) | O(replay items) per update | storage linear in facts: bounded only by capping F or K; the input form is the largest term |
| `der-M` | rehearse plus O(F·K·A) scores plus the A canonical candidates | A extra scores per replayed item | the scores scale with the answer space; here A = 9 is tiny, so this arm is **optimistic about DER** |
| `stable-path-replay` | O(F·K·d) | freezing the image-to-code path costs context learning | d, not input, but still linear in F; freezing blocks representation growth |
| `consolidate-cache` | O(F·K·d) cues plus a head of O(d·h + h·A) plus B | O(steps·F·K·(d·h)) per consolidation | cue storage is linear in F, so it does not solve the storage problem, only shrinks the constant (codes vs inputs); the head's capacity must grow with F; A = 9 means many facts share answers, **unlike a general key→value memory** |
| `consolidate-random` | O(R·d) | O(R) teacher queries | coverage of the F relevant regions depends on the sampling distribution relative to them; no rate is claimed, and success may fall as d grows |
| `consolidate-other` | none | O(R) | same coverage problem; cheap but information-free about marked pairs |
| frozen path copy (`frozen-snapshot`, destination interface) | B (the image-to-code path; the full pathway for the snapshot) | the frozen path's forward passes at inference | constant fraction of the model (the encoder alone is 56% here); at scale the largest cost, so a design needing it must beat simply keeping the old pathway |

What this design cannot show: that a protected store scales in capacity (F is 18), that an answer space larger than nine works (the pair task has nine values), or that cue storage can be cut below linear in facts. Stated so the results are not over-read.
No scale probe (decided 2026-10-31). A larger answer space is the next scaling axis and is not in this study.

## Internet-scale assessment (written 2026-10-31; estimates are order-of-magnitude [derived] unless tagged)

Take 10^9–10^12 retained arbitrary facts. Obstructions to this design, most severe first:
1. **Capacity of the weights.** An arbitrary fact costs bits. If a network stores on the order of 2 bits per parameter [familiar: reported for language models on synthetic facts; not re-read], 10^12 facts of about 30 bits need roughly 10^13 parameters. Compressible (structured) facts cost far less, which is why selection and structure carry the scaling argument; arbitrary memory must remain a small, selected fraction.
2. **Writes into the destination interfere with each other.** This study protects the destination from context gradients only. Adding facts to a dense destination forgets its earlier facts unless old ones are replayed (retaining cues: linear storage; self-generated replay: the coverage problem) or writes are sparse and addressed so that a write touches few parameters. A dense head, as here, is the least scalable choice; an addressed (key-value or sparse-expert) destination is the scalable form and is the natural next comparison.
3. **A frozen interface contradicts a growing model.** Stability is bought by freezing the image-to-code path; at scale the representation must keep learning, so the destination's keys drift. Remedies (re-encoding from retained keys, alignment or backward-compatible training of the representation) all need something retained or extra training.
4. **Cue source and throughput.** Cues come from the recent store, which is bounded by design, so consolidation must keep pace with the arrival rate (streaming, with temporary storage bounded by the window, as the cue-discard rule requires). Compute is linear in facts times rounds, about 10^21 FLOPs for 10^12 facts at 10^9 FLOPs per distillation, comparable to training a large model.
5. **Answer space and readout.** Nine candidates are scored exhaustively. A huge answer space needs hierarchical or approximate retrieval, or a generative readout; anchoring answers in shared concept codes (the proposed `consolidate-anchor`) helps only if the concept codes are compositional.
6. **Key separation.** Many near-identical keys need pattern separation (sparse, high-dimensional codes), which favors the sparse component; whether `mixed` provides it at scale is untested.
7. **Routing.** A system must know which queries the destination can answer; at scale this is a familiarity or retrieval problem of its own (deferred here).
8. **Selection bookkeeping.** Recurrence, use and importance signals need counters per candidate memory; fine for a bounded recent store, not for unbounded candidates.

What this study can and cannot inform: it measures distillation fidelity, cue discard, generalization to independent renderings and the storage comparison against keeping the old pathway; it does not address obstructions 1, 2, 3, 5, 6 or 7. The first result worth having for scaling is obstruction 2: whether a destination can accept a second marked set without forgetting the first, which is the durability question the reviewer identified.

## Computation (theoretical; to be replaced by measured costs)

Per arm-seed: blocks 1–4 (8,200 updates) plus 11–12 branches of 2,000 updates and a 250-update B teaching each: about 8,200 + 12 × 2,250 = 35,000 updates, about twice the momentum-reset study (4,102 s for 40 arm-seeds), so on the order of 2.5 hours for 30 arm-seeds before the added losses; a pilot gives the estimate.
Time: training O(updates · batch · P); buffer replay adds 16 items to a tenth of the updates (about 1% of items at the batch size in use, to be measured); DER adds nine score evaluations per replayed item; distillation of the head is O(steps · cues · d · h) and negligible next to training; the frozen encoder copy adds one forward pass per cue. Storage: a stored rendering is 3,136 bytes (28×28 float32), so K = 4 over 18 pairs is 72 renderings, about 226 KB (14% of one model's weight bytes), plus DER scores and nine canonical collection images (about 28 KB); a frozen image-encoder copy is 855 KB (56% of the budget) plus the head. Selection and tuning (the seed-299 pilot) are recorded as costs.

## Risks

- The consolidation arms report a different quantity (destination recall) from the host-network branches; cross-family differences are not equivalent comparisons until routing exists.
- Teacher recall at 0 is only about 0.43–0.51, so a faithful destination preserves a half-correct memory; because it is trained on the teacher's outputs, its accuracy is expected to follow the teacher's, and deviations in either direction are reported (not bounded by theory).
- The frozen image-to-code path counts as persistent storage and inference cost and pins the representation; a destination built on it cannot benefit from later representation improvements.
- `hold-M` replays far more than a bounded remedy would and is a reference only.
- Ten seeds and many contrasts: no multiplicity adjustment is specified, so single contrasts are not to be over-read; the primary reading is the per-remedy retention benefit.

## Decisions

Decided 2026-10-31: stronger rule for retention benefit; both established baselines and consolidation in one study. **Adopted in this draft from the external review, pending the researchers' confirmation:** M = 18 and K = 4; K = 1 called a one-cue cache (no address mechanism); the full frozen inference path counted in storage and inference cost; `dense` not added; interleaving deferred; `frozen-snapshot` added as the storage baseline; cues discarded after distillation; primary contrast named; the three-way preservation outcome.
Decided 2026-10-31: no scale probe. **Open [decide]:**
1. **Primary contrast** (reading 2).
2. **Anchored destination** (below).

## Connecting to what is already stored (proposed, from the researchers' note)

Transfer is expected to work best when a new memory is connected to something already stored. In this task the count concepts are already stored in the host's structure; a marked pair is an arbitrary key that needs a link to an existing concept. A destination that outputs into the host's existing count-code space, rather than a new nine-way head, makes a new fact cost one key-to-concept link instead of a new concept, which is also the more scalable form (answer space shared with the structure). Proposed branch `consolidate-anchor` (mixed): the destination head regresses to the teacher's code of the count's canonical collection (in the frozen path's code space) and answers by the nearest canonical code; same cues, budget and discard rule. Proposed additional reading: **derived relations among marked pairs**, accuracy of comparing two marked symbols' counts (which is larger) using the destination's codes through the host's existing comparison machinery, against the nine-way head and against `ctx`; a connection benefit would show as derived accuracy exceeding what the individual recall predicts. This is an added hypothesis, not yet specified at the level of the other branches (the comparison harness needs a check), and is listed for the researchers' approval.
