# Retention remedies v1: contract (DRAFT, not frozen)

Status: **draft 2026-10-31**, for the researchers' review. Nothing is implemented. Arms, seeds, branches, thresholds and readings change only before freezing; items marked **[decide]** need a decision.
Follows [momentum reset v1](momentum-reset-v1-results.md) (the reset delays the early drop and does not change the long decline; kept as an optional component) and the
[acceptable-cost table](momentum-reset-v1-contract.md) recorded earlier. Decided 2026-10-31: **the stronger rule for retention benefit; both the established baselines and the consolidation design are in this study.**

## Question

Can arbitrary facts selected for preservation survive 2,000 updates of other material while the sparse transfer advantage and the ability to learn new material are kept, and at what compute and storage cost?
This is the project's larger question (dense facts plus sparse transfer, at acceptable cost) stated as a test of remedies.

## Design

Common start: end of block 4, Option A, **fresh seeds 350–359**, N = 45 pairs, L = 2,000 updates, 64 training renderings per symbol, GPU; code and streams as in the forgetting-mechanism and momentum-reset runners (`experiments/retention_remedies.py`, additive).
**Marked set M**: 18 of the 45 pairs (two per count), drawn by a seeded stream identical across arms and branches; the other 27 are the unmarked set U. Marking is an explicit given signal; a learned selector is out of scope.
Every remedy sees only M. Recall is measured separately on M and U (nine candidates, chance 0.111, the existing matching protocol applied to each subset's evaluation renderings).

Arms: `mixed` (host), `sparse48` (the concurrent transfer baseline and a descriptive host), `mixed-h125` (descriptive host; its decay interacts with replay). [decide: add `dense`?]

Branches (all start from the same frozen state; the same context episodes in every branch):

| Branch | What changes | Applies to |
|---|---|---|
| `ctx` | context episodes only | all arms |
| `ctx-m0` | first-moment reset ([momentum reset v1](momentum-reset-v1-results.md)) | mixed, mixed-h125 |
| `hold-M` | context + M pair items every update, drawn from the full rendering pool (upper reference; storage not bounded) | all arms |
| `rehearse-M` | bounded rehearsal: a buffer of K = 4 stored renderings per M pair; 16 buffer items added to every 10th update | all arms |
| `rehearse-M-m0` | `rehearse-M` with the first-moment reset (is the reset additive to replay?) | mixed |
| `der-M` | `rehearse-M` plus stored matching scores: each buffered rendering keeps the model's nine candidate scores (against nine stored canonical collection images) at storage time and a penalty keeps current scores near them (Dark Experience Replay form) | all arms |
| `stable-enc-replay` | both encoders frozen; buffer of stored item codes (K = 4 per M pair) trains the reader only (REMIND form: stable feature interface + replay of stored features) | mixed |
| `consolidate-cache` | protected destination (below) trained in a window before the 2,000 updates, cues = K = 4 stored item codes per M pair, teacher responses from the end-of-block-4 network | all arms |
| `consolidate-cache1` | as above with K = 1 (smallest cache; stands for "retained addresses", see Decisions) | mixed |
| `consolidate-random` | no stored cues: R random vectors in code space (per-dimension mean and spread measured on context-task codes, not on pair symbols) | mixed |
| `consolidate-other` | no stored cues: renderings of symbols that belong to neither A nor B | mixed |
| `consolidate-interleaved` | `consolidate-cache`, but 25 distillation steps every 50 context updates with the current (decaying) network as teacher, no initial window | mixed |

**Destination**: a frozen copy of the end-of-block-4 image encoder (the stable input interface) plus a small trained head (one hidden layer) that outputs nine count scores; trained only by distillation (KL to the teacher's softmax over the nine counts, the teacher scoring each cue against the nine canonical collection images); it receives no context-task gradients (checked bitwise).
The destination answers by arg-max of its nine scores; it is compared in nine-way accuracy, chance 0.111. **How the system would choose between the main network and the destination (routing) is not tested**; the consolidation branches report the destination's recall as "the best store available", which is a different quantity from the host network's matching recall. This caveat is carried into every cross-family comparison.

After 2,000 updates every branch continues with 250 updates teaching the never-seen set B (as in the forgetting-mechanism relearning), for the new-learning reading.
Hyperparameters not fixed here (head width, distillation steps and learning rate, DER penalty weight, buffer cadence, R) are fixed by a seed-299 pilot before freezing; that pilot and its cost are recorded, and no scored seed is used for tuning.

## Measures

Per branch and evaluation point (0, 50, 250, 500, 1,000, 2,000): recall on M and on U (host recall; destination recall for consolidation branches); destination fidelity to the teacher (agreement with the teacher's arg-max on M evaluation renderings); transfer (a) at 2,000 as in the earlier studies; held-out context performance; accuracy on B after the 250 updates; the four training losses; destination parameters unchanged (bitwise); stored bytes by component; extra training updates and wall time over `ctx`; inference cost.

## Readings (prespecified; margin 0.05; positive differences mean better; **the stronger rule: an effect is "established meaningful" only if the paired 95% t-interval's lower bound exceeds +0.05**; noninferior if the lower bound exceeds −0.05; ten seeds)

1. **Checks.** Destination bitwise unchanged in the context phase; buffer and cue contents exclude U; M and U identical across branches; `ctx` reproduces the earlier recall trajectory form. A failed check is a runner fault, reported as such.
2. **Retention benefit (primary).** Per remedy and host arm: recall on M at 2,000 minus the same arm's `ctx` recall on M. Outcome: established meaningful (lower bound > 0.05), inconclusive, or not beneficial.
3. **Preservation.** Recall on M at 2,000 minus the branch's own recall at 0; "preserved" if noninferior, otherwise "partial rescue" or "none". For consolidation branches the reference is the teacher's recall at 0 (the destination cannot exceed its teacher).
4. **Selectivity.** Recall on M minus recall on U at 2,000 per branch (for `ctx` this is about 0 by construction). Selectivity is by design for stored-cue remedies, so it is reported, not counted as a result.
5. **Cue sources** (mixed). `consolidate-cache` minus `consolidate-random` and minus `consolidate-other` on destination recall on M at 2,000 (is the information in the cues?); `consolidate-cache` minus `consolidate-cache1` (value of more cues); `consolidate-cache` minus `consolidate-interleaved` (does a decaying teacher matter?).
6. **Cost gates** (per remedy and host arm), each noninferior at margin 0.05: (i) final transfer (a) against the concurrent `sparse48` `ctx` (the architectural objective); (ii) accuracy on B after 250 updates against the same arm's `ctx` (learning cost; a loss of ability to learn new pairs); (iii) held-out context performance against the same arm's `ctx`.
7. **Resources**, measured: extra persistent bytes (buffer, cue cache, canonical collections, frozen encoder copy, head) against the provisional budget of one model's weight bytes (about 1.53 MB; 382,694 parameters at 4 bytes: image encoder 213,888, sequence encoder 142,144, model 26,662); extra training time against the provisional 25%; inference overhead reported separately. Over-budget methods are still reported on a quality–cost frontier.
8. **Verdicts, kept separate.** *Useful remedy*: reading 2 established meaningful for some remedy. *Successful combined architecture*: also gates (i) and (ii) and (iii) noninferior and resources within budget. *First-moment reset*: `rehearse-M-m0` minus `rehearse-M` (additive?) and `ctx-m0` replicated; no claim beyond those.
9. **Descriptive.** All trajectories; `hold-M` against bounded variants; M against U; losses; `mixed-h125` and `sparse48` host comparisons; consolidated recall against teacher recall.

## Prior work (written 2026-10-31; [prior-work.md](prior-work.md), section "Retention remedies and consolidation")

- Components correspond to: rehearsal and pseudorehearsal (Robins 1995) [abstract]; Dark Experience Replay [read: intro excerpt]; REMIND-style stable features plus feature replay [read: method excerpt]; distillation (Hinton et al. 2015) [abstract]; generative replay (Shin et al. 2017) [abstract]; complementary learning systems [abstract]. Cue cache = stored-feature replay; random latents = pseudorehearsal control; destination = distillation into a protected store.
- Known failure modes: pseudorehearsal and generative replay degrade when cues do not cover the old distribution (the reason for the random and other-symbol controls); replay protects only what is replayed (M only, by design); DER-style logit matching needs stored outputs; a frozen interface limits new learning (gate ii); unrouted stores are not a usable system (routing untested).
- What follows from theory without running: a destination protected from context gradients cannot lose the pairs through them; a destination trained on teacher outputs cannot exceed the teacher. Those are checks, not findings. Stored-label replay and cue-cache distillation carry the same stored information; the added content of consolidation is the teacher's soft answers and the protection.
- What the study adds: a comparison of these remedies on matched arbitrary pairs against retention, new learning, transfer, compute and storage in this architecture. Otherwise nothing new.
- Cheaper alternatives included: `hold-M`, `rehearse-M` and `ctx-m0`. Not included: weight-importance regularizers (EWC, Synaptic Intelligence) and Learning without Forgetting (cue sets on context inputs do not cover arbitrary pairs; no prediction made).

## Computation (theoretical; to be replaced by measured costs)

Per arm-seed: blocks 1–4 (8,200 updates) plus 11–12 branches of 2,000 updates and a 250-update B teaching each: about 8,200 + 12 × 2,250 = 35,000 updates, about twice the momentum-reset study (4,102 s for 40 arm-seeds), so on the order of 2.5 hours for 30 arm-seeds before the added losses; a pilot gives the estimate.
Time: training O(updates · batch · P); buffer replay adds 16 items to a tenth of the updates (about 1% of items at the batch size in use, to be measured); DER adds nine score evaluations per replayed item; distillation of the head is O(steps · cues · d · h) and negligible next to training; the frozen encoder copy adds one forward pass per cue. Storage: a stored rendering is 3,136 bytes (28×28 float32), so K = 4 over 18 pairs is 72 renderings, about 226 KB (14% of one model's weight bytes), plus DER scores and nine canonical collection images (about 28 KB); a frozen image-encoder copy is 855 KB (56% of the budget) plus the head. Selection and tuning (the seed-299 pilot) are recorded as costs.

## Risks

- The consolidation arms report a different quantity (destination recall) from the host-network branches; cross-family differences are not equivalent comparisons until routing exists.
- Teacher recall at 0 is only about 0.43–0.51, so a faithful destination preserves a half-correct memory; the destination's ceiling is the teacher's.
- A frozen encoder copy counts as storage but also pins the representation; a destination built on it cannot benefit from later representation improvements.
- `hold-M` replays far more than a bounded remedy would and is a reference only.
- Ten seeds and many contrasts: no multiplicity adjustment is specified, so single contrasts are not to be over-read; the primary reading is the per-remedy retention benefit.

## Decisions

Decided 2026-10-31: stronger rule for retention benefit; both established baselines and consolidation in one study. **Open [decide]**:
1. **Retained addresses.** The previous description listed three cue sources (cache, retained addresses, random latents). Retained addresses need a map from a key to the activation that reactivates the recent network, which this architecture does not have; I propose testing them only as the smallest cache (K = 1) and not building the map. Alternatively, name an address mechanism to build.
2. **Arms.** `mixed`, `sparse48`, `mixed-h125`; adding `dense` is cheap in code but adds time.
3. **Marked set size** (18 of 45) and **K = 4**.
4. **Provisional budget** (25% extra training time, one model's weight bytes of extra storage) and whether the frozen encoder copy counts in full.
5. **Interleaved timing** (`consolidate-interleaved`) kept in this study or deferred.
