# Rule critic binding v1

**Contract frozen 2026-11-07 before smoke or main training.** User authorized drafting and running the next
experiment. Follow-up to [rule critic diagnostic v1](rule-critic-diagnostic-v1.md). This is a component diagnostic,
not a replacement reasoning architecture or a solution to answer-only learning.

## Question and choice

Can explicit relation comparisons and variable-name normalization make the local critic follow a told rule on
relations absent from rule training? Fix binding before revisiting witness selection. Keep the readers frozen,
the previous diagnostic's ancestor-style fragment, splits, direct program supervision and training budget.
No walker training, graph construction, MIL or answer evaluation in this study: supplied programs isolate binding.

## Prior work reviewed before freezing

Sources inspected 2026-11-07: abstracts of [Neural Theorem Provers](https://arxiv.org/abs/1705.11040),
[Pointer Networks](https://arxiv.org/abs/1506.03134), [relative representations](https://arxiv.org/abs/2209.15430),
and [Deep Sets](https://arxiv.org/abs/1703.06114). Shared findings: [prior work](prior-work.md).

- **[abstract]** NTP uses a kernel on symbol vectors for soft unification inside backward chaining. Here we reuse
  the smaller idea of symbol comparison, using cosine features; this is not an NTP implementation.
- **[abstract; analogy]** Relative representations use similarities to anchors. Rule atoms act as local comparison
  targets here; their published model-stitching results do not establish rule-following.
- **[abstract]** Deep Sets supplies the permutation-invariant pooling pattern. We pool atoms within rules and rules
  within a set. Pointer Networks selects input positions; our scorer does not, so it is not a pointer network.
- **[derived]** Alpha-normalization (canonical names for bound variables) preserves the formula. In this ordered-body
  fragment, assigning head arguments first and then unseen body variables is invariant to consistent variable renaming.
  This is syntax preprocessing, not rule execution. Body reordering is not tested or guaranteed.
- **[derived]** Once START→R, R→R and R→END pass, longer R chains pass automatically. Length transfer on this fragment
  is structurally expected, not evidence for general logical reasoning.
- **Known failures:** similar but different relation codes can collide; direct labels are extra supervision;
  pooling may lose structure in richer rules; fixed single-derived-symbol roles do not solve arbitrary symbol binding.
- **Cheaper alternative:** exact symbolic relation equality and the fragment's three allowed transitions are an
  evaluation reference (100% valid acceptance, zero invalid acceptance), not a deployed component. An unchanged
  neural baseline is rerun at the same sample budget. There is no novelty claim.

## Arms

1. `original`: unchanged rule encoder and 64-hidden MLP critic from the diagnostic.
2. `canonical`: same network, after renaming variables independently in each rule by first occurrence in the head,
   then ordered body. Tests whether variable names account for the failure.
3. `matching`: same local-check interface and 64-hidden MLP building blocks, with relative inputs instead of raw
   symbol identities. For every atom, provide cosine similarity of its frozen relation code to the previous and
   current step's frozen codes; atom role (head/body1/body2), canonical subject/object positions, asked-relation
   flag, and base-relation flag. Similarities for derived atoms and START/END are zero. Derived identity is represented
   only by its asked flag: this is sufficient only for the present one-derived-relation fragment. An atom MLP
   (13→64→64) is summed per rule, a rule MLP (64→64) is summed over rules, and an output MLP (68→64→1) receives
   that context plus previous/current direction and START/END flags. Sigmoid gives the local verdict.

The matching arm changes the representation and pooling, not just adds one feature; a win establishes that this
small component repair works, not which individual feature caused it. No learned symbolic equality threshold,
compiled transition table, `R` metadata, orientation metadata or labels are supplied to the scorer.

## Data, training and reads

Reuse diagnostic v1 exactly: seeds 450–453, split seeds 2029/2030, 300 rule sets, one R and one H, forward/reversed
and left/right recursion. Hold out 20% of eligible relations including parent-of and just-bigger-than; hold out one
orientation/side combination per training relation. Report exact split IDs. Frozen relation codes have dimension 16.

Direct supervision only: per sampled rule set, R and RR as positives, one random and one hard negative, lengths 1–2.
Sixteen rule sets per step; 3,000 Adam steps, lr 0.003; BCE on the product of local scores. Identical data RNG streams
across arms. Checkpoints 500/1000/2000/3000; no checkpoint selection or tuning. The final checkpoint is primary.
Record actual distinct (rule-set, program) training examples and exposures, including generated negatives.

Primary reads reuse diagnostic v1: valid acceptance at each length 1–12; each hard-negative category separately
(direction reversal, relation replacement, splice, wrong orientation), random negatives; error location; fresh-symbol,
variable-name and rule-order invariance. Held-out relations and instances separately; seed values and mean/range.
Add nearest-code relation substitution negatives (most cosine-similar different base relation) at lengths 2–12,
all substitution positions. This tests a known failure of soft matching. All reported acceptance uses every local
score >0.5. Program labels are offline training/evaluation data only.

**Success:** mean held-out-relation valid acceptance ≥0.90 at every length 3–12; each hard-negative category and
nearest-code category ≤0.10; each invariance change ≤0.02. Report worst seed as well; do not hide a failed seed.
Symbolic reference must pass the full evaluation, including transformations and hard negatives.

## Predictions and decisions

- **Hypothesis:** canonicalization removes variable-renaming sensitivity but alone does not solve held-out relations.
- **Hypothesis:** matching meets the success gate. High nearest-code errors would instead expose code discrimination
  as a remaining limit.
- Matching success justifies a later, separately contracted answer-supervision experiment using this component.
- Matching failure calls for inspecting matching margins/role decoding. No expansion of the walker follows from it.
- Better long-length acceptance alone is insufficient; negative rejection and renaming invariance must also pass.

## Computation contract

Let A be total atoms (5 here), J rules (2), d=16, h=64, S directed hop types, U distinct local pairs per batch,
B=16 rule sets, K≤2 training length, P=4 programs per rule set, T=3000 updates.
Oracle fact matrices used only to reproduce the split cost O(RN²) storage/construction; no neural graph checks.
Canonicalization O(A), code comparison preparation O(SAd); matching scoring O(U(Ah²+Jh²+h²)), with O(UAh)
temporary activation floats. Training O(T B [SAd + PK(Ah²+Jh²+h²)]); pair deduplication can reduce work.
Full evaluation table O(S²Ah²) time, O(S²+chunk·Ah) workspace; program checks O(K).
Original/canonical scoring uses cached rule tokens, O(Ah²+U h²) per rule set, plus attention O(UJh).
No candidate search, beam selection, optimizer convergence solver or covariance state.

Matching has 13→64→64, 64→64 and 68→64→1 layers, with float32 parameters; report exact counts and bytes.
Adam moments require 2× parameter bytes plus scalar step counters; gradients another 1×. Frozen code buffers,
reader checkpoint, retained rule sets, generated training-example audit and temporary memory are counted separately.
Only critic checkpoints/results are newly retained on disk; data are regenerated from seeds. Report checkpoint bytes.

Estimated 1–3 CPU minutes training per fit, evaluation may dominate matching tables; 12 fits, 3 workers,
approximately 10–25 minutes wall, peak <1.5 GiB per worker. These are estimates, not speed claims.
Smoke: all arms, seed 450, 30 rule sets/100 steps, 50/100 checkpoints; separate files and explicit cost.
No tuning: smoke checks mechanics only; changes after observing quality require an amendment and fresh main run.

Runtime JSON: source/dependency/contract hashes, dirty revision, command, checkpoint provenance, hardware/software,
BLAS/thread configuration; per-phase CPU and wall time, total worker CPU, process peak RSS (not whole-job peak),
training examples/exposures/local checks, evaluation calls, parameter/buffer/checkpoint bytes. Separate deterministic
scientific JSON. Count smoke, verification, main and report generation costs; unavailable quantities are null with reasons.
Report quality versus measured cost at equal data, with no inference-latency or scaling claims. No historical files changed.

## Commands

    EV_SMOKE=1 EV_WORKERS=3 .venv/bin/python -m experiments.rule_critic_binding
    EV_WORKERS=3 .venv/bin/python -m experiments.rule_critic_binding

Results will be recorded separately in `docs/rule-critic-binding-v1-results.md` to preserve this contract's hash.
