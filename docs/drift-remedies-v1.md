# Drift remedies v1: rehearsal, weight penalty and per-kind registration

**Status: draft contract, 2026-11-05. Not frozen; nothing run.** It follows the [moving-representation
notes](moving-representation-notes.md) and the [consistency critic](consistency-critic-v1.md) results: on the two-space base,
`eq` errors on a drifted script are in its codes, not in the `eq` reader, so `eq` is fixed by fixing drift. User decision
(2026-11-05): fix the drift first; asking, with one trigger for all operators, is step 2 on the best arm.

## Question

When a later script is taught, which remedy keeps an earlier script's codes valid for every relation, at what storage and
compute, and at what cost to learning the new script? The three families of the notes:

- **Keep evidence:** rehearsal of a few stored frozen sense codes.
- **Stop the movement:** a weight-importance penalty (EWC).
- **Isolate parameters:** freeze the object encoder after block 1 and give each later kind a small registration.

## System (as equality v2)

The `two-ctx-enbuf` arm of [equality v2](equality-relations-v2.md): frozen DINOv2-S and TrOCR-S sense codes (768-dim,
cached), object encoder E (768 → 128 → 16), context normalization within each kind, pair feature G, one shared operator
readout S with operator codes (`eq`, `more`, and `sum10` from block 3 on collections). English buffer: 4 stored codes per
count, replayed with labelled relations to collections, in every arm. 1,500 Adam steps per block, lr 1e-3, seeds 300–303;
withheld counts as v2. Thai is never taught.

**Two orders**, so that the absent script differs:
- **A:** collections and English, then Chinese, then Devanagari (v2's order).
- **B:** collections and English, then Devanagari, then Chinese.

Call the block-2 script the **earlier script** and the block-3 script the **new script**.

## Arms

| Arm | Remedy | What is added in block 3 (block 2 for registration) |
|---|---|---|
| `none` | — | Nothing: `two-ctx-enbuf` |
| `reh1`, `reh2`, `reh4` | Rehearsal | Stored frozen sense codes of the earlier script, 1, 2 or 4 per taught count, chosen at the end of block 2, replayed with labelled relations to collections, min(stored, 2) per count per step, as v2's buffers |
| `ewc` | Weight penalty | Online EWC on all parameters: after each block, a diagonal empirical Fisher from squared gradients of that block's own loss on 64 minibatches, summed over blocks, penalty λ/2 Σ F_i (θ_i − θ*_i)² against the parameters at the end of the last block |
| `ewc-reh1` | Both | `ewc` plus `reh1` |
| `reg` | Per-kind registration | E frozen after block 1. Each later kind k gets a residual adapter, c = E(x) + B_k ReLU(A_k x), with A_k 768 → 16 and B_k 16 → 16 (B_k initialized to zero), trained only in its own block and frozen after it. G, S, relation and operator codes keep training. The kind is told, as context normalization already does |

`reh4` in order A is v2's `two-ctx-enzhbuf` design. **Reproduction check:** its report must equal v2's `two-ctx-enzhbuf`
exactly; otherwise the difference is stated and `reh4` is reported against this run only.

**Not included:**
- **LwF.** It preserves outputs on present data only, so by the notes' argument it cannot protect an absent script. The notes
  give it a prediction opposite to EWC's; testing that is deferred.
- **Synaptic intelligence.** The same family as EWC, with importance computed online.
- **Generative replay** (van de Ven et al. 2020). The storage-free alternative to rehearsal; deferred.
- **Routing for `reg`.** The kind is told. Routing by frozen sense code was measured in the [sense growth
  pilot](sense-growth-v1-pilot.md), where no rule was safe on every stream; `reg` is an upper bound for registration, not a
  test of routing.

## EWC selection (before the main seeds)

λ ∈ {1, 10, 100, 1000} on seeds 310 and 311, order A only. The criterion is fixed now: the mean of the earlier script's and
the new script's `eq` identity at block 3, so retention and acquisition count equally. The selected λ is used for `ewc` and
`ewc-reh1` in both orders. Selection runs and their cost are reported.

## Reads (end of every block; test halves as v2)

**Primary, at block 3:**
1. **Earlier script:** `eq` identity (the collection count the system judges equal), and nearest-count identity.
2. **New script:** `eq` identity (acquisition).

**Secondary:**
- **Code movement of the earlier script** from block 2 to block 3: mean ‖c₃ − c₂‖ over its items, relative to the spread of
  its codes at block 2.
- **`eq` as a judgement** for the earlier script: false acceptance at |Δ| = 1 and |Δ| ≥ 2, and substitution against truth for
  `more` and `sum10` among accepted pairs, as in the [equality diagnostics](equality-diagnostics-v2.md).
- **Retention elsewhere:** English `eq` identity, and collection balanced accuracy for each operator.
- **Thai:** `eq` identity, as a never-taught control.
- **Costs:** retained bytes per arm (stored codes, Fisher and anchor, adapters), extra compute per step, and selection cost.

## Predictions

1. **[read: v2]** `reh4` order A reproduces v2's `two-ctx-enzhbuf` (Chinese `eq` identity 0.81 at block 3).
2. **[derived]** The earlier script's block-3 `eq` identity rises with codes per count. `reh1` recovers at least half of the gap
   from `none` to `reh4`.
3. **[prior work: van de Ven and Tolias 2019]** `ewc` at the selected λ lies between `none` and `reh4` for the earlier script.
   Over the selection grid, any λ that brings the earlier script within 0.05 of `reh4` costs the new script at least 0.05 of
   acquisition (stability against plasticity).
4. **[derived]** `ewc-reh1` is at least the better of `ewc` and `reh1` on the earlier script, within 0.02.
5. **[derived]** `reg`: the earlier script's codes do not move in block 3 (movement 0, because E and its adapter are frozen).
   Its `eq` identity changes only through the readers and is at least `reh4`'s. Acquisition of the new script is the open
   question: more than 0.05 below `none` means a frame fixed after block 1 lacks capacity for later kinds.
6. **[derived]** Wherever the earlier script's identity is restored, its false acceptance at |Δ| = 1 falls with it: the
   adjacent-count `eq` errors are drift errors.
7. **[derived]** Order B gives the same ranking of arms as order A.

A remedy that protects the earlier script by stopping the new one from being learned does not pass. The reads must be judged
on both primary measures together.

## Prior work

See [prior work: drift remedies](prior-work.md#drift-remedies-regularization-rehearsal-and-per-kind-adapters).
- **Taxonomy.** The arms are the three families of continual learning: replay, regularization, parameter isolation (De Lange
  et al. 2021).
- **Rehearsal.** A buffer of frozen sense codes is latent replay with frozen lower layers (Pellegrini et al. 2020), which is
  equivalent to rehearsal from the input. Under the program's rules it is rehearsal and is the plain buffer itself. The arm
  measures its cost curve, not a new method.
- **Weight penalty.** EWC (Kirkpatrick et al. 2017) in its online, summed form. The literature predicts that regularization
  fails when task identity must be inferred (van de Ven and Tolias 2019). Prediction 3 follows from that.
- **Registration.** Residual adapters (Rebuffi et al. 2017) with the domain given; the routing cost they avoid is the one the
  growth pilot measured.
- **Known failure modes:**
  - EWC constraints accumulate and stiffen the network over many tasks.
  - The Fisher here is a batch empirical estimate, not the true Fisher.
  - Rehearsal storage grows with kinds.
  - Adapters need the kind, and a frame fixed early may lack capacity.
- **Cheaper alternatives:** drift compensation was tried in the natural-image pilot and stayed below freezing. Relative
  representations cancel shared movement, not drift specific to one kind.
- **What it adds:** nothing new as method. It is the needed comparison, on this system, of the three families at measured cost,
  with `eq` read as a drift symptom.

## Scaling

| Arm | Retained state | Here | Growth with kinds K | Extra compute |
|---|---|---|---|---|
| `reh-n` | n codes × taught counts × 768 float32 per earlier script | about 3 KB per code: about 21–27 KB at n = 1, 85–110 KB at n = 4 (7–9 taught counts) | Linear in K | Replayed rows per step: 2 per count per buffered script |
| `ewc` | Anchor and summed Fisher, 2 × 105,610 floats | 845 KB | Constant | One penalty term per step; 64 extra backward passes per block |
| `reg` | Adapter per kind, 12,560 floats | 50 KB per kind | Linear in K, plus a router when the kind is not told | One small matrix product per input |

At this scale EWC stores more than rehearsal at 4 codes per count. Rehearsal passes EWC's storage after about 8 scripts at
n = 4, or about 30 at n = 1. That crossover is part of the result.

## Computation (to be measured)

From v2's runtime, a run is about 1,440 s on one CPU thread.
- **Main runs:** 7 arms × 2 orders × 4 seeds = 56 runs, about 2.8 h on 8 workers.
- **Selection:** 8 runs, about 25 min.

The runner must be crash-safe (durable per run). Runtime JSON is separate from results; measured items as the computation
policy requires.
