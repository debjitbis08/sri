# Drift remedies v1: rehearsal, weight penalty and per-kind registration

**Status: contract frozen 2026-11-05, before the run; run and results 2026-11-05 (below).** It follows the [moving-representation
notes](moving-representation-notes.md) and the [consistency critic](consistency-critic-v1.md) results: on the two-space base,
`eq` errors on a drifted script are in its codes, not in the `eq` reader, so `eq` is fixed by fixing drift. User decision
(2026-11-05): fix the drift first; asking, with one trigger for all operators, is step 2 on the best arm.

## Question

When a later script is taught, which remedy keeps an earlier script's codes valid for every relation, at what storage and
compute, and at what cost to learning the new script? The three families of the notes:

- **Keep evidence:** rehearsal of a few stored frozen sense codes.
- **Stop the movement:** a weight-importance penalty (EWC).
- **Isolate parameters:** freeze the object encoder after block 1 and give each later kind a small registration. **Comparison
  only:** it is told the script, which is acceptable for now and not long term (user decision, 2026-11-05). It shows what
  isolation would buy with perfect routing. It is not a candidate remedy, and no conclusion about adopting it is drawn.

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
| `ewc` | Weight penalty | Online EWC on all parameters: after each block, a diagonal empirical Fisher from squared gradients of that block's own loss on 64 minibatches, summed over blocks, each block's Fisher normalized to mean 1 over parameters, penalty λ/2 Σ F_i (θ_i − θ*_i)² against the parameters at the end of the last block |
| `ewc-reh1` | Both | `ewc` plus `reh1` |
| `reg` | Per-kind registration | E frozen after block 1. Each later kind k gets a residual adapter, c = E(x) + B_k ReLU(A_k x), with A_k 768 → 16 and B_k 16 → 16 (B_k initialized to zero), trained only in its own block and frozen after it. G, S, relation and operator codes keep training. The kind is told, as context normalization already does |

`reh4` in order A is v2's `two-ctx-enzhbuf` design, and `none` in order A is v2's `two-ctx-enbuf`. **Reproduction check:**
their reports must equal v2's exactly; otherwise the difference is stated and `reh4` is reported against this run only.

**Not included:**
- **LwF** (user decision, 2026-11-05). It preserves outputs on present data only, so by the notes' argument it cannot protect
  an absent script.
- **Synaptic intelligence.** The same family as EWC, with importance computed online.
- **Generative replay** (van de Ven et al. 2020). The storage-free alternative to rehearsal; deferred.
- **Routing for `reg`.** The kind is told. Routing by frozen sense code was measured in the [sense growth
  pilot](sense-growth-v1-pilot.md), where no rule was safe on every stream. `reg` is an upper bound for registration, used
  only as a comparison; a registration remedy for the system would have to route without being told.

## EWC selection (before the main seeds)

λ ∈ {0.1, 1, 10, 100, 1000} on seeds 310 and 311, order A only (the Fisher is normalized, so λ sets the penalty against a loss of order 1; the penalty at each block's end is logged). The criterion is fixed now: the mean of the earlier script's and
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
| `reg` | Adapter per kind, 12,576 floats | 50 KB per kind | Linear in K, plus a router when the kind is not told | One small matrix product per input |

At this scale EWC stores more than rehearsal at 4 codes per count. Rehearsal passes EWC's storage after about 8 scripts at
n = 4, or about 30 at n = 1. That crossover is part of the result.

## Computation (planned)

From v2's runtime, a run is about 1,440 s on one CPU thread.
- **Main runs:** 7 arms × 2 orders × 4 seeds = 56 runs, about 2.8 h on 8 workers.
- **Selection:** 10 runs, about 50 min.

The runner must be crash-safe (durable per run). Runtime JSON is separate from results; measured items as the computation
policy requires.

## Results

Run 2026-11-05 at revision `a4f838d` (runner and contract unchanged since the freeze; their hashes are in the runtime JSONs).
Results in `reports/drift-remedies-v1.json.gz` (compressed with `gzip -n -9`; uncompressed SHA-256
`9a1431cf93474f07fbcc5f889608585fcc75b911393c15cd1ae833c253cb65ea`; decompress with `gunzip -k` before running
`.venv/bin/python -m experiments.drift_remedies_v1 summary`) and `reports/drift-remedies-v1-selection.json`. Means over
seeds 300–303 at block 3 unless stated.

**Reproduction is exact.** `none` and `reh4` in order A equal v2's `two-ctx-enbuf` and `two-ctx-enzhbuf` on every identity,
relation and operator read, for all four seeds.

### EWC selection

Seeds 310 and 311, order A. Penalty is λ/2 Σ F (θ − θ*)² at the end of each block, against a final loss of about 0.25.

| λ | Chinese `eq` at block 2 | Chinese `eq` at block 3 | Devanagari `eq` at block 3 | Criterion | Chinese movement | Penalty at block 2 / 3 |
|---|---|---|---|---|---|---|
| **0.1 (selected)** | 0.85 | 0.76 | 0.78 | 0.77 | 0.36 | 0.05 / 0.12 |
| 1 | 0.80 | 0.70 | 0.74 | 0.72 | 0.37 | 0.16 / 0.46 |
| 10 | 0.69 | 0.50 | 0.62 | 0.56 | 0.34 | 0.63 / 1.5 |
| 100 | 0.54 | 0.43 | 0.45 | 0.44 | 0.28 | 2.8 / 4.9 |
| 1000 | 0.33 | 0.34 | 0.20 | 0.27 | 0.13 | 8.4 / 8.4 |

The selected λ is the smallest in the grid, so the optimum may lie below it; the grid was frozen and is not extended here.
Larger λ does not trade the new script for the earlier one: it lowers both, because the penalty accumulated from block 1
stops Chinese being learned in block 2 at all (0.85 → 0.33). At λ = 1000 codes barely move (0.13) and Chinese is still at
0.34, since it was never acquired.

### Main reads

Earlier script = Chinese in order A, Devanagari in order B. Identity is `eq` identity unless "nearest". Movement is the
earlier script's code movement from block 2 to block 3 relative to its spread. False acceptance is for the earlier script.

| Arm | Earlier `eq` A / B | Earlier nearest A / B | New script `eq` A / B | English `eq` A / B | Movement A / B | False accept, \|Δ\| = 1, A / B | Retained bytes |
|---|---|---|---|---|---|---|---|
| `none` | 0.60 / 0.49 | 0.58 / 0.52 | 0.81 / 0.85 | 0.66 / 0.69 | 0.46 / 0.51 | 0.20 / 0.21 | 0 |
| `reh1` | 0.80 / 0.66 | 0.75 / 0.68 | 0.85 / 0.83 | 0.73 / 0.69 | 0.33 / 0.39 | 0.10 / 0.18 | 21,504 |
| `reh2` | 0.79 / 0.68 | 0.77 / 0.70 | 0.83 / 0.84 | 0.70 / 0.71 | 0.32 / 0.37 | 0.09 / 0.18 | 43,008 |
| `reh4` | 0.81 / 0.73 | 0.79 / 0.75 | 0.84 / 0.83 | 0.71 / 0.70 | 0.30 / 0.33 | 0.08 / 0.15 | 86,016 |
| `ewc` (λ 0.1) | 0.77 / 0.53 | 0.70 / 0.60 | 0.82 / 0.83 | 0.82 / 0.86 | 0.35 / 0.40 | 0.13 / 0.24 | 844,880 |
| `ewc-reh1` | 0.81 / 0.72 | 0.77 / 0.72 | 0.81 / 0.82 | 0.85 / 0.88 | 0.29 / 0.33 | 0.12 / 0.18 | 866,384 |
| `reg` (told the script; comparison only) | 0.82 / 0.81 | 0.82 / 0.80 | 0.81 / 0.81 | 0.94 / 0.96 | 0.00 / 0.00 | 0.11 / 0.12 | 100,608 |

Seed spread of the earlier script's `eq` identity (standard deviation over four seeds): 0.01–0.03 for `reh` arms in order
A, 0.02 for `reh2` and `reh4` in order B, 0.07 for `reh1`, `ewc` and `ewc-reh1` in order B (`reh1` B: 0.56–0.74).
The earlier script's identity at block 2, before the new script, is 0.82–0.85 in every arm. Collection balanced accuracy
is 0.96–0.99 for every operator in every arm. Thai stays at 0.12–0.19.

### Against the predictions

1. **Holds.** Exact reproduction; `reh4` order A gives Chinese `eq` identity 0.81.
2. **Holds for the half-gap, partly for the rise.** `reh1` recovers 95% of the gap from `none` to `reh4` in order A and 71%
   in order B. Identity rises with codes per count in order B (0.66, 0.68, 0.73) and is flat in order A (0.80, 0.79, 0.81),
   saturated at one code per count.
3. **First part holds, second fails.** `ewc` lies between `none` and `reh4` in both orders (0.77; 0.53). There is no
   stability–plasticity trade on the grid: no λ above the smallest brings the earlier script closer to `reh4`, and larger λ
   blocks the earlier script's own acquisition in block 2. At the selected λ, EWC's main effect is on English, the
   oldest script (0.66 → 0.82 in A, 0.69 → 0.86 in B), not on the block-2 script.
4. **Holds.** `ewc-reh1` is at least the better of `ewc` and `reh1` (A 0.81 against 0.80; B 0.72 against 0.66).
5. **Holds.** The earlier script's codes do not move (0.00). Its identity, 0.82 and 0.81, is at least `reh4`'s and only
   0.02 below its block-2 value: with codes fixed, the readers alone lose almost nothing. New-script `eq` acquisition is
   0.00 and 0.04 below `none`, inside the 0.05 bound. Its nearest identity is lower (0.77 / 0.79 against 0.82 / 0.83), a
   small cost of a frame fixed after block 1.
6. **Holds in order A, partly in order B.** In A, false acceptance at |Δ| = 1 falls from 0.20 to 0.08–0.13 wherever
   identity is restored. In B it falls to 0.12 under `reg` and 0.15 under `reh4`, stays at 0.18 under `reh1`, `reh2` and
   `ewc-reh1`, and is 0.24 under `ewc`.
7. **Holds at the ends, not in the middle.** `reg` is first, `none` last and `ewc` second to last in both orders. The order
   among `reh1`, `reh2`, `reh4` and `ewc-reh1` differs, within seed spread in order A (0.79–0.81). Devanagari is harder to
   keep than Chinese: it moves more without remedy (0.51 against 0.46) and every non-told arm keeps less of it.

No arm protects the earlier script by stopping the new one: new-script `eq` acquisition is within 0.04 of `none` in every
arm and order.

### Bearing on the design

- **The `eq` errors on drifted scripts are drift errors.** Holding the codes still (`reg`) brings the earlier script to
  within 0.02 of its block-2 value and halves adjacent false acceptance. What is left after rehearsal is code movement, not
  the `eq` reader.
- **Rehearsal is the cheap remedy.** One stored frozen sense code per count (21.5 KB per script) recovers most of the loss
  in order A and about 70% in order B; four per count recover more in order B.
- **EWC alone is weak here and protects the oldest kind.** During block 2 the penalty holds the block-1 solution, so it keeps
  English and, at larger λ, slows acquisition of the block-2 script more than it later protects it.
- **Best arm not told the script: `ewc-reh1`.** It ties `reh4` on the earlier script (0.81 / 0.72 against 0.81 / 0.73) and
  keeps English far better (0.85 / 0.88 against 0.71 / 0.70), at 866 KB, constant in the number of kinds, against 86 KB
  growing linearly. Under the contract's crossover, rehearsal at n = 4 passes this after about 8 scripts.
- **What routing would buy.** `reg` beats every untold arm on English (0.94–0.96) and on Devanagari as the earlier script
  (0.81 against 0.73). That gap is the value of a registration router that does not need to be told the kind; the growth
  pilot found no routing rule safe on every stream.

## Computation (measured)

Runtime in `reports/drift-remedies-v1-runtime.json` and `reports/drift-remedies-v1-selection-runtime.json`. CPU, one
thread per worker, 8 workers, float32.
- **Main:** 56 runs, 11,554 s wall (3.2 h), 90,245 s aggregate worker CPU. Median per run 1,426 s (`reg`) to 1,706 s
  (`none`); 4,500 optimizer steps each. Peak RSS per worker 1.7 GB.
- **Selection:** 10 runs, 2,118 s wall, 14,292 s aggregate worker CPU; selection is 14% of the total CPU.
- **EWC's Fisher:** 63 s per run (64 minibatches after each block; the estimate after block 3 is not used), about 4% of a
  run. EWC runs are not slower than `none` at the median (1,614 s against 1,706 s), so the per-step penalty is below the
  timing noise here.
- **Retained bytes** are as the scaling table: 21,504 per script at one code per count (7 taught counts × 768 float32),
  844,880 for EWC's anchor and Fisher, 50,304 per adapter.
