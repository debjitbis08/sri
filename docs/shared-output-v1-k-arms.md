# Shared output v1: K arms (relational completion readout)

Status: **draft extension of the [shared output v1 contract](shared-output-v1-contract.md), 2026-11-04, not frozen. [Pilot](shared-output-v1-k-arms-pilot.md) run 2026-11-04: criteria fail for `common-k-shared`.** It adds two
readout arms to the [pilot](shared-output-v1-pilot.md) and changes nothing else. The proposal came from the user (2026-11-04): replace
`g` and `P` with one compatibility S(r, a, b, y), with a query encoder Q, a key encoder K and a scorer F. Translation into a base
form is scrapped for now (user, 2026-11-04), so this readout is tested on its own.

## Question

`common-raw` already reads every relation through one scorer, s([Q(r, a, b), y]), a cross-encoder, and transferred names at
+0.05 over `gp` (seeds −0.10 to +0.19). Does a learned key projection K change that?

1. **`common-k-cand`, the proposal as written.** S = Q(r, a, b) · K(y), with K over candidates only.
2. **`common-k-shared`.** The one K is also applied to both query slots: S = Q(r, K(a), K(b)) · K(y).

**Why the second arm.** A name X_n is taught only as `same(X_n, collection n)` and tested in a *query slot*, (n∓1, X_n) → n±1.
In `common-k-cand` the `same` loss shapes K(collection) and Q's reading of X_n. No term pulls X_n's code toward n's where
continuation reads it. In `common-k-shared` the `same` loss compares K(X_n) with K(n), and continuation reads K(X_n). So the
sameness evidence acts in the space the relations read. This holds substitutivity by a shared learned embedding, not by
grounding: no recalled item replaces X_n.

## Arms

All arms: same seeds, curriculum, losses, weights, batches, candidate draws and participants M as the pilot. Only the readout
differs.

| Arm | Score | Slots read | Source |
|---|---|---|---|
| `gp` | `g(a, b; r)`; continuation −exp(scale)·‖P(prev, cur) − y‖² + b_c | raw codes | pilot report (reused) |
| `common-raw` | s([Q(r, a, b), y]), an MLP | raw codes | pilot report (reused) |
| `common-k-cand` | exp(scale) · ⟨Q(r, a, b), K(y)⟩ / √16 + b_s | raw codes | new |
| `common-k-shared` | exp(scale) · ⟨Q(r, K(a), K(b)), K(y)⟩ / √16 + b_s | K(codes) | new |

**Shapes.**
- **Q** is `g` with its last layer 64 → 16, as in `common`. Its input is [slot 1, slot 2, token]. A pair query holds ∅, a learned
  empty vector, in slot 2. In `common-k-shared`, ∅ lives in K's output space.
- **K(y) = P([y, κ])**, with κ a learned constant 16-vector (initialized at zero). K keeps `P`'s masked sparse48 first layer, its
  RigL schedule and its 64 → 16 output. The constant operand is the cheapest way to keep `P`'s masks and rewiring. In effect, its
  share of the first layer is a learned bias.
- **F** is a scaled dot product with a learned log-scale (`scale`, the parameter `gp`'s distance logit uses) and a bias b_s.
  `same`, `next`, `more` and continuation all read the same S.
- **Parameter counts.** The K arms have 1,015 more reader parameters than `gp`: `g`'s 64 → 16 output instead of 64 → 1 adds 975,
  plus κ, b_s, the continuation token and ∅, minus b_c. That is about 8% of 13,003. They are reported, not matched. Matching would
  shrink a hidden layer and break the shared sparse48 layout.

**Unchanged reads.** Recall is the top participant, with abstention below a validation threshold. `predict` returns the recalled
participant, as in the common arms, so composition rolls a participant forward. Thresholds, identification, N1–N3, and the gate
and audit machinery are as in the contract.

**Added read (all new runs).** *Substitutability.* Is a name's nearest per-count mean of M its own count, in core space and, for
the K arms, in K space? This is read at block 3, delay and relearn. The reused `gp` and `common-raw` reports lack it.

## Reuse and the reproduction check

`gp` and `common-raw` are not rerun. Their pilot reports are reused. The readout code was refactored: `_q` takes `None` for the
empty slot. So `common-raw` at seed 300 is rerun under its own tag, and every measure in that report must equal the pilot's, except
the added `substitutability` key. If any measure differs, the reuse is withdrawn and all four arms are rerun together.

## Predictions

1. **[derived] `common-k-cand` ≈ `common-raw` on N1.** K does not reach the query slot (see *Why the second arm*).
2. **[derived] `common-k-shared` transfers if K-space substitutability rises.** N1 given identification should track how often
   K(X_n) lands nearest count n. If K-space substitutability is high and N1 is not, Q does not read K-space nearness as sameness.
3. **[prior work] Dot-product scoring costs some familiar accuracy** against the cross-encoder (Humeau et al. 2020). It can
   accept only a half-space of K space per query. Familiar continuation in the K arms may fall below `common-raw`'s 0.94 at delay.
4. **[prior work] Shared embeddings are exposed to interference.** Relearning English moves the shared image projection most
   ([dynamics](relearning-dynamics-v1.md)). In `common-k-shared` that movement passes through K into every slot. So relearn reads
   may fall more than in `common-k-cand`.
5. **[local evidence] Never-taught symbols will still be accepted.** Nothing in either arm trains rejection of unfamiliar inputs.

## Pre-declared criteria (pilot, at delay unless stated)

Descriptive at four seeds.

- **Gates, per K arm.** Name identification at block 3 ≥ 0.80. If it is lower, N1 is also read given correct identification,
  and the unconditioned comparison is reported but not interpreted. Familiar continuation ≥ 0.60 and ≥ `common-raw` − 0.10.
- **Advance `common-k-shared` to confirmation** if:
  - **Transfer:** `common-k-shared` − `common-raw` on N1 (mean of directions) ≥ +0.10 paired, and positive in at least 3 of 4
    seeds;
  - **Rejection:** removed-set false acceptance no more than 0.05 above `gp`'s;
  - **Prior abilities:** familiar reads and each N3 prior read within 0.05 of `gp`, and English matching after relearning too.
- **Attribution, reported.** `common-k-cand` − `common-raw` (a key tower on candidates), `common-k-shared` − `common-k-cand` (K
  in the slots), `common-k-shared` − `gp`.
- **Not a criterion.** Agreement between recall and acceptance holds by construction in every common arm, since both read one
  logit (brief: "a wiring check"). The proposal's test, whether learning to predict improves verification, needs arms that train
  one read without the other. Those are not in this extension.

## Prior work

Search date 2026-11-04. No new search. Every source is **[familiar]**, recalled rather than re-read. The shared survey is
[prior work](prior-work.md#shared-output-scorer-and-naming-only-transfer). This adds the entries below to it.

- **Bi-encoders versus cross-encoders.** Dual encoders (DSSM, Huang et al. 2013; dense passage retrieval, Karpukhin et al. 2020)
  score ⟨Q(query), K(candidate)⟩. Poly-encoders (Humeau et al. 2020) show that cross-encoders are more accurate and bi-encoders
  are much cheaper over many candidates. `common-raw` is the cross-encoder and `common-k-cand` the bi-encoder.
- **Siamese and shared embeddings.** Signature verification (Bromley et al. 1993) and contrastive metric learning (Chopra,
  Hadsell and LeCun 2005) apply one encoder to both sides of a sameness judgment. Knowledge-graph embeddings (TransE, Bordes et
  al. 2013; DistMult, Yang et al. 2015; ComplEx, Trouillon et al. 2016) share each entity's embedding across the head, tail and
  candidate positions. That sharing is what lets a fact about an entity in one position inform a query in another.
  `common-k-shared` is this design on perceptual codes.
- **Inductive entities** from few links (Hamaguchi et al. 2017): a new entity is embedded from its neighbours and then answers
  queries about relations it was never seen in. Naming-only transfer is the one-link case.
- **Contrastive objectives.** The 9-way cross-entropy over candidates is InfoNCE (van den Oord et al. 2018) with the update's
  candidates as negatives.
- **Known failure modes.**
  - Representational collapse of a shared encoder under a sameness loss alone. Here, negatives in matching and continuation
    counter it.
  - DistMult-style dot-product scores are symmetric when Q does not depend on the slot order. Here Q reads ordered slots and a
    relation token, so asymmetric relations remain expressible.
  - A dot product accepts a half-space per query. Relations with a multimodal acceptance set (several separated valid fillers)
    need Q to place them on one side. This limits universality.
  - Uncalibrated scores and open-set acceptance (Hendrycks and Gimpel 2017), as in the contract.
- **Cheaper alternative.** Read-time grounding was cheaper and lifted N1 to 0.77, but the user dropped grounding. Cosine on
  core codes with no learned K is the zero-parameter reference: the core-space substitutability read.

## Supplied versus learned

| Supplied | Learned |
|---|---|
| One K shared across candidates (both arms) and slots (`common-k-shared`); the dot-product form of F | Q, K, κ, ∅, tokens, scale, b_s; the projection and core |
| M's membership, candidate sets, training targets | — |
| Decision rules: top participant, abstain below a validation threshold | — |

No relation-specific form is supplied. Every relation reads the same S.

## Scaling

- **Inference over N candidates.** K(y) does not depend on the query, so the keys of a fixed inventory are computed once per read
  point and cached: O(N) passes of K. Each query then costs one Q pass plus N dot products of width 16. The cross-encoder needs N
  MLP passes per query. At large N, maximum-inner-product search over cached keys is sublinear (approximate nearest neighbours),
  consistent with a cheap first stage over memory.
- **Memory.** Keys are 16 floats per participant. A long-term memory stores sensory traces and computes keys at read time, so
  key caches are invalidated when K changes. During training, keys of M are recomputed per update.
- **Names** add no parameters. Each costs only its naming rows.

## Computation

**Declared dimensions.** These are the contract's: code 16, reader hidden 64, tokens 8, 48 structure units per first layer, M of 72
participants, 45 names, 8,200 updates per run, 16 batch episodes per update. Two new arms × 4 seeds = 8 runs, plus 1 reproduction
run and 2 smoke runs.

**Theoretical cost.** These are estimates, multiply-add × 2, dense before masking.
- **Q or K pass:** about 2 · (40·64 + 64·64 + 64·16) ≈ 15k FLOPs for Q, and 2 · (32·64 + 64·64 + 64·16) ≈ 14k for K.
- **Continuation row with 9 candidates:**
  - `common-k-cand`: 1 Q + 9 K + 9 dots ≈ 140k FLOPs;
  - `common-k-shared`: plus 2 K passes for the slots, ≈ 170k;
  - `common-raw`: about 127k;
  - `gp`: about 14.6k.
- **Query over M at a read point:** 72 K passes, cached, then about 15k + 72 · 32 FLOPs per query.

**Storage.** Reader parameters are about 56 KB float32 per arm (14,018 parameters), plus Adam's two moments. Cached keys are
72 · 16 floats (4.6 KB). No raw images are retained.

**Measured.** In `reports/shared-output-v1-pilot-k-runtime.json`, `-pilot-ksmoke-runtime.json` and
`-pilot-repro-runtime.json`; the reproduction comparison in `reports/shared-output-v1-repro-check.json`: wall and CPU seconds per run, aggregate worker CPU, peak RSS and GPU peak memory.
The reproduction and smoke runs count as selection cost. No tuning: all settings are the contract's.

**Expected cost.** Common-arm runs took 230–261 s each. 8 runs on 3 workers take about 12–15 minutes, plus about 4 minutes for
the reproduction.

## Commands

```
.venv/bin/python -m experiments.shared_output_repro
NFI_DEVICE=cuda SO_TAG=-repro SO_ARMS=common-raw SO_SEEDS=300 SO_WORKERS=1 /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot pilot
NFI_DEVICE=cuda SO_ARMS=common-k-cand,common-k-shared SO_OUT=pilot-k SO_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot pilot
.venv/bin/python -m experiments.shared_output_summary reports/shared-output-v1-pilot.json reports/shared-output-v1-pilot-k.json
```

## Implementation

- `src/shared_output.py`: arms `common-k-cand` and `common-k-shared` (`KEYED`), `Readout.key`, and `_q` with the empty slot as
  `None`.
- `experiments/shared_output.py`: the substitutability read. The identification read's Devanagari labels are now built from the
  actual set sizes: smoke data holds fewer than 5 forms per count. Full runs are unchanged (the reproduction is exact).
- `experiments/shared_output_repro.py`: the reproduction comparison.
- `experiments/shared_output_summary.py`: merged reports and the K-arm criteria.
- `tests/test_shared_output.py`: 3 added checks (12 in all):
  - parameter counts and masks;
  - gradient to items and κ;
  - K entering the query only in `common-k-shared`.
