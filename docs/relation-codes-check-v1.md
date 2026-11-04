# Relation codes check v1

Status: **contract written 2026-11-05 before any run; results follow below as they come.** The user asked for a comprehensive
check of their idea (2026-11-05) and said not to wait for their decisions; every choice here is mine and is open to revision.

## The idea

After the [two-script reads](two-script-reads-v1.md): relational order survived teaching a new script while count identity did
not, because a shared projection bent and moved every form's absolute place. The user's proposal: **store relation-level
information, not count-level information, and for that store the relation itself as a code in a code space.** Open: one code
space for operators and objects, or two.

## Questions

1. **Is relation-level information there when identity is gone?** For pairs of forms whose items are not identified (Chinese
   after block 3), can the relation between them (next, previous, same, +2, larger) be read, and better than by identifying
   both items and subtracting?
2. **Ordinal or metric?** Is what survives the order only (rank), or also the step sizes? A difference code c(b) − c(a) needs
   step sizes; an ordinal code needs only rank.
3. **Can forgotten identity be derived from relations?** Given one anchor item of known count, are the other items' counts
   recoverable through relation reads from the anchor?
4. **One code space or two?** Answered only by training (part C below), since stage A reads one fixed code space.

## Part A: relation reads on existing checkpoints (no training of the system)

**Checkpoints.** The shared output pilot block-1, block-3, delay and relearn checkpoints of `gp` (the arm in which Chinese was
learned at block 2, 0.89, then lost) and `common-k-cand`, seeds 300–303. Kinds: collections, taps, English, Chinese, Devanagari,
Thai; 30 evaluation items per count.

**Codes.** Each item's core code (`core_inner`, what readers read) and its projection output, as in
[where the mismatch is](two-script-reads-v1.md#where-the-mismatch-is-2026-11-05).

**Pairs.** For each ordered pair of kinds (A, B) in a fixed list, all 81 count pairs (x, y) with item index aligned, so 81 × 30
pairs. The relation target is d = y − x (17 classes, −8..8). Reported: exact d; next (d = +1) detection AUROC against all other
pairs; sign of d among d ≠ 0; |d̂ − d| ≤ 1.

**Relation reads** (each fitted only on pairs drawn from collections and taps, including mixed collection–taps pairs, from the
first half of items; scored on held-out halves and on all script pairs):
- **R-diff:** a linear softmax read on c(b) − c(a). The metric, offset code (TransE / vector-offset style).
- **R-pair:** a two-layer MLP (64 hidden) on [c(a), c(b), c(b) − c(a)]. A learned pair encoder: a relation code computed from
  the example pair.
- **Via identity (the count-level baseline):** an MLP of the same width fitted to identify single items (collections and taps,
  first half), d̂ = id(b) − id(a). Relation-level information exceeds count-level information where a relation read beats this.
- **System readers:** the trained `next` and `more` readers' own logits on the pair (next AUROC and sign accuracy). These read the
  relation the way the system does.
- **Rank code (fixed rule, diagnostic):** each item's position on the collection count axis (ridge fit on collections, as in v4)
  converted to its rank within its own kind, then to a count by quantile (9 equal bins). This uses only the order within a
  kind and the fact that a kind's items cover nine counts evenly: an assumption, stated. Reported as identity and as d.

**Within-kind order.** Per kind: Spearman correlation of axis position with count (rank kept) against the slope of the axis
position on count (scale kept; collections ≈ 1).

## Part B: identity derived from relations

For each kind, each anchor (one item of known count a₀) predicts every other item's count as a₀ + d̂(anchor, x) with R-pair and
R-diff, clipped to 1..9. Reported with one anchor (averaged over all anchors) and with three anchors of different counts
(averaged d̂-votes, 200 random triples). Compared with `same` identification over M and nearest-mean identification.

## Part C: one code space or two (trained pilot), only if part A finds relation information beyond identity

Planned shape, fixed after part A: a small standard network on the frozen sense codes, taught collections, then Chinese, then
Devanagari, comparing (i) count-level teaching (matching, as now), (ii) relation codes in the object space (one space), and
(iii) relation codes in a separate structural space bound to objects (two spaces). The decisive read: after Devanagari, do
relations still apply to Chinese, and can Chinese identity be derived from them? Its contract is added here before it runs.

## Predictions (part A and B)

1. **[local evidence]** For Chinese pairs at block 3 in `gp`, sign accuracy stays high (≈ 0.85, as comparison) while exact d is
   low. R-pair beats via-identity on exact d and within one.
2. **[derived from the v4 compression]** R-diff transfers worse than R-pair and the rank code: steps are compressed, so offsets
   shrink. Next AUROC on Chinese pairs is well above chance but exact +1 is weak.
3. **[derived]** Spearman is high (≥ 0.8) for Chinese at block 3 while slope is far below 1: order kept, scale lost. The rank
   code then recovers much of Chinese identity.
4. Thai: no order, every read near chance.
5. Anchored derivation beats `same` for Chinese at block 3 in `gp`, if prediction 1 holds.

## Prior work

[familiar] sources, recalled.
- **Relational representations as codes.** TransE (Bordes et al. 2013) and word2vec analogies treat a relation as an offset
  between entity codes: R-diff. Relation Networks (Santoro et al. 2017) compute a relation code from a pair with an MLP: R-pair.
- **One space or two.** Tensor product representations (Smolensky 1990) keep separate role and filler spaces bound by an outer
  product; holographic reduced representations (Plate 1995) and vector-symbolic architectures bind in one space. The
  Tolman–Eichenbaum Machine (Whittington et al. 2020) separates a structural code (relations, abstract position) from a sensory
  object code and binds them; its structural code transfers across environments with new objects. This is the closest match to
  the idea, and predicts that a separate structural space generalizes to new objects.
- **Order versus magnitude.** The symbolic distance effect and ordinal position coding (Nieder) favour an analogue order code.
  Transitive inference shows order derived for untrained pairs.
- **Known failure modes.** A pair MLP can memorize per-item identity and subtract, which is count-level information in
  disguise; the via-identity baseline of the same width controls for it. Probes measure what is decodable, not what the system
  uses; the system-reader rows control for that. The rank code assumes a balanced item set.
- **Cheaper alternatives.** Nearest-mean and ridge identity (v4) are the cheap identity baselines, included.

## Computation (declared)

Per checkpoint: codes for 6 kinds × 270 items (one projection and core solve each). Pairs: 10 kind pairs × 2,430. R-diff is
a 16 × 17 softmax and R-pair an MLP of about 16·3·64 + 64·17 ≈ 4.2k weights, each trained full batch with Adam for 400 steps on
about 3.6k pairs. Identity MLP ~2k weights. System readers: 2 logits per pair. All under 1 GFLOP per checkpoint; 2 arms ×
4 seeds × 4 points = 32 checkpoints. Storage: nothing retained but the report. Timing goes to a separate runtime JSON.

## Commands

```
.venv/bin/python -m experiments.relation_codes_check
.venv/bin/python -m experiments.relation_codes_check summary
```

## Part A2: context-normalized identity (added before its run, after part A's results)

`experiments/relation_codes_cn.py`, report `reports/relation-codes-check-v1-cn.json`. Each kind's codes are standardized over that
kind's own evaluation items (context normalization, Webb et al. 2020: the context is one kind's 270 items); collections over their
own. Identity by nearest collection-count mean and by the ridge count axis, raw and context-normalized. It tests whether a
per-script shift and scale, which a neural normalization over a context removes, accounts for the identity the rank code recovers.
**Prediction:** context-normalized identity for Chinese at block 3 in `gp` approaches the rank code's 0.48; raw stays near 0.15.

## Results, parts A, A2 and B (run 2026-11-05)

Core codes (`core_inner`) unless stated; mean over seeds 300–303, range of seed means in brackets. Chance: exact d 1/9 on
average over the 81 count pairs, within one about 0.31, sign 0.5, identity 0.11. `gp` is the arm in which Chinese was learned
(0.89 at block 2) and then lost.

**Within-kind order.** Spearman ρ of each item's place on the collection count axis against its count (order kept), the axis
slope (scale kept; collections 0.98–1.00), and identity from rank within the kind:

| Arm | Point | Chinese ρ | Chinese slope | Chinese rank identity | Chinese probe identity | English ρ | Devanagari ρ |
|---|---|---|---|---|---|---|---|
| `gp` | block 1 (untaught) | 0.05 | 0.05 | 0.19 | 0.20 | **0.94** | −0.14 |
| | block 3 | **0.87** (0.80–0.94) | 0.44 | **0.48** (0.40–0.57) | 0.21 | **0.20** | 0.70 |
| | delay | 0.72 | 0.31 | 0.41 | 0.18 | 0.12 | 0.58 |
| | relearn | 0.96 | 0.72 | 0.71 | 0.34 | 0.78 | 0.66 |
| `common-k-cand` | block 3 | 0.87 | 0.41 | 0.49 | 0.22 | 0.06 | 0.89 |
| | delay | 0.80 | 0.36 | 0.43 | 0.25 | 0.04 | 0.86 |

The projection output gives the same picture (Chinese ρ 0.82 at block 3 in `gp`, rank identity 0.47).

**Pair relations, `gp`, block 3** (exact d / within one / sign / next AUROC):

| Pair | R-pair | R-diff | Via identity | Rank code | System readers (sign / next AUROC) |
|---|---|---|---|---|---|
| Collection × collection | 0.89 / 1.00 / 1.00 / 1.00 | — | — | — | 1.00 / 1.00 |
| Chinese × Chinese | 0.26 / 0.52 / 0.85 / 0.79 | 0.27 / 0.50 / 0.82 / 0.76 | 0.24 / 0.52 / 0.73 | **0.37 / 0.69 / 0.85** | **0.87** / 0.71 |
| Collection × Chinese | 0.17 / 0.62 / 0.85 / 0.72 | — | 0.21 / 0.68 / 0.84 | **0.47 / 0.81 / 0.90** | 0.89 / 0.67 |
| Chinese × Devanagari | 0.15 / 0.51 / 0.76 / 0.72 | — | 0.19 / 0.57 / 0.75 | 0.24 / 0.67 / 0.80 | 0.81 / 0.75 |
| Devanagari × Devanagari | 0.40 / 0.66 / 0.80 / 0.84 | — | 0.39 / 0.73 / 0.75 | 0.46 / 0.73 / 0.83 | 0.85 / 1.00 |
| English × English | 0.11 / 0.32 / 0.48 / 0.50 | — | 0.15 / 0.35 / 0.25 | 0.19 / 0.33 / 0.52 | 0.57 / 0.50 |
| Thai × Thai | 0.17 / 0.30 / 0.32 / 0.51 | — | 0.15 / 0.30 / 0.20 | 0.17 / 0.27 / 0.38 | 0.46 / 0.52 |

At block 1, English × English (just taught) reads 0.31 / 0.60 / 0.79 with R-pair and **0.51 / 0.87 / 0.91** with the rank code.

**Identity derived from anchors** (part B), Chinese, block 3: R-pair 0.23 (one anchor), 0.24 (three), 0.19 (one collection
anchor); R-diff 0.23, 0.28, 0.17; against probe identity 0.21 and `same` 0.20. `common-k-cand`: 0.25, 0.29, 0.22 against 0.22.

**Context-normalized identity** (part A2), nearest collection-count mean, raw → context-normalized:

| Arm | Point | Chinese | Devanagari | English | Thai |
|---|---|---|---|---|---|
| `gp` | block 3 | 0.15 → **0.35** (axis 0.39) | 0.82 → 0.41 | 0.12 → 0.16 | 0.12 → 0.14 |
| | delay | 0.10 → 0.25 | 0.49 → 0.39 | 0.11 → 0.14 | 0.11 → 0.16 |
| `common-k-cand` | block 3 | 0.16 → 0.37 | 0.86 → 0.78 | 0.12 → 0.12 | 0.14 → 0.14 |
| | relearn | 0.53 → 0.78 | 0.56 → 0.75 | 0.17 → 0.31 | 0.10 → 0.14 |

**Predictions.**
1. *Sign high, exact low, R-pair beating via-identity:* the first two hold (sign 0.85, exact 0.26); the third fails, 0.26 against
   0.24 exact and equal within one. Only the sign is clearly beyond identity (0.85 against 0.73).
2. *R-diff worse than R-pair:* fails; they are equal. Both read the same bent codes.
3. *Order kept, scale lost, rank code recovers identity:* **holds.** ρ 0.87, slope 0.44, rank identity 0.48 against 0.21.
4. *Thai at chance:* holds.
5. *Anchored derivation beats `same`:* fails (0.23–0.28 against 0.20).
6. *(A2) context normalization approaches the rank code:* partly. It recovers 0.35 of the rank code's 0.48 for Chinese, and costs
   Devanagari in `gp` (0.82 → 0.41), where nothing was misaligned.

**What this shows.**
1. **Relation information read from a pair is no better than identity.** A relation code computed from two object codes
   (R-pair, R-diff), whether learned or an offset, carries what the two identities carry, plus order. Anchored derivation through
   such codes recovers nothing. A pair relation code inherits the bend of the object codes it is computed from.
2. **What survives is order across the whole set of a script.** Chinese forms keep their rank among themselves (ρ 0.87) while their
   scale shrinks (slope 0.44). Rank within the script recovers identity at 0.48, more than twice any per-item read. The relative
   position that survives is relative to the *other Chinese forms*, not to collections, and reading it needs the set.
3. **Order is not always kept.** English, taught in block 1 (ρ 0.94), loses its order by block 3 (ρ 0.20). The idea's premise,
   that relational order is maintained, holds for one forgotten script (Chinese, one block back) and fails for the other
   (English, two blocks back, weakly learned: matching 0.26 at block 1).
4. **A neural context normalization recovers part of it** (0.35), and harms what was aligned. It removes a per-script shift and
   scale; the rest of the bend is not affine.
5. **For the design:** storing relations as codes computed from pairs of current object codes would not have protected Chinese.
   What would have is structure defined over a context, the relation of each item to the other items of its kind, or
   relations held in a space that later teaching does not bend. Part C tests this.

### Computation (measured)

- Part A/B: 8 tasks × 4 checkpoints × 2 levels, 171 s wall on 4 CPU workers, 547 s aggregate CPU, 2.5 GB peak RSS per worker.
  The one-checkpoint checks took about 80 s.
- Part A2: 112 s wall, 339 s aggregate CPU.
