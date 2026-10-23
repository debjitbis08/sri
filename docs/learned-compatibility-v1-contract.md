# Learned compatibility v1 — bounded contract

**Status at declaration:** specified before implementation. Question: can
compatibility learned through a core support **unseen combinations of seen
components**, and give multiple valid responses where several are valid and one
where only one is? This is a mode-level study. It reuses the phase B semantics
of whole-module moves and exact integer energies. Earlier reports stay unchanged.

## Why the core is the candidate for generalization

A direct pair table, a ternary table, or a core with one mode per training pair
(phase B's enumerating construction) stores each observed combination
independently. Under such a table, a held-out pair receives no support, so
**held-out recall is 0 by construction**. Generalization requires parameter sharing.
A pairwise star shares parameters: satellite modes associated with the same core mode
k form a rectangle, A-set(k) × B-set(k), and the supported relation is the union of
these rectangles. This is phase B's pairwise limit turned into an inductive bias.
Whether it helps depends on what the core learns, which is what this study measures.

## Model

- A and B each have m = 6 modes; the core has n ∈ {2, 3, 4, 8} modes.
- Learned binary supports W_A[a][k] and W_B[b][k] ∈ {0,1}, all starting at 0.
  Supports **saturate**: compatibility is a relation, not a frequency. Unbounded
  counts would break ties between equally valid responses by training frequency.
- Energy: E(a,b,k) = −W_A[a][k] − W_B[b][k].

**Responses to a cue.** Clamp A = a. B and K make whole-module moves. An attractor is
a sink of the move graph restricted to moves that do not increase energy: a set
closed under such moves, with no strictly decreasing exit. Resp(a) is the set of B
modes appearing in attractors. Cues on B are handled symmetrically.

*Set-valued attractors are declared deliberately.* Two responses b, b′ sharing core
mode k are one B move apart, so by phase B's adjacency lemma they cannot both be
strict minima. Multiple valid responses within one core mode can exist only as one
equal-energy attractor set. Responses in different core modes may still be separate
strict minima.

## Learning rules (local information only; no access to the ground truth)

Training presents valid pairs (a,b) one at a time, in a given order. The core takes
the mode of lowest energy with A = a and B = b clamped.

- **Recruitment:** if the best energy is 0, meaning neither a nor b is supported
  anywhere, the core takes the lowest-index unused mode. If none is unused, it takes
  mode 0. This forced reuse is reported separately.
- **Ties:** among equal-energy best modes, the lowest index wins.
- **Join (local):** set W_A[a][k*] = W_B[b][k*] = 1. Nothing else changes.
- **Merge (nonlocal consolidation):** if a's supporting modes and b's supporting modes
  are disjoint and both non-empty, merge every mode supporting b into a's
  lowest-index mode, by OR-ing supports and freeing the merged modes. Then apply
  join. This rule is explicitly *not* local; it is the consolidation-style comparison.
- **Controls:**
  - *memorize* stores a direct pair table, with no core;
  - *ternary* stores W[a][b][k] for the triple used;
  - *enumerate* gives every training pair its own fresh core mode, assigned in
    training order (so each pair is its own 1×1 rectangle).

## Relations, declared

- **blocks:** {a0,a1,a2}×{b0,b1,b2}, {a3,a4}×{b3,b4}, and {a5}×{b5}. That is 14 pairs.
  Cue a5 has exactly one valid response; the others have 2 or 3.
- **overlap:** {a0,a1,a2,a3}×{b0,b1} ∪ {a2,a3,a4,a5}×{b2,b3,b4}. That is 20 pairs.
  a2 and a3 are valid in both rectangles.
- **random:** a fixed 14-pair relation drawn with seed 0, every row and column
  non-empty. This is a control without declared rectangle structure.

**Splits.** Leave-one-out and leave-two-out over valid pairs, plus 100 seeded
random splits that hold out half the pairs. A split is admissible only if every
mode used by the relation still appears in training; inadmissible splits are
counted, not run. **Orders:** 20 seeded orders per split for each learner.

*Amendment before any run.* Plain seeded half-splits were admissible only 2 of 100
times for blocks, and 1 of 100 for random, because holding out half the pairs usually
removes some mode entirely. Half-splits are therefore collected by seeded rejection
sampling until 100 **distinct admissible** splits are found, or until all of them
are found if fewer exist (searched to seed 200,000). Leave-one-out and leave-two-out
are unchanged.

## Propositions, proved before running

- **P-memo.** memorize, ternary, and enumerate give held-out recall 0.
- **P-response.** The attractor projection equals the union of B-set(k) over every k
  with W_A[a][k] = 1. This is checked against the dynamics in every run.
- **P-merge.** With n at least the number of connected components of the training
  bipartite graph, merge learns exactly those components, whatever the order.
  Hence:
  - **blocks:** held-out recall is 1 iff each block's training graph is connected;
    precision is 1 whenever n ≥ the number of components.
  - **overlap:** the overlap connects everything into one component, so merge
    supports all of {a0…a5}×{b0…b4}, a superset of the relation. Precision is
    20/30 when nothing is held out.
  - **capacity:** with fewer core modes than components, forced reuse merges
    components and creates invalid responses.
- **P-join.** Each join rectangle lies inside a merge component, so join's responses
  are a subset of merge's: precision at least merge's, recall at most merge's.
  Join is order-dependent.

### Corrections after the first run (original text above is kept)

The first run reported violations of all three checked propositions. Each traced
to an error in the proposition, not in the code:

1. **P-response scope.** The overlap relation never uses b5, but b5 was evaluated
   as a cue. An untrained cue has no support, so its attractors take in every
   supported mode, while the formula predicts the empty set. *Correction:* the
   proposition holds for trained cues, and cues are restricted to the modes a
   relation uses. This also removes an unfair precision penalty from overlap.
2. **P-merge condition.** "n ≥ the components of the final training graph" is
   insufficient: an intermediate prefix of the training order can have more
   components. Forced reuse then fuses pieces that belong to different final
   components. A concrete counterexample is in `tests/test_compat_learning.py`.
   *Correction:* n ≥ the maximum component count over training prefixes.
3. **P-join subset.** This fails under forced reuse in either learner.
   *Correction:* it holds when neither learner had forced reuse.

The report records the checks under both the declared and the corrected
conditions.

4. **P-join precision claim (found in review).** Response containment implies that
   join's recall is at most merge's. It does **not** imply that join's precision is
   at least merge's: the discarded responses can be disproportionately valid.
   Counterexample inside the declared grid: overlap, leave-one-out split 13
   (holding out (3,4)), order 9, n = 8, with no forced reuse. Join has precision
   19/29 and held-out recall 0; merge has 20/30 and 1. *Correction:* P-join claims
   only containment, and hence recall(join) ≤ recall(merge). The precision clause is
   withdrawn. The aggregate results are unaffected. The counterexample is in the tests.

## Measurements

For each learner, relation, split, n, and order:

- **Held-out recall:** a held-out valid pair counts as recovered if b ∈ Resp(a) and
  a ∈ Resp(b).
- **Seen recall.**
- **Precision:** valid responses over all responses, across all cues.
- **Invalid responses:** count.
- **Exact response sets:** the fraction of cues with Resp = V.
- **Multiplicity:** for cues with |V| = 1, whether |Resp| = 1; for cues with
  |V| > 1, whether |Resp| > 1.
- **Core use:** modes used, merges, forced reuses.
- **Core participation:** with K clamped at each single mode, the response set
  shrinks for cues whose valid responses span several core modes.

## Refutation conditions

- If any proposition fails in a run, there is a defect to resolve before results are
  read.
- If merge or join reaches precision 1 and held-out recall 1 on **random** splits as
  often as on **blocks**, the apparent generalization is just overgeneralization.
- If no learner reaches held-out recall above 0 with precision 1 on any blocks split,
  the core gives no inductive benefit here.

## Scope discount

Mode level only. There are no unit-level internals: graded results exist only for
two-mode modules. Learning rules and relation families are hand-declared, with no
claim that they are a biological or SRI learning law. Training sees only valid
pairs; negative evidence is not used. Stop at the declared grid.
