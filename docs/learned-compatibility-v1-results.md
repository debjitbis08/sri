# Learned compatibility v1 — the core generalizes by shared modes, exactly as far as evidence connects

**Result:** at mode level, a pairwise core that *learns* shared modes generalizes
to unseen combinations of seen components. With every pair but one in training,
it gives exact response sets on the block relation:

- held-out recall 1 and precision 1;
- a unique response for the cue with one valid partner;
- multiple responses for cues with several.

Memorization, ternary tables, and an enumerating core recover **no** held-out pair
in any run. What the core learns is precisely the connected components of the
training pairs. So it generalizes as far as training evidence connects modes, and
it **overgeneralizes** when relations overlap or lack rectangle structure. The
local join rule is order-dependent and weaker. Neither rule represents overlapping
categories.

[Contract](learned-compatibility-v1-contract.md), including the corrections made
after the first run. [Report](../reports/learned-compatibility-v1.json):
98,452 runs, about 4 minutes. Every attractor set is computed from the dynamics and
matched against the proposition's formula, with 0 violations after correction.

## Propositions

- **P-memo:** confirmed. The controls recovered 0 held-out pairs in all 1,812
  control runs.
- **P-response** (trained cues): confirmed in every run.
- **P-merge:** with the corrected condition (core size ≥ the largest component count
  over training prefixes), merge equals the training components in 29,999 of 29,999
  runs. Under the condition as originally declared, 1,800 of 38,300 runs mismatched;
  every one involved forced reuse.
- **P-join ⊆ merge:** confirmed in all 26,681 runs without forced reuse. It failed
  in 3,435 runs with forced reuse. Containment implies only recall(join) ≤
  recall(merge). The contract's further claim that join's precision is at least
  merge's was **wrong** and has been withdrawn (correction 4). A review
  counterexample within the grid has join at 19/29 against merge at 20/30.

## Main results at n = 8 (no forced reuse for merge)

| Relation, split | Learner | Held-out recall | Precision | Perfect runs (recall 1 and precision 1) | Exact cues |
| --- | --- | --- | --- | --- | --- |
| blocks, leave-one-out | merge | 1.000 | 1.000 | 100% | 1.000 |
| blocks, leave-one-out | join | 0.792 | 1.000 | 79% | 0.965 |
| blocks, leave-two-out | merge | 0.973 | 1.000 | 97% | 0.991 |
| blocks, half | merge | 0.123 | 1.000 | 0% | 0.240 |
| overlap, leave-one-out | merge | 1.000 | 0.667 | 0% | 0.182 |
| overlap, leave-one-out | join | 0.828 | 0.749 | 4.5% | 0.342 |
| random, leave-one-out | merge | 0.917 | 0.398 | 0% | 0.007 |
| random, leave-one-out | join | 0.275 | 0.605 | 0% | 0.226 |
| any relation and split | memorize, ternary, enumerate | 0 | 1.000 | 0% | — |

**Multiple responses where appropriate.** On blocks leave-one-out, merge gives a
unique response to every cue with one valid partner (100%) and multiple responses
to every cue with several (100%). Responses within a core mode form one
equal-energy attractor set, as the contract required. Phase B's adjacency lemma rules
out the alternative, multiple strict minima differing only in B.

## How to read the pattern

- **Generalization is transitive closure through the core.** If a0 is trained with
  b0, and b0 with a1, a shared core mode makes a1 compatible with a0's other
  partners. The held-out recall of merge is exactly the rate at which held-out pairs
  stay within connected training components. Leave-two-out loses 2.7% where a
  removal disconnects a block. Half-splits usually fragment the blocks (recall
  0.123), and precision stays 1. The learner does not guess beyond its evidence.
- **The refutation check passes.** Perfect generalization occurs on blocks but
  never on random (0% in every split and core size). The random relation is one
  connected component, so merge supports almost everything: recall 0.917 at
  precision 0.398. High recall alone would have been misleading.
- **Overlap is not represented.** Overlapping rectangles connect, so merge learns
  their bounding rectangle: precision exactly 20/30, as predicted. Join can keep
  two core modes for a2 and a3, but recovers the two rectangles exactly in only
  about 4% of runs. Overlapping categories need either negative evidence or a rule
  that can split a component, and this study had neither.
- **Capacity.** Below the needed core size, forced reuse fuses unrelated
  components. On blocks leave-one-out, merge precision falls to 0.848 at n = 3 and
  0.577 at n = 2.
- **Local versus consolidation.** Join is local but order-dependent: on blocks
  leave-one-out, 79% of orders give perfect generalization. Merge is order-free and
  exact, but needs a nonlocal operation (merging core modes). This is the
  consolidation question of open problem OP-19, in miniature.
- **Core participation.** For merge, every cue lives in one core mode, so clamping
  that mode leaves the response unchanged. The core matters as a *shared latent
  category*, not as dynamic selection. Responses that need the core to move
  between modes appear only for join, on overlap and random: up to 0.58 cues per
  run.

## What this does not establish

- **This is learned category completion, not learned graded mediation.** The study
  returns to whole-mode moves and allows equal-energy attractor sets. The unit-level
  recovery and basin results of the graded interface do not transfer automatically.
  Two-mode graded modules cannot even express 6-mode satellites.
- **There is no architectural advantage for the core yet.** The controls show an
  advantage over memorization, but their zero recall is built into their rules.
  By P-merge, a connected-component baseline with no explicit core reproduces merge's
  responses exactly. The core's possible advantage must come from what components
  cannot do. The obvious candidate is overlapping compatibility: keeping a mode in
  several categories without merging away meaningful incompatibilities. That is the
  limitation this study makes concrete, and the next substantive challenge.

## Scope

- Mode level only, with hand-declared learning rules, three declared relations,
  and positive examples only.
- 6×6 satellite modes and core sizes up to 8.
- Not tested: unit-level transfer (graded interfaces exist only for two-mode
  modules), negative evidence, splitting components, more satellites, and noise.
- The merge rule is nonlocal by design and is not proposed as SRI's plasticity
  law.

Reproduce with `.venv/bin/python -m experiments.learned_compatibility` and
`.venv/bin/python -m unittest discover -s tests -v`.
