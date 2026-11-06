# Composition critic check v1

**Status: diagnostic on frozen readers, no walker training.** User direction, 2026-11-06: proof feedback needs three
levels (hop, composition, answer), and composition is the missing one. The consistency critic is the network to check
it, locally, step by step rather than over the whole chain. The question left open is whether a step can depend on
more than the previous one. Runner `experiments/composition_critic_check.py`; reports
`reports/composition-critic-check-v1{,-runtime}.json`. Inputs: the four phase-1 readers of conflict search v1.

## Question

Can a critic learned only from the true/false labels of questions tell valid proof programs from spurious ones? Which
form of memory does it need: the previous step only, a carried summary, or a look-back over earlier steps?

## What the critic sees

A **program** is a found path's sequence of steps, each a relation and a direction, between START and END markers,
together with the question's relation q. Hop truth is the existing two-route check and is not repeated. Every candidate
path runs over facts the frozen reader confirms, using permitted relations only (the 32 of
[the diagnostics](conflict-search-diagnostics-v1.md)). Relation and direction embeddings belong to the critic and are
learned. No proof definition is given to it; those are used only to score it.

## Three variants (same size class, 16-dim embeddings, 32 hidden)

At each step t (the END marker counts as a step), a local check gives c_t = σ(·). A program's score is ∏ c_t, so it is
checked as it grows and can be pruned at the first failing step.

- **`pairwise`:** c_t from [q, step t−1, step t].
- **`summary`:** c_t from [q, h_t, step t], where h_t = GRU over the earlier steps (starting from START).
- **`lookback`:** `pairwise` plus attention over steps 0…t−2, keyed by learned projections and queried from [q, step t].

## Training (multiple instance, labels of questions only)

- **Rows:** withheld-relation training questions (257 T, about 2,864 F per seed).
- **Candidates per row:** every distinct program of at most 2 steps that connects a to b in the confirmed graph (about
  470 per row). This is exhaustive at that length.
- **Loss:** a question's score is the maximum program score among its candidates, with binary cross-entropy against its
  label. On false questions every candidate is a negative. On true questions only the best has to pass. Each step
  takes 128 T and 128 F rows; 1,500 Adam steps, lr 3e-3. Seed per run.
- **Known gap:** some training questions need 3 steps (named-kind through `subkind-of`), so their valid program is not
  among the candidates and the maximum falls on a spurious one. This is recorded, not fixed.

## Reads

On the test questions of the withheld relations (as in the diagnostics), candidates are:

- every program of at most 2 steps;
- 30 sampled random paths of each length 3, 4 and 5 that end at b;
- the shortest valid proof in the confirmed graph, when there is one (the evaluation answers "if the walker proposed
  it, would the critic accept it?").

1. **Composition (program level):** AUROC of the critic's score, valid programs against spurious ones, by relation and
   by length; acceptance rates (score > 0.5) for each.
2. **Answer (row level):** question score = maximum over candidates. T recall, F recall and balanced accuracy at 0.5,
   and AUROC; with and without the valid proof among the candidates.
3. **Length:** ancestor programs `parent-of>`×k for k = 3–5 (trained on k ≤ 2), and kind-of `is-a>` `subkind-of>`×k.
4. **Counting probe:** scores of `parent-of>`×k, k = 1–5, under q = grandparent (valid only at k = 2) and q = ancestor
  (valid for all k).

## Expectations

- **[derived]** `pairwise` generalizes to ancestor 3–5 and to longer kind chains. `summary` may not, unless its state
  settles. `lookback` behaves like `pairwise`.
- **[derived]** On the counting probe, `pairwise` accepts grandparent for every k ≥ 2, because it cannot count.
  `summary` and `lookback` can reject k = 3, but only if training contains a negative that teaches it. The grandparent
  training rows contain no 3-step `parent-of` path, so all three may accept k = 3. That would be a data limit, not
  evidence about the architecture.
- **[familiar]** Learning acceptance from positive and negative strings is grammatical inference. First-order (bigram)
  constraints are identifiable from short strings and extend to any length; counting is not (Delétang et al. 2023).

## Computation (declared)

Candidate enumeration per row: one 66 × 246 × 66 product for 2-step programs (about 0.1 ms) plus 90 sampled walks;
storage of about 1.5 M short programs per seed. Training: 1,500 steps × about 120 k programs × a 32-unit network per
variant; estimated 1–2 CPU min per variant and seed. Total 12 runs (3 variants × 4 seeds) on 4 workers: about 5–10 min
wall. Measured times go in the runtime JSON. No tuning: the constants above are fixed before the run.
