# Relations as objects v1: learning a new relation from its relations to known ones

**Contract, written 2026-11-06 before the run.** It follows the [design note](relations-as-objects-notes.md). The
user's direction: no fixed vocabulary of relation properties; relations among relations are learned; relations are
treated as objects. The user agreed to the note's open points ("Ok", 2026-11-06: synthetic relations, the first
relations among relations, the scale) and asked for the study to be completed overnight. The design is mine, not
externally reviewed. Runner `experiments/relations_as_objects.py`; reports
`reports/relations-as-objects-v1{,-summary,-runtime}.json`.

## Question

Given a relation the system has never seen, told only how it relates to known relations (and to categories), does
the system answer questions about its instances? Does that need a link from what is said about a relation to how it
behaves? Does merging entity and relation codes into one space hurt?

## World

- **Base:** the everyday world (`src/everyday_truth_v2.py`) with the category is-a facts of
  [type meanings v1](type-meanings-v1.md) (moo is-a sound, …). Individual T/F facts of the training split of seed s
  (`conflict_search.splits`). Kind quantities, fill-the-tail and connectives are left out: they do not bear on the
  question.
- **Synthetic relations (62), generated once (world seed 2026)** over the categories with at least 5 members
  (person, animal, object, property, sound, colour, body-part, ability, food, substance). Each holds or not for every
  pair of its domain × range, labelled T or F:
  - 12 **inverse families:** S over A × B (each pair true with p = 0.25) and its partner S′(b, a) = S(a, b);
  - 8 **synonym families:** S and S₂ with the same extension;
  - 8 **implication families:** broad Sᵦ (p = 0.4) and narrow Sₙ ⊂ Sᵦ (each true pair of Sᵦ kept with p = 0.5);
  - 6 **symmetric relations** over A × A (p = 0.25, no self-pairs).
- **Relations among relations: five arbitrary tokens** (ρ₁ … ρ₅). Their truth is computed from the extensions, never
  named or built in:
  - ρ₁(X, Y): Y's true pairs are X's reversed. That covers inverse pairs; a symmetric relation is ρ₁ with itself;
  - ρ₂(X, Y): X's true pairs are a subset of Y's (X ≠ Y);
  - ρ₃(X, Y): same true pairs (X ≠ Y);
  - ρ₄(X, C) / ρ₅(X, C): category C is among X's subject / object categories (from its taught facts).

  ρ₁–ρ₃ are labelled T or F for every ordered pair of relations with at least one true fact (real and synthetic).
  ρ₄–ρ₅ cover every relation × category, for categories that are entities.
- **Relations as objects:** every relation, including the ρs, gets an **object code** (an entity row) used when it is
  an argument. ρ-facts are statements whose subject and object are relation objects (or a category for ρ₄–ρ₅).

## Held-out relations (15), never seen in training: no instances, no ρ-facts, no meaningless examples

- **Real:** `child-of` (its inverse `parent-of` is trained) and `darker-than` (inverse of `brighter-than`).
- **Synthetic:** 4 inverse partners S′, 3 synonyms S₂, 2 broad Sᵦ (their Sₙ trained), 2 narrow Sₙ (their Sᵦ trained),
  2 symmetric relations. Each from a different family, chosen once.

## Model: logic route plus a plain intuition route

As conflict search v1's dual route (separate codes for intuition and logic, D = 32, relation codes 16, MLPs with 128
hidden), with the applicability output of type meanings v1. Logic's **operator code** for relation r (the relation
slot) depends on the arm:

| Arm | Operator code | What facts about a relation can do |
|---|---|---|
| `map` | R_op = f(E_L[obj(r)]), f an MLP (32 → 128 → 16) | shape its object code, which determines its behaviour |
| `nomap` | its own table R_L[r] (today's model) | shape its object code only; behaviour cannot change |
| `merged` | R_op = E_L[obj(r)] (one space; the relation slot is 32 wide) | the same code is object and operator |

Intuition keeps its own relation table in every arm.

## Training (3,000 steps, Adam lr 3e-3, weight decay 1e-4; seeds 450–453)

- **Facts:** world facts, category is-a facts, synthetic instances and ρ-facts, all as T/F statements. Logic is
  trained on support targets (T (1, 0), F (0, 1)); intuition on BCE. Each step samples 8,192 statements with weight
  equal per (relation, value) cell.
- **Applicability:** target 1 on every fact, 0 on meaningless examples: 50 out-of-type pairs per training relation
  with a category type (real permitted relations and synthetic). The term weighs the same as the fact term.
- **Introduction pressure:** each step, the instance facts of a random 30% of synthetic training relations are
  dropped from the batch, so those relations are learned that step only through their ρ-facts. This acts only through
  the object code (in `nomap` it has no effect on behaviour).

## Test: introducing a held-out relation

Everything trained is frozen. For each held-out relation h, new codes are fitted (500 Adam steps, lr 0.01, with the
Gaussian prior of the fast-map dax on each code). The fitted codes: h's object code in every arm; in `nomap` also its
own operator code; intuition's relation code for h from instances only. Fitting data, by condition:

- **`rho`:** h's ρ-facts against the trained relations and categories (T and F), plus k instance pairs;
- **`none`:** k instance pairs only;
- **`shuffled`:** h's ρ-facts with the ρ labels permuted (ρ₁ → ρ₃ → ρ₂ → ρ₁, ρ₄ ↔ ρ₅), plus k instance pairs.

k ∈ {0, 1, 4}, where a pair is one true and one false instance, drawn once per seed.

## Reads (logic route unless stated; means over held-out relations of a type and over seeds)

1. **Instance accuracy:** balanced accuracy (U counted wrong) on all h's labelled instances except the given ones.
   This is fully predictable from ρ-facts for inverse and synonym relations (and `child-of`, `darker-than`).
2. **Predictable subsets:** a held-out broad relation, T recall on pairs where its narrow relation is true; a held-out
   narrow relation, F recall where its broad relation is false; a symmetric relation, accuracy on the reverses of the
   given pairs (k ≥ 1).
3. **Applicability:** the share of 300 out-of-type triples of h recognized as meaningless; the share of its in-type
   labelled pairs kept meaningful.
4. **Reference reads:** intuition fitted from the k instances; the trained relations' accuracy on their own instances
   (upper bound); world-fact accuracy per arm (the cost of merging).

## Predictions

1. **[derived, exact]** In `nomap`, `rho` = `none`: h's behaviour does not depend on its object code. Any difference
   is a bug.
2. **[derived]** In `map`, `rho` beats `none` at k = 0 by ≥ 0.2 balanced accuracy on inverse and synonym relations,
   and reaches ≥ 0.7 on them.
3. **[derived]** `shuffled` is below `rho` by ≥ 0.15 on those relations: the meaning is carried by which ρ is stated,
   not by extra exposure.
4. **[derived]** Applicability from ρ₄–ρ₅ alone (`map`, `rho`, k = 0) recognizes ≥ 0.7 of h's out-of-type triples.
   Held-out relations reached 0.35–0.41 in type meanings v1, without type statements.
5. **[derived]** Real held-out relations (`child-of`, `darker-than`) do as synthetic inverses within 0.15. The ρ₁ link
   is learned across 12 + 7 inverse examples plus the symmetric ones.
6. **[the user's concern]** `merged` is below `map` on world-fact accuracy (in-split test facts) by ≥ 0.02, or below it
   on held-out relations. Either would show the cost of one space.
7. **[derived]** Implication subsets are partial: they gain over `none`, but less than inverse and synonym.

## Prior work (from memory, 2026-11-06, **[familiar]**)

- **Few-shot relation learning in knowledge graphs:** GMatching (Xiong et al. 2018) and MetaR (Chen et al. 2019)
  meta-learn to represent a new relation from a few instance pairs. That is the `none` condition with k > 0. Known
  limit: no use of what is known about the relation besides its examples.
- **Zero-shot relations from descriptions:** relation embeddings generated from text descriptions (Qin et al. 2020,
  ZSGAN) and **hypernetworks** (Ha et al. 2016), which produce one network's parameters from another's input. The `map`
  arm's f is a hypernetwork from a relation's object code to its operator, and the ρ-facts are the description, in
  the system's own relational form instead of text.
- **Reified relations and higher-order relations:** DORA (Doumas, Hummel and Sandhofer 2008) learns predicates that
  later become arguments. Linear relational embedding (Paccanaro and Hinton 2001) represents relations as learned
  objects. Hinton's family trees (1986). Ontology languages give inverse, sub-property and domain/range as axioms. Here
  they are learned tokens.
- **Known failure modes:**
  - meta-facts absorbed into codes without being used (the is-a result of type meanings v1), addressed by the map and
    the introduction pressure;
  - too few relation pairs (addressed by synthetic families, 62 relations);
  - synthetic relations being easier than real ones (read 5 checks this on `child-of` and `darker-than`);
  - the fitted code drifting off the trained distribution (addressed by the Gaussian prior, as for the dax).
- **Cheaper alternative:** an explicit inverse/synonym operator (transpose, copy) would solve inverse and synonym
  relations exactly. It is excluded as a built-in relational form (user direction, no fixed vocabulary).
- **What this adds:** whether relations among relations, learned from extensions as ordinary facts about relation
  objects, can introduce a new relation; and whether a learned object-to-operator map is what makes that possible.

## Scale

A new relation costs one object code and a fit on its ρ-facts. ρ-facts grow with relations² for ρ₁–ρ₃; at scale they
would be sampled, as here, not enumerated. The map f is shared by all relations, so it grows with nothing. More
relations give more examples of each ρ, which is what the meta-learning needs.

## Computation (declared)

About 40 k statements (17 k world facts, 1.6 k is-a, about 20 k synthetic instances, about 31 k ρ-facts, mostly F),
8,192 per step. Training 3,000 steps: estimated 3–6 CPU min per run. Test fits: 15 relations × 3 conditions × 3 k ×
500 steps on a few hundred statements, about 1–3 CPU min per run. 12 runs (3 arms × 4 seeds) on 6 workers: about
20–40 min wall. Storage: one checkpoint per run, not tracked. No tuning: every constant is fixed here. Resumable per
run with `experiments/durable.py`.

## Commands

    EV_WORKERS=6 .venv/bin/python -m experiments.relations_as_objects       # rerun to resume
    .venv/bin/python -m experiments.relations_as_objects summary
