# Relations as objects: learning what a relation means

**Status: design note, 2026-11-06. Not a contract; nothing run.** It records the user's direction after
[type meanings v1](type-meanings-v1.md) and proposes a first test for agreement.

## User direction (2026-11-06)

- The recurring problem is understanding what a relation means. Equality was the same problem, is-a is the same, and
  more will turn up. Without clear semantics the system will not generalize them.
- **No fixed starting vocabulary** of relation properties (no built-in "symmetric", "transitive", "inverse").
- **Relations among relations are learned.** Relations need to be treated as objects.

## The diagnosis this answers

| Problem | What the relation means | What the system did |
|---|---|---|
| Equality ([diagnostics v2](equality-diagnostics-v2.md)) | substitution; transitive | `eq` became "within about one": not transitive |
| Is-a ([type meanings v1](type-meanings-v1.md)) | inheritance; chains through subkind-of | learned "moo is-a sound", never used it |
| Applicability (type meanings v1) | holds only between certain kinds of thing | per relation; 0.35–0.41 on relations without examples |
| Composition ([critic check](composition-critic-check-v1.md)) | grandparent = parent then parent | needed an external critic |
| Symmetry, inverse, transitivity ([everyday relations v2](everyday-relations-v2-results.md)) | e.g. lives-with is symmetric | partial: 0.67–0.85 |

In each case the relation is a code fitted to its examples, and nothing ties it to other relations. The
[relational expression design](relational-expression-core-design.md#relations-among-relations-and-what-is-supplied-versus-learned)
already warned that "a relation name alone does nothing": its consequences need a definition or learned behaviour.
The user rules out the definition, so it must be learned behaviour.

## What "relations as objects" means here

1. **One space.** Relations get codes in the same space as entities, and the reader takes the relation as an object
   code. Today relations have their own tables (`R_I`, `R_L`, 16 dims) separate from entities (`E_I`, `E_L`).
2. **Relations can be arguments.** Statements may take relations as subject or object:
   `parent-of ρ₁ child-of`, `lives-with ρ₁ lives-with`, `parent-of ρ₂ ancestor-of`, `parent-of ρ₃ person`. The
   relations among relations (ρ₁, ρ₂, …) are arbitrary tokens, like every other relation. No name or property is
   given; what they mean is whatever their examples make them. In the world's actual structure: ρ₁ holds for the
   seven inverse pairs, and a relation with itself is symmetry; ρ₂ is implication; ρ₃ and ρ₄ are subject and object
   types against the categories of type meanings v1. No vocabulary is needed for symmetry, because "R ρ₁ R" carries it.
3. **Meaning is learned only where using it is required.** Type meanings v1 showed that a fact the system can store
   but never needs is not used: the categories were learned and ignored. So the training has to contain episodes
   where a relation is introduced mainly through its relations to other relations, and instance questions about it
   can only be answered by using them. This is learning to learn new relations, the same pressure principle as
   [constraints are pressure](abstraction-learning-principles.md).
4. **A relation's meaning is then its place among other relations and categories, plus how the reader uses that
   place.** There is no table of properties.

This stays within the program's constraints: universal (no relational form is built in, and the relations among
relations are learned from examples), purely neural (codes and readers; the consistency critic may check learned
patterns), and no rules supplied at query time.

## Prior work (from memory, to be checked before a contract) **[familiar]**

- **Reification:** treating relations as things that statements can be about. RDF reification, Wikidata properties
  as items, OWL punning. Standard for representation, but these systems give the meaning by axioms. The user's
  direction excludes that.
- **Relations as learned objects in neural models:** Hinton's family trees (1986) gave relations input units like
  people. Linear relational embedding (Paccanaro and Hinton 2001) learned relations as matrices and could represent
  relations of relations. Tensor product representations (Smolensky 1990) bind roles and fillers.
- **Learning relational predicates that later become arguments:** DORA (Doumas, Hummel and Sandhofer 2008) learns
  structured relations from examples, and its learned predicates can then be arguments of higher-order relations. This
  is the closest model of what the user describes. Holyoak and colleagues argue that human relational reasoning needs
  relations reified as objects of thought.
- **Learning to learn relational structure:** relational schemas and learning-to-learn in hierarchical Bayesian models
  (Kemp, Tenenbaum and colleagues, 2006–2010); meta-learning for compositional generalization (Lake 2019,
  meta-seq2seq). Both find that abstract structure is learned when many episodes share it.
- **Known failure modes:** too few relation pairs to learn a relation among relations from. The world has 7 inverse
  pairs and 6 symmetric relations, which is too few episodes. Meta-facts absorbed into codes without being used (the
  is-a failure). Memorizing the training pairs instead of the relation among relations.

## A first test (proposal)

**Question:** given a relation it has never seen instances of, plus its relations to known relations, does the system
answer instance questions correctly? For example: told `R_new ρ₁ parent-of`, does it get `R_new(b, a)` from
`parent-of(a, b)`?

**World.** The world's real relations, plus **synthetic relations** over the existing entities, so there are enough
episodes. Each synthetic family is a relation with an inverse partner, a symmetric relation, or a relation with an
implied broader one, all over chosen categories. Synthetic relations are generated from the world's facts (for example
"R holds where colour-of and has both hold"), so they are real relations over real entities, not noise.

**Training.** Instances of training relations, the relations among relations for them, and **introduction episodes**:
a relation presented with its ρ-facts and k instances (k = 0, 1, 4), then asked about further instances.

**Test (held-out relations):** inverse direction, symmetry, implied facts and applicability (via ρ₃/ρ₄ against the
categories, read through the meaningless output). Each with k = 0, 1 and 4 instances.

**Controls:**
- the same held-out relations taught with k instances and **no** ρ-facts (what examples alone give);
- ρ-facts with the ρ labels **shuffled** (do the learned relations among relations carry the meaning, or only the
  extra exposure?);
- ρ-facts **removed at test** after training with them (is the meaning used, or only absorbed into codes?);
- the relation taught with all its instances (the upper bound).

**Architecture change:** a shared code table for entities and relations, with the reader's relation slot reading
that table. Everything else stays as in conflict search v1's phase 1, including the applicability output.

## Open for the user

1. **Synthetic relations:** acceptable? Without them there are too few relation pairs to learn from.
2. **First relations among relations:** inverse (with symmetry as self-inverse), implication, and subject/object
   types. Composition (grandparent = parent then parent) relates three relations, and would come later.
3. **The test scale:** about 40–60 synthetic relations, about 10 held out, 4 seeds. To be costed in the contract.
