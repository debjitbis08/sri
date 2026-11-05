# Everyday relations v2: relation properties, open world, relation-conditioned pairs

Status: **draft contract, 2026-11-05. Not frozen; awaiting user review.**

Follows [everyday relations v1](everyday-relations-v1-results.md). User direction (2026-11-05): run all three follow-up
tests, and first fix the v1 weaknesses the user's questions exposed:

1. **Fast mapping was not a fair test.** The v1 `dax` was fitted from true facts only, with the reader frozen. Its code grew
   to about 4 times a normal animal code (`two`: 1.5 against 0.4) and left the region the reader knows. By contrast, the
   held-out lizard kept only "is a reptile", "is an animal" and the matching false `is-a` facts, but its code was learned in
   training: it got reptile properties at 0.99.
2. **Inverse was taught on one relation pair only** (parent/child, tree A, 12 people). Nothing presented "the reverse of"
   as something shared across relations.
3. **"Opposite" was a list of 13 unexplained pairs.** The adjectives appeared in no other fact, so there was nothing to
   understand. For one-order pairs, the training rows said "cold is opposite to nothing", which contradicted the test.

All domains are trained together, inputs stay one-hot, and there are no sentences or plurals, as in v1.

## Questions

1. **Open world (test A).** Is v1's "untaught means false" caused by closed-world training, in which every untaught pair
   is labelled false? Or is it the reader?
2. **Relation-conditioned pair space (test B).** If G sees the relation, `q = G([c_a, c_b, r])`, does the two-space core
   catch up with `flat` on composition and fast mapping?
3. **Relation properties across relations (test C).** When several relations share a property (symmetry, inverse,
   transitivity), does a relation with incomplete examples inherit it?
4. **Meaning.** Once adjectives mean something (they apply to things at the ends of a taught scale), is `opposite`
   inferred for adjective pairs whose opposite was never taught? Does a new word with a known meaning get the right
   opposite?

## Part 0: fair fast mapping on v1 checkpoints (no training)

This answers the user's question 1 directly, before v2 trains anything. On the 20 v1 real-world checkpoints, `dax` and
`fep` are refitted three ways (k = 1, 2, 4 true facts as in v1):

| Fit | Facts | Code constraint |
|---|---|---|
| `v1` | true facts only | weight decay 1e-4 (v1 reproduction) |
| `row` | the true facts plus the matching false `is-a` facts (not a mammal, fish, …), the row the lizard had | weight decay 1e-4 |
| `row-prior` | as `row` | Gaussian prior on the code: mean and covariance of the trained entity codes, penalty ½ Mahalanobis² |

Read as in v1: bird or fish majority properties, class rank, code norm against the mean entity norm.

**Prediction [derived from the lizard result]:** `row-prior` reaches the T1 level (≥ 0.9) in `two` and `flat`; `row`
alone is in between. If `row-prior` stays low, the frozen reader, not the fit, is the limit.

## The world (v2)

v1's world (`src/everyday_world.py`) with these additions. The tables are the specification; the generator stays
deterministic.

**Scales give adjectives meaning.** 6 dimensions, each with a comparative relation over 6–8 things, and two pole
adjectives:

| Dimension | Comparative (and its inverse) | Things in order | Poles |
|---|---|---|---|
| size | `bigger-than` / `smaller-than` | v1's 12 animals | `big`, `small` |
| temperature | `hotter-than` / `colder-than` | sun, fire, soup, tea, milk, water, ice, snow | `hot`, `cold` |
| speed | `faster-than` / `slower-than` | cheetah, horse, dog, cat, duck, turtle, snail | `fast`, `slow` |
| weight | `heavier-than` / `lighter-than` | whale, elephant, horse, dog, cat, mouse, feather | `heavy`, `light` |
| loudness | `louder-than` / `quieter-than` | thunder, lion, dog, cat, bee, mouse, butterfly | `loud`, `quiet` |
| brightness | `brighter-than` / `darker-than` | sun, snow, sky, grass, chocolate, coal, night-sky | `bright`, `dark` |

- A new relation `is` holds (thing, adjective) for the top 2 things on each scale (pole 1) and the bottom 2 (pole 2): the
  sun is hot, snow is cold.
- `opposite` is the pair of poles of one scale. v1's other adjective pairs (wet/dry, up/down, day/night, happy/sad) are
  dropped, because nothing gives them meaning.

**More relations with each property:**

| Property | Relations |
|---|---|
| Symmetric | `opposite`, `spouse-of`, `sibling-of`, `same-colour-as` (from `colour-of`), `same-class-as` (from `is-a`), `lives-with` (shares a home) |
| Inverse pairs | `parent-of`/`child-of`, `baby-of`/`has-baby`, `eats`/`eaten-by` (animal eats animal: a small predator table), and the 6 comparative pairs |
| Transitive | the 12 comparatives, `is-a` (robin → bird → animal) |

About 230 entities and 36 relations. The exact counts are fixed when the generator is written and recorded at freezing.

## Held-out tests

Each test is held out from training; lists are fixed in the generator.

| Test | Held out | Follows from |
|---|---|---|
| **P1 inheritance, hierarchy** | v1's T1, T2 (continuity) | class facts |
| **P2 symmetry** | in each of the 6 symmetric relations, one order of a third of its true pairs (the other order trained) | the relation's own other pairs, and the other symmetric relations |
| **P3 inverse** | in each of the 9 inverse pairs, the second relation for a third of the entities (whole rows, so no contradicting false facts); the first relation is trained for them | the pair's other entities, and the other inverse pairs |
| **P4 transitivity** | for 2 of the 6 dimensions (temperature, loudness), both comparatives are trained on adjacent pairs only; the other 4 dimensions are trained on every pair | the other dimensions' complete orders |
| **P5 opposite from meaning** | `opposite` rows of 2 dimensions' poles (speed, weight), both orders, whole rows; their `is` facts and comparatives trained | the poles' meaning, plus `opposite` on the other 4 dimensions |
| **P6 composition** | v1's T7 (grandparent), and `same-colour-as` for 4 things whose `colour-of` is trained | parent ∘ parent; colour ∘ colour⁻¹ |
| **P7 not inferable** | v1's T9 (colour of 3 animals) and T1 `specific` | negative controls |

In closed-world training, a held-out query is never also trained as false. P2 holds out only the one order. P3 and P5
hold out whole rows, so v1's contradiction (cold opposite to nothing) cannot arise.

## Fast mapping and new relations (frozen network, `row-prior` fit from Part 0)

| New | Given (in order) | Read |
|---|---|---|
| `dax`, `fep` | as v1, with the false `is-a` row | as v1 |
| `zorp` | `bigger-than` against cat and dog, both orders | `bigger-than` and `smaller-than` against all 12 |
| `toasty` (a new word meaning hot) | `is(sun, toasty)`, `is(fire, toasty)`, and `is(snow, toasty)`, `is(ice, toasty)` false | `opposite(toasty, cold)` and the reverse; `is` for the other temperature things |
| `kim` | as v1 | as v1 |

New relations (only the relation code fitted, 4 or 16 examples): `smaller-than-v` (a held-back copy of `smaller-than`, as
v1) and `same-home-as`.

## Arms

| Arm | Scorer | Training |
|---|---|---|
| `two-closed` | v1 `two`: `S([G([c_a, c_b]), r])` | closed world (v1) |
| `two-open` | as above | open world |
| `tworel-closed` | `S([G([c_a, c_b, r]), r])`; G's input grows by 16 | closed |
| `tworel-open` | as above | open |
| `flat-closed`, `flat-open` | v1 `flat` | closed / open |
| `complex-open` | ComplEx (comparison only, built-in form) | open |
| shuffled: `two-open`, `tworel-open`, `flat-open` | v1's shuffle (independent head and tail permutations per relation) | open |

D = 32 for all; `flat`'s width is re-matched to `tworel`'s parameter count (`two`'s count is recorded). Seeds 410–413.

**Open world:** every trained true fact, plus negatives drawn fresh each step. For each positive, 4 corrupted triples
(head or tail replaced uniformly from the relation's head or tail set). A drawn negative that is a known trained fact is
redrawn. One that is a held-out test fact is not: the learner does not know it, as in standard negative sampling. The
rate at which test facts are drawn as negatives is recorded.

**Closed world:** as v1, with every trained query labelled.

Both regimes: full batch of positives (with their sampled negatives in open world), balanced per relation, every
relation weighted equally, Adam lr 3e-3, weight decay 1e-4, 3,000 steps.

## Reads

- Per test: balanced accuracy at threshold 0 and AUROC, mean and per seed. In open world the threshold is still 0.
  Calibration is part of the question.
- **False-positive rate on trained false queries** (closed-world labels) for open-world arms. Open-world training can
  accept everything, and P-tests alone would not show it.
- P2–P4 per relation, to see whether every relation gains the property or only some.
- P4 by rank distance.
- Relation space: v1's probe and analogy reads. For `tworel` q contains r by construction, so its probe is not
  evidence of abstraction. Its analogy read uses pairs of different relations with the same property (does a symmetric
  pair's nearest q come from a symmetric relation?).
- Learning order (v1 prediction 2, retried): per-relation accuracy every 25 steps for the first 500 steps.

## Predictions

1. **[Dong et al. 2014 local closed world; v1 diagnostics; familiar]** Open world turns v1's "ranked but rejected" into
   accepted: P2 and P3 balanced accuracy rise in `two` and `flat` by ≥ 0.15 over closed world. AUROC changes by less
   than 0.05. The false-positive rate on trained false queries rises, but stays ≤ 0.10.
2. **[TransH/TransR, relation-specific views; derived from v1]** `tworel` reaches `flat` on P6 composition and fast mapping
   (within 0.05). `two` stays below.
3. **[Lake and Baroni 2023, meta-learning for compositionality; analogy]** Properties transfer partly. P2 and P3 in
   open-world arms are above shuffled by ≥ 0.15, and better than v1's single-relation T5/T6. P4 transitivity stays
   weak (≤ 0.65), because the reader has no shared order unless the codes of different scales align. With 6 relations
   per property, this world is far from MLC's thousands of episodes.
4. **[Toutanova and Chen 2015; analogy]** P3 is the easiest property: in knowledge-base benchmarks the inverse of a held-out
   triple is trivially exploited by learned scorers. P3 ≥ P2 ≥ P4.
5. **[derived]** P5 opposite from meaning is a two-step inference (the poles of the same scale). It is above shuffled but
   below P2 in every arm. `toasty`'s opposite (cold) is found at k = 4 in at most half the seeds.
6. Part 0 prediction as stated there.
7. P7 at chance; shuffled arms at the `prior` level.

## Prior work

Searched 2026-11-05 (queries: "Toutanova Chen observed versus latent features inverse relations FB15k"; "Lake Baroni
2023 meta-learning systematic generalization"; "TransH TransR relation-specific projection"), plus v1's search.

- **Open world and negatives.** Negative sampling by corrupting a head or tail is the standard KB training regime (TransE,
  Bordes et al. 2013) **[familiar]**. The Knowledge Vault's local closed-world assumption (Dong et al. 2014) **[familiar]**
  treats only missing values of a known (subject, relation) as false. v1's closed world labelled every untaught pair false,
  which is the regime these methods avoid.
- **Inverse leakage.** [Toutanova and Chen 2015](https://www.microsoft.com/en-us/research/publication/observed-versus-latent-features-for-knowledge-base-and-text-inference/)
  **[search excerpts]**: FB15k/WN18 contain many reverse triples, and a simple observed-features model rivals latent models.
  FB15k-237 removes them. [ConvE (Dettmers et al. 2018)](https://arxiv.org/pdf/1707.01476) **[search excerpts]** reports
  the inverse rule baseline. Implication: P3 is known to be learnable by scorers when the inverse fact for the same pair is
  trained.
- **Relation-specific pair views.** [TransH (Wang et al. 2014)](https://mlanthology.org/aaai/2014/wang2014aaai-knowledge)
  and TransR (Lin et al. 2015) **[search excerpts]** give each relation its own projection of entities. `tworel` is a
  learned, non-linear version of a relation-conditioned view, without their built-in translation.
- **Learning properties from many relations.** [Lake and Baroni 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620072)
  **[abstract]**: meta-learning over a stream of few-shot compositional tasks gives human-like systematic
  generalization. Test C is the same idea with very few tasks (6–12 relations per property) and no episodic training.
  It is expected to transfer weakly.
- **Fair fast mapping.** A Gaussian prior on a new code from the existing codes is a standard Bayesian treatment of a new
  embedding **[derived]**. The program alternative (no prior, more facts) is the `row` fit.
- **What this adds:** whether the program's core, without built-in forms, gets relation properties from a handful of
  same-property relations, and whether v1's failures were the training regime or the architecture. These are internal
  questions; nothing here is new to the field.

## Scale

- Open-world training costs O(positives × negatives) per step instead of O(all queries). It scales with facts, not with
  entities². Closed world does not scale and is kept only as the v1 comparison.
- `tworel`: G's input grows by the relation-code width; q is computed per (pair, relation), so one pair is no longer
  coded once for all relations. That costs R times more G calls when one pair is read under all R relations.
- A new relation adds one code; a new entity one code. The `row-prior` fit needs the entity-code mean and covariance
  (D + D² numbers, kept up to date incrementally).

## Computation (declared)

- Part 0: 5 arms × 4 seeds × 2 entities × 3 k × 3 fits = 360 fits of 500 steps on one code: about 2 CPU min.
- v2 main: 10 configurations × 4 seeds = 40 runs. World about 2× v1's queries, so about 5–8 CPU min per run (v1: 3.6),
  including the finer early logging. Total ≈ 4–5 CPU h, about 35 min wall on 8 workers. No GPU.
- Stored: reports and summaries; checkpoints in `runs/` (about 120 kB each, not tracked).
- No tuning: the open-world negative ratio (4) and the prior strength (1) are fixed here.

## Open decisions for the user

1. **Scales.** Are the 6 dimensions and their things the right concepts? Should colours be a scale too (light/dark is
   already brightness)?
2. **Synthetic relations.** Test C would be stronger with many more relations per property, which would mean adding
   abstract relations (random symmetric relations over the existing entities) to the real ones. Should v2 stay with
   everyday relations only, as drafted?
