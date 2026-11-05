# Everyday relations v3: few "no"s, and two-word phrases

Status: **draft contract, 2026-11-05. Not frozen; awaiting user review.**

Follows [everyday relations v2](everyday-relations-v2-results.md). User direction (2026-11-05):
- Fix the open world with option 1: false facts only where the answer is known.
- Add a phrase test ("dog runs" against "runs dog") with its four parts: untaught word pairs, a new word known only by
  meaning, a new word known only from use, and grammatical-but-silly phrases.
- Fast mapping must not be given exclusions ("a bird cannot be a fish" must be learned).

## What "option 1" means here

v2's closed world already never trained the held-out facts. What made v1 reject everything new was the opposite
pressure: every other false pair was taught as false, so untaught came to mean false. v2's open world failed because its
random negatives were drawn, in a small world, from the held-out facts themselves.

v3's **few-"no"s regime** is positive–unlabelled learning, as a child hears it:
- the true facts are taught;
- a **fixed** set of false facts is taught too, chosen once before training: for each trained true fact, one corruption
  of its head or tail that is known false;
- every other pair is **never mentioned**: held-out test facts and most false pairs alike.

The learner cannot tell a held-out fact from an unmentioned false pair. Mentioned negatives are never held-out queries,
because a teacher says "no" only to what is false and not on the test. About 1,200 false facts are mentioned, against
about 11,000 unmentioned false pairs. So the held-out facts are a small minority of the unmentioned pairs, and leaving
them out of the negatives does not mark them.

**Read for this regime:** the P- and G-tests as before, plus the **false-acceptance rate on unmentioned false pairs**
(not held out). Accepting everything would pass the tests and fail this read.

## The world (v3)

v2's world (`src/everyday_world_v2.py`) with a phrase domain added. The changes go in `src/everyday_world_v3.py`.

**Words:**
- 19 noun words: v1's 12, plus whale, elephant, lion, mouse, ant, cheetah, turtle (so that size, speed, weight and
  loudness words have animals to describe);
- 5 verb words: `fly-w`, `swim-w`, `run-w`, `climb-w`, `sing-w`;
- 8 adjective words: big, small, fast, slow, heavy, light, loud, quiet.

Each word has `names` (word → animal, ability or adjective) and `is-a` (`noun`, `verb` or `adjective`).

**Two new relations over ordered word pairs (w1, w2):**

| Relation | True when | Examples |
|---|---|---|
| `good-phrase` | (noun, verb) or (adjective, noun): word order only | "dog runs" ✓, "runs dog" ✗, "big whale" ✓, "whale big" ✗, "dog eagle" ✗ |
| `makes-sense` | a good phrase whose meaning holds: the named animal can do the named action, or has the named adjective (`is`) | "eagle flies" ✓, "dog flies" ✗ (good order, false meaning), "big whale" ✓, "small whale" ✗ |

v2's `can-subject` is replaced by `makes-sense`.

## Phrase tests (G)

| Test | Held out | Follows from |
|---|---|---|
| **G1 untaught word pairs** | `good-phrase` for a quarter of all ordered word pairs, both orders | the category order learned on other pairs |
| **G2 known only by meaning** (in training) | for 2 noun words (`robin-w`, `shark-w`): every `good-phrase`, `makes-sense` and `is-a` query, as head or tail; only `names` is trained | the named animal is an animal, so the word is a noun |
| **G3 known only from use** (in training) | for 2 noun words (`cow-w`, `frog-w`): everything except 2 `good-phrase` facts with verbs (`cow-w run-w`, `cow-w swim-w`) and the reversed pair as false; no `names`, no `is-a` | the order pattern of nouns: other verbs, adjectives, and `is-a(·, noun)` |
| **G4 grammatical but silly** | `makes-sense` for a quarter of (noun, verb) and (adjective, noun) pairs | `names` and the animal's facts |

**G4's key read is the dissociation:** among held-out pairs in good order with false meaning ("dog flies"), the
fraction judged `good-phrase` true and `makes-sense` false.

**Fast mapping of new words** (frozen network, code kept among the known codes by the Gaussian prior, as in the v2
addendum; no exclusions given):
- **`blick-w`** (G2 as fast mapping), given `names(blick-w, owl)`. Read `good-phrase` with every verb and adjective in
  both orders, `is-a(blick-w, ·)`, and `makes-sense` with the verbs (an owl flies only).
- **`zib-w`** (G3 as fast mapping), given `good-phrase(zib-w, run-w)` and `good-phrase(zib-w, swim-w)` (k = 2), then also
  `good-phrase(run-w, zib-w)` false (k = 3). Read `good-phrase` with the other verbs and the adjectives,
  `is-a(zib-w, noun)` and `is-a(zib-w, verb)`.

Phrases longer than two words, plurals and agreement stay out (user decisions; sentences need a sequence reader).

## Carried over from v2

P1–P8 as in v2, with P8 (word selection) replaced by G4. The fast-mapping entities `dax`, `fep`, `zorp`, `toasty` and
`kim` and the new relations `smaller-than-v` and `both-fly` are kept. Every fast-mapping fit uses the prior and no
exclusions (`dax`/`fep` read the other classes as tests).

## Arms

| Arm | Regime | World |
|---|---|---|
| `two`, `tworel`, `flat` | closed (v2) | real and shuffled |
| `two`, `tworel`, `flat` | few "no"s | real and shuffled |
| `complex` | few "no"s | real (comparison only) |

That is 13 configurations × seeds 420–423 = 52 runs. D = 32; `flat` width as v2 (255); Adam lr 3e-3, weight decay 1e-4,
3,000 full-batch steps, every relation weighted equally, balanced within each relation. The mentioned negatives are drawn
once per seed. In the shuffled world they are drawn from the shuffled labels.

## Predictions

1. **[positive–unlabelled learning; v1–v2 diagnostics; familiar]** Few "no"s raises acceptance of held-out true facts:
   - P2, P3, G1 true-positive rate rises by ≥ 0.2 over closed world, and balanced accuracy rises.
   - The false-acceptance rate on unmentioned false pairs rises too, but stays below 0.15 for the MLP arms.
   - P5 stays at chance: its failure is cardinality, not the closed world.
2. **[Elman 1990; distributional learning of categories; familiar]** G1 ≥ 0.9 in every MLP arm and regime, above its
   shuffled control by ≥ 0.3. Word order between categories is a property shared by many pairs (19 × 5 + 8 × 19), like
   inheritance.
3. **[Pinker 1984, semantic bootstrapping; v1–v2 composition results; familiar]** G2 (category from meaning) is the
   weakest phrase test. It needs a word's category to follow from what it names (two steps). Expected 0.6–0.8 in
   training, lower for `blick-w`.
4. **[Gleitman 1990, syntactic bootstrapping; familiar]** G3 (category from use) is high (≥ 0.85) for the other verbs.
   It is lower for adjectives (the adjective-before-noun position was never seen with this word), and `is-a(·, noun)`
   follows.
5. **[Chomsky 1957, "colorless green ideas"; derived]** G4 dissociation ≥ 0.7 in `tworel` and `flat`: order and meaning
   are separate relations with separate codes. `two` is lower (q is shared across relations).
6. P1 inheritance, exclusivity in fast mapping and P7 behave as in v2 (closed world) in both regimes.

## Prior work

No new search; sources are **[familiar]** unless linked in v1–v2 or the shared survey.

- **Few "no"s is positive–unlabelled (PU) learning** (Elkan and Noto 2008). It is the setting of KB completion without a
  closed world, and of children who hear mostly true statements. The local closed-world assumption (Dong et al. 2014)
  is the KB version. A known failure is accepting too much, hence the false-acceptance read.
- **Word categories from distribution:** Elman (1990), "Finding structure in time": a simple recurrent network groups
  words into nouns and verbs from their contexts. G1 and G3 are the two-word version without recurrence.
- **Semantic and syntactic bootstrapping:** Pinker (1984) proposes children infer a word's category from its meaning (G2).
  Gleitman (1990) proposes they infer meaning and category from syntax (G3). Both are debated accounts of child language,
  not settled mechanisms. Here they name the two directions tested.
- **Grammatical but meaningless:** Chomsky (1957). G4 tests whether order and meaning are separable.
- **Order relations in pair scorers:** antisymmetric order is what v1 T4 failed and v2 P4 partly passed. `good-phrase`
  is asymmetric between categories, not a chain, so it is closer to inheritance than to transitivity.
- **What this adds:** whether the program's core learns category-level word order from pairs, and gets a word's category
  from its meaning or its use, without built-in categories. Nothing here is new to the field.

## Scale

- The few-"no"s regime costs O(true facts) per step, against closed world's O(all pairs). That is the regime that scales.
- A phrase is a pair, so it costs one query. Longer phrases need either pairs chained along the phrase or a sequence
  reader; both are out of scope.
- A new word costs one code; its category comes from the existing reader.

## Computation (declared)

- World: about 230 entities, 37 relations, about 19,000 queries (fixed at freezing).
- 52 runs at about 5–7 CPU min each (v2: 4.6, with a larger world now), about 5–6 CPU h. About 45 min wall on 8 workers.
- Fast mapping: 7 entities, about 20 fits per run, included in that time.
- Stored: reports; checkpoints in `runs/` (not tracked).
- No tuning: the 1:1 mentioned-negative ratio is fixed here.

## Open decisions for the user

1. **Ratio of "no"s.** 1 mentioned false fact per true fact, as drafted. More "no"s moves toward the closed world; fewer
   makes over-acceptance more likely.
2. **Words.** Are the 19 nouns, 5 verbs and 8 adjectives and the two phrase patterns (noun–verb, adjective–noun) right
   for a first phrase test?
