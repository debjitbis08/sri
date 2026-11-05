# Taught truth v1: one predictive reader, truth as learned values

Status: **draft contract, 2026-11-06. Not frozen; awaiting user review.** Stage 1 of 2. Stage 2 is the multi-step loop
(`loop-gen`, [multistep-reader-v1](multistep-reader-v1.md), to be revised) on top of this reader.

User direction (2026-11-06):
- The loop's reader must be generative (`loop-gen`).
- When combining g and P there must be **one reader that can judge as well as predict**.
- **Truth has been built in, not taught.** The system must be taught an algebra of truth values: T, F, don't know, and
  others such as always and sometimes.
- Chained questions as their own relations are accepted (higher-order relations; Stage 2).

## Alignment with the shared-output brief (Mīmāṃsā)

The [shared-output brief](shared-output-brief.md) set out the theory this stage must follow. It is inspired by
Mīmāṃsā's distinction among means of knowing (*pramāṇas*): perception, testimony, inference, comparison and postulation.

| Brief | Here |
|---|---|
| A claim is accepted because a ground supports it under conditions (*J supports p under C*) | v1 teaches the judgments; grounds and conditions (provenance) are the brief's stage 3, deferred |
| Separate positive and negative support give **supported, opposed, unknown, disputed** | the individual value set is these four (Belnap's four values, FDE), not Kleene's three |
| Always and never quantify over admissible cases; sometimes needs a qualifying case; "sometimes but not always" needs a counterexample; most needs population evidence; a default survives a counterexample to a universal | the kind values and their definitions below |
| A failed observation establishes absence only under suitable detection conditions | "opposed" is taught only where the world states absence. U is taught where the world is open. "Untaught" is never a value |
| A proof and repeated observation give different warrants | deductive necessity is left out of v1 (no derivations in this world); "always" here is empirical, over all members |

**[familiar; inspired by]** Mīmāṃsā's intrinsic validity (*svataḥ prāmāṇya*, Kumārila): a cognition is accepted unless
defeated by a conflicting cognition or a defect in its source. That is the spirit of the default link below (usually →
supported until an exception is known). It is an analogy that motivates the rule; the rule is not an implementation of
the doctrine.

## What is built in today, and what changes

In every reader so far:
- one output unit means "true", trained directly on labels, with a fixed threshold at 0;
- "untaught means false" came from the labelling (closed world).

Here:
- **One reader outputs a code** that fills the missing slot of a statement: either the tail ("what does blick name?") or
  the truth value ("can an owl swim?").
- **Truth values are learned codes,** like entity codes. The answer is the nearest candidate code.
- **The algebra is taught by examples:** the values, the connectives over them, and how kind-level statements bear on
  individuals. Nothing about truth is wired in, except comparing a predicted code with candidate codes. That comparison
  is the smallest consistency mechanism, and the statement format (which slot is missing) is given.

## The value algebra (sources and our choices)

| Statement type | Values | Established source | Our choice |
|---|---|---|---|
| Individual ("this robin can fly") | **supported (T), opposed (F), unknown (U), disputed (D)** | Belnap's four-valued logic (FDE), of which Kleene's K3 (T, F, U) is a part ([SEP: Many-Valued Logic](https://plato.stanford.edu/entries/logic-manyvalued/)); the brief's four judgments | D is taught only where the world has conflicting reports (below) |
| Kind ("birds can fly") | **always, usually, sometimes, never** | intermediate quantifiers ([Peterson](https://projecteuclid.org/download/pdf_1/euclid.ndjfl/1093634570)); generics ([SEP: Generics](https://plato.stanford.edu/entries/generics/)); the brief's evidence conditions | always: every member supported; never: every member opposed; sometimes: at least one supported and at least one opposed member; usually: sometimes, with more than half supported |
| Link: kind → individual | always → T, never → F, usually → T (default), sometimes → U | default logic, Reiter 1980 ([SEP: Non-monotonic Logic](https://plato.stanford.edu/entries/logic-nonmonotonic/)) | **ours**: the mapping, in particular "sometimes → don't know" |
| Connectives | `not`, `and`, `or` over T, F, U, D | Belnap's tables (on T, F, U they are K3's) | taught as statements: (T, and, U) → U; (T, and, D) → D; not D = D |

Left out of v1: grounds and their revision (the brief's stage 3); necessity in the definitional or deductive sense; time
("at night").
Teaching a second algebra (for example Łukasiewicz's) to test universality is a later step.

## The reader

Statements have four slots, (a, r, b, v); one slot is missing. `MASK` is a learned code for the missing slot.

```
q   = G([c_a, c_b or MASK, r])           (the tworel pair space of v2–v3, with the relation)
out = P([q, r, slot])                     (a code; `slot` says which slot is missing)
answer = the candidate whose code is nearest (cosine) to out:
   v missing  → candidates: the 8 value codes
   b missing  → candidates: all entity codes
```

- Training is cross-entropy over the candidates' cosine scores, with a learned temperature: the nearest-code readout,
  trained.
- Value codes are rows of the same embedding table as entities, so the connectives' statements (T, and, U) use them as
  arguments.

## The world

v3's world (`src/everyday_world_v3.py`), with statements generated by `src/everyday_truth.py`:

- **Individual statements:** every v3 query (a, r, b) with v = T if true, F if false (closed world, as v3).
- **Don't know (U), taught only where the world leaves it open:** colour statements for the 24 animals with no colour in
  the world. `colour-of(robin, c)` is U for every colour: nobody knows a robin's colour. U is never taught for
  untaught facts.
- **Kind statements:** for each of the 6 classes and each `has`, `can`, `lives-in`, `eats` value: (class, r, value) with
  the quantity value from the members. For example (bird, can, fly) → usually; (bird, has, feathers) → always;
  (mammal, can, swim) → sometimes; (fish, has, fur) → never.
- **Fill-the-tail statements:** (a, r, ?) for true individual facts, answer b (several true answers share the loss).
  This is the "predict" half of the reader that Stage 2's loop needs.
- **Disputed (D), taught only where the world has conflicting reports:** a small hand table of facts on which a child
  hears both answers, for example `is-a(whale, fish)`, `is-a(bat, bird)`, `is-a(spider, insect)` (spider added),
  `colour-of(sky, white)`. About 6 statements, all trained. In v1, D cannot be inferred: there are no sources to conflict.
  It is taught so that the algebra, and Stage 2's connectives, have it.
- **Connectives:** Belnap's tables for `not` (4 rows, with b a fixed `nil` entity), `and` (16) and `or` (16), all
  trained.

## Tests

| Test | Held out | Correct answer | Follows from |
|---|---|---|---|
| **S1 T/F as before** | v3's held-out individual tests (P1 inheritance, P2 symmetry, P3 inverse, P4 order, P5 opposite from meaning, G1 word order) | T or F | as in v3 |
| **S2 kind quantities** | a quarter of the kind statements | always / usually / sometimes / never | the members' individual facts (aggregation) |
| **S3 default to individuals** | P1's held-out individuals (robin, salmon, dog, bee, lizard, frog) | read two ways: world truth (T/F), and the default answer (always → T, never → F, usually → T, sometimes → U) | kind statements and membership |
| **S4 don't know** | (a) `colour-of` for 4 of the 24 colourless animals; (b) v3's P7 (colours of elephant, flamingo, crow: true in the world, but nothing tells the learner) | U for both | (a) the taught pattern "animals' colours are unknown"; (b) nothing; U is the learner's correct state |
| **S4 control** | inferable held-out facts (S1) | not U | U must not become an escape |
| **S5 new bird** | `dax`, given only `is-a(dax, bird)` = T (code kept among the known codes, no exclusions) | feathers T (always), fly T (usually), swim U (sometimes), fur F (never) | kind statements |

## Arms

| Arm | Reader | Taught |
|---|---|---|
| `unit` | v3 `tworel` with the built-in truth unit | individual T/F only (the v3 baseline) |
| `pred` | the predictive reader above | everything above |
| `pred-flat` | as `pred`, with `q = [c_a, c_b, r]` (no pair space) | everything |

`unit` cannot express U or quantities. It is compared on S1 and on S3's world-truth read.

Seeds 440–443, 12 runs. Full batch; every statement kind (individual, kind, fill-tail, connective) weighted equally;
within individual statements, balanced per relation and value. Adam lr 3e-3, weight decay 1e-4, 3,000 steps. D = 32;
value codes share the table.

## Predictions

1. **[derived]** Taught truth costs nothing on T/F: `pred` S1 within 0.05 of `unit`.
2. **[Peterson; generics; derived]** S2 kind quantities ≥ 0.6 (4-way; chance 0.25). Aggregating over members is a new
   kind of inference for this reader. "Always" and "never" are easier than "usually" and "sometimes".
3. **[Reiter; v1–v3 inheritance]** S3 default read:
   - always → T and never → F ≥ 0.9;
   - usually → T ≥ 0.8;
   - sometimes → U ≥ 0.5.

   On world truth, `pred` is within 0.05 of `unit`.
4. **[derived]** S4: (a) ≥ 0.8. (b) U ≥ 0.5: the learner says "don't know" where it cannot know, instead of guessing.
   The control: U on inferable held-out facts ≤ 0.15.
5. **[v2 addendum]** S5: the dax's four answers follow the default mapping in ≥ 3 of 4 seeds for always and never.
6. The connective tables are fitted exactly (they are taught in full; they matter for Stage 2).

## Prior work

Searched 2026-11-06 (queries on generics, intermediate quantifiers, many-valued logic and bilattices, non-monotonic
logic). The SEP entries are cited from familiar knowledge; their pages were not fetched in this session.

- **Many-valued logic:**
  - Kleene's strong 3-valued logic (K3) and Belnap's four-valued logic (FDE) of which it is a part. Fitting shows K3 and
    Kripke-style semantics as sublogics of Belnap's simplest bilattice
    ([Fitting](https://id144254.securedata.net/melvinfitting/bookspapers/pdf/papers/BilSemLP.pdf), **[search
    excerpts]**). [Belnap's logic in computer science](https://arxiv.org/pdf/2503.20679) **[search excerpts]**.
  - Bilattices (Ginsberg 1988) **[familiar]** separate a truth ordering from a knowledge ordering. That is the two
    questions here: how the world is (quantity, T/F) and what is known (U).
- **Quantities:** Peterson's intermediate quantifiers (almost-all, most, many, few), with a 5-quantity syllogistic.
  [Iero, Third and Piwek](https://arxiv.org/pdf/1805.08707.pdf) **[search excerpts]** give a formal system that subsumes
  it.
- **Generics:** "birds fly" tolerates exceptions; its truth conditions are contested (how many is enough; normality
  accounts) ([Cohen, Genericity](https://philarchive.org/rec/COHG-5), **[search excerpts]**). Our ≥ ½ threshold for
  "usually" is a choice, not a settled semantics.
- **Default reasoning:** Reiter's default logic (1980) and preferential semantics **[search excerpts; familiar]**. A
  conclusion can be withdrawn when an exception is learned (non-monotonic).
- **Neural many-valued logic:** Logic Tensor Networks (already in [the core design note](relational-expression-core-design.md))
  build fuzzy truth into the network's semantics. Here the values and connectives are taught, not given.
- **Predict and judge in one network:** masked-slot prediction, as in masked language models and in KB models that score
  a missing argument by ranking all candidates. The candidate ranking with a learned temperature is InfoNCE-style
  training **[familiar]**.
- **What this adds:** truth values, their connectives and the kind–individual link as taught relations in the program's
  core, with "don't know" used only where the learner cannot know. The value algebra combines established parts with
  our own link rule.

## Scale

- Value codes: 8; a new value costs one code.
- Answering a truth question compares one output with 7 codes. Filling a tail compares with N entity codes (cached;
  one dot product each). One reader call per question.
- Kind statements are one per (kind, property). Aggregation over members is learned, not computed at reading time.

## Computation (declared)

- About 17k individual statements, 150 kind statements, 6 disputed statements, 2k fill-the-tail statements and 36
  connective rows.
- Per step: about 20k reader calls plus about 2k × N tail comparisons. Similar to v3 per run (about 5 CPU min).
  12 runs, about 10 min wall on 8 workers.
- Stored: reports; checkpoints in `runs/` (not tracked). No tuning.

## Open decisions for the user

1. **The threshold for "usually".** More than half the members, as drafted (Peterson's "most", and the brief's population
   evidence)? Or higher, such as ≥ ¾, closer to "almost all"?
2. **"Disputed" without sources.** v1 teaches D only on a few hand-picked contested facts, because learning *when* to
   say D needs sources (the brief's stage 3). Keep D in the algebra now, as drafted, or leave it to stage 3?
3. **"Sometimes → don't know" for an individual.** That is our link rule: if only some birds swim, a new bird's
   swimming is unknown. Is that the reading you want, or should "sometimes" give F by default?
