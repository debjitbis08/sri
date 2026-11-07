# Amortized introduction v1: training a system to apply what is said about a new relation

**Contract, frozen 2026-11-07 before the run** (user: "Ok go ahead"). The user added a direction for what follows:
train the application on small examples and test it on larger ones. This run does not test that. Follows [relations as objects v1](relations-as-objects-v1.md) and the
[learning to apply relations](relations-application-notes.md) note. The user's direction (2026-11-07): learning the
theory is not enough, the system must practise applying it; try amortized introduction. The design is mine, not
externally reviewed. Planned runner `experiments/amortized_introduction.py`; reports
`reports/amortized-introduction-v1{,-summary,-runtime}.json`.

## Question

v1 taught relations among relations as facts, and they were learned (0.999–1.000) but never used: held-out relations
stayed at chance with or without them. In v1, no training step ever required a relation's behaviour to come from
what is said about it. Here every training step does that for some relations. An encoder builds a relation's code
only from its statements (its relations to known relations and to categories), and the code is scored on the
relation's instance answers.

1. Does a system trained this way answer instance questions about a held-out relation from its description alone?
2. Does it generalize the practice, or memorize which description belongs to which trained relation?
3. Does training on behaviour matter, or is it enough to regress the encoder onto the stored codes (the cheaper
   standard method)?

## World

The world is v1's, reused unchanged (`relations_as_objects.World`): the everyday world with category is-a facts, the
62 synthetic relations in inverse, synonym, implication and symmetric families, the five arbitrary tokens ρ₁…ρ₅ whose
truth is computed from extensions, and the same 15 held-out relations (`child-of`, `darker-than`, 4 inverse partners,
3 synonyms, 2 broad, 2 narrow, 2 symmetric). Training uses the same seeds 450–453 and splits.

**Change from v1:** ρ-statements are no longer trained as T/F facts on object codes. They are the encoder's input, its
description of a relation. Object codes are unused. Teaching the system to *judge* ρ-statements worked in v1 and can
be added back later. It does not bear on whether a description is used.

## Description of a relation

For relation X, the description D(X) is a set of statements (ρ, direction, partner, label):

- every T statement of ρ₁–ρ₃ between X and a trained relation, with X as either argument;
- ρ₁(X, X), with the partner given as a learned *self* token;
- all ρ₄/ρ₅ statements of X against the 11 categories, T and F (22);
- 64 F statements of ρ₁–ρ₃, sampled from X's F statements. They are resampled in every training episode and drawn
  once per seed and relation at test.

Partners are always trained relations, never held-out ones. A relation partner enters as its stored operator code
R_L[p] (16). A category partner enters as its entity code E_L[c] (32). Gradients flow into both, so the code space
can arrange itself so that descriptions are usable.

## Model

The model is v1's `nomap` arm: dual route, D = 32, relation codes 16, MLPs with 128 hidden units, plus the
applicability output. Every relation keeps a stored operator code R_L for ordinary training. The addition is an
**introduction encoder** Enc:

- **element:** φ([ρ embedding (8, learned per token); direction; label ±1; partner slot 48 = R_L[p] | E_L[c] |
  self, zero-padded]) → 64, an MLP with 128 hidden units;
- **pooling:** attention pooling with 4 heads and learned queries (Set-Transformer PMA, one seed per head). Keys and
  values are linear maps of the elements; the heads are concatenated to 64;
- **output:** an MLP 64 → 128 → 16, giving the operator code c_X.

c_X replaces R_L[X] wherever X is introduced. That covers both the T/F head and the applicability head. Nothing
stored for X (code, optimizer state) is on that path. The encoder has 35,016 parameters. It gets **no weight
decay**. In the smoke check, Adam's L2 term collapsed the encoder to a constant output within 100 steps in `episodes`.
Adam rescales the decay to full step size on weights whose answer gradient is still small. With no decay, the code
varies across relations (std 0.22 at 500 steps). Intuition keeps its own
table in every arm and is never introduced. It is a reference read.

## Arms (seeds 450–453; 3,000 steps, Adam lr 3e-3, weight decay 1e-4, as v1)

| Arm | Introduction episodes in training | What trains Enc |
|---|---|---|
| `episodes` | each step, a random 30% of trained relations with a description (about 27 of 91) are introduced: all their facts in the batch (instances, real world facts, meaningless examples) use c_X | their answers, as ordinary facts |
| `virtual` | as `episodes`, plus 16 **virtual relations** per step (below) | their answers |
| `regress` | the same 30% are chosen, but their facts keep R_L[X] | ‖c_X − R_L[X]‖² / mean ‖R_L‖², R_L detached (mimicking) |

**Virtual relations** are fresh relations with exactly known extensions. For a trained relation Q with labelled
pairs, *copy-of-Q* has the same extension and *inverse-of-Q* has it reversed. Their descriptions are computed from
their extensions by v1's ρ truth function, against trained relations, exactly as for real relations. Nothing about
copying or reversing is given to the model: a virtual relation is just another relation with a description and
instances. There are 78 for training (39 set-A relations × 2). They multiply the examples of "a ρ₁ or ρ₃ partner
determines the answers" from about 20 natural cases. They also break the match between a partner's identity and one fixed trained relation, which is the
memorization route (prior work). Each step samples 16 virtual relations (Q from set A below, copy or inverse with
equal chance) and 64 labelled pairs from each (32 T and 32 F, with replacement). Their loss weighs the same as the
ordinary fact loss.

**Split of trained relations for virtual relations.** Set B holds out virtual relations for testing:

- every member of a family that contains a held-out relation;
- `parent-of` and `brighter-than`;
- a random half of the remaining trained relations with a description (seed 2028).

Set A is the rest: 39 relations against B's 52 (seed 450). A virtual relation of Q ∈ A is dropped from training if
its extension matches any held-out relation's (ρ₃ or ρ₁ true against it). None matched in the smoke check. For a real
Q, a training virtual relation uses only Q's training-split facts, so no world test fact enters training through it.
Test virtual relations use the full extension.

## Test (everything frozen)

**Held-out relations** (15), under three conditions and k ∈ {0, 4} instance pairs (one T and one F each), drawn as in
v1:

- **`rho`**: c_h = Enc(D(h)). At k = 4, c_h is then fitted on the 4 pairs (500 Adam steps, lr 0.01), with a Gaussian
  prior centred on Enc(D(h)) and the precision of the stored codes;
- **`shuffled`**: Enc(D(h)) with the ρ labels permuted as in v1 (ρ₁ → ρ₃ → ρ₂ → ρ₁, ρ₄ ↔ ρ₅); k = 0 only;
- **`none`**: v1's fit of an operator code from the prior mean on the k pairs (k = 0 is the prior mean).

**Virtual test relations:** copy-of-Q and inverse-of-Q for every Q ∈ B with both T and F labels, under `rho` and
`shuffled` at k = 0.

## Reads (logic route; means over relations of a type and over seeds)

1. **Instance accuracy:** balanced accuracy (U counted wrong) on all labelled instances except the given ones. It is
   fully predictable from the description for inverse and synonym relations, `child-of`, `darker-than`, and
   virtual relations.
2. **Entailed subsets only** for partially determined relations: broad, T recall where the narrow partner is true;
   narrow, F recall where the broad partner is false; symmetric, accuracy on reverses of the given pairs (k = 4).
   Other instances of these relations are not determined by the description and are reported but not judged.
3. **Applicability:** the share of 300 out-of-type triples recognized as meaningless, and the share of in-type
   labelled pairs kept meaningful. Both are required.
4. **Memorization gap:** instance accuracy of introduced relations that were seen in training (natural, and virtual
   from set A), against held-out relations and virtual relations from set B.
5. **Costs and references:** world-fact accuracy per arm, trained relations through their stored codes, intuition
   fitted on the 4 pairs, and v1's frozen results.

## Predictions

1. **[derived, exact]** Symmetric held-out relations at k = 0 are at chance on non-entailed instances: a description
   cannot pin down a random symmetric extension.
2. **[main]** `virtual`, `rho`, k = 0: held-out inverse and synonym relations (synthetic, plus `child-of` and
   `darker-than`) reach ≥ 0.75. That is ≥ 0.2 above `none` (v1: 0.52 against 0.52).
3. **[derived]** `shuffled` is ≥ 0.15 below `rho` wherever `rho` succeeds: the answers follow which ρ is stated.
4. **[hypothesis]** `virtual`, set-B virtual relations: copies ≥ 0.85 and inverses ≥ 0.75, within 0.1 of set-A
   virtual relations. That would mean the practice generalizes rather than being memorized.
5. **[hypothesis]** `episodes` (natural tasks only) does well on synonyms (≥ 0.75) but not on inverses (< 0.65). It
   has about 20 inverse examples to learn from, and a memorization gap ≥ 0.2 between trained and held-out
   relations.
6. **[hypothesis]** `regress` is ≥ 0.1 below `episodes` on held-out inverse and synonym relations, because a stored
   code is one of many that satisfy a description and regression targets it without regard to behaviour. If it is
   not below, the cheaper method suffices whenever stored codes exist.
7. **[derived]** Applicability from ρ₄/ρ₅ (`virtual` and `episodes`, `rho`, k = 0): ≥ 0.7 recognized *and* ≥ 0.7
   kept meaningful (v1: 0.83 and 0.44, a default).
8. **[derived]** At k = 4, `rho` + fit is ≥ 0.1 above `none` on inverse and synonym relations (v1's k = 4: 0.55–0.69).
9. **[cost]** World-fact accuracy (logic, mean over the T/F tests) is within 0.02 of v1 `nomap` (0.633) in every arm.

## Prior work

Search 2026-11-07. Queries: "OntoZSL ontology-enhanced zero-shot learning knowledge graph relation embeddings",
"Attentive Mimicking Schick Schütze rare word embeddings", "Meta-learning without memorization Yin 2020"; plus the
note's MAML and Neural LP searches.

- **Amortized inference** (Gershman and Goodman 2014; the VAE's encoder) **[familiar]**: a network trained to output
  what an optimizer would find, across many problems. Enc replaces v1's per-relation fit. The **Conditional Neural
  Process** (Garnelo et al. 2018) **[familiar]** is the closest form: a permutation-invariant encoder of a context
  set gives a code that conditions a decoder, trained end to end on target answers. Here the context is statements
  about the relation, not input–output examples. Attention pooling is the Set Transformer's PMA (Lee et al. 2019)
  over a Deep Sets element map (Zaheer et al. 2017) **[familiar]**.
- **Zero-shot relations from an ontology:** [OntoZSL (Geng et al. 2021)](https://arxiv.org/abs/2102.07339)
  **[abstract]** generates embeddings for unseen KG relations from an ontological schema (domain, range, hierarchy)
  with text. This is the established counterpart of this study. Differences: there the schema relations are named
  axioms with text descriptions and a pre-trained KG embedding; here ρ₁…ρ₅ are arbitrary tokens whose meaning must
  be learned from use, and inverse and sameness are among them. ZSGAN (Qin et al. 2020) **[familiar]** does the same
  from text. Attribute-based zero-shot classification (Lampert et al. 2009) **[familiar]** is the older form.
- **Mimicking** ([Attentive Mimicking, Schick and Schütze 2019](https://arxiv.org/abs/1904.01617) **[abstract]**;
  Mimick, Pinter et al. 2017 **[familiar]**): an encoder trained to reproduce stored embeddings of known items from
  their contexts, with attention choosing informative contexts, then applied to rare items. This is the `regress`
  arm. Prediction 6 tests whether behaviour training adds anything over it.
- **Few-shot relations** (GMatching, Xiong et al. 2018; MetaR, Chen et al. 2019) **[familiar]**: codes for a new
  relation from k instance pairs. This is the `none` condition at k = 4, here a fit rather than an encoder.
- **Known failure: memorization.** [Yin et al. 2020](https://arxiv.org/abs/1912.03820) **[abstract]**: when the
  task can be identified without using its data, a meta-learner can solve its training tasks without learning to
  adapt, and fails on new tasks. Here the encoder could recognize a trained relation from its description and output
  a memorized code. Partner identity is the fingerprint, since a synonym's description names its synonym. The
  defences: virtual relations (many tasks per partner type, built from known relations), set-B virtual relations as
  a test, and read 4. Task augmentation (MetaMix, Yao et al. 2021) **[familiar]** is the established remedy that
  virtual relations correspond to.
- **Other known failures:** too few meta-training tasks (91 natural, about 20 inverse), addressed by virtual relations
  and checked by prediction 5; descriptions that do not determine the answers (symmetric, implication), handled by
  judging only entailed instances (prediction 1, read 2); an encoder output off the code distribution, read by the
  k = 4 fit and the applicability head.
- **What theory predicts:** nothing forces an inverse to be a learnable map in the code space. That depends on what
  joint training makes of it. Synonymy only needs the encoder to pass one partner's code through, a pattern
  attention pooling represents directly. So synonyms should be easier than inverses **[derived]**.
- **Cheaper alternatives:** an explicit copy or transpose of the partner would solve inverse and synonym relations
  exactly. It is excluded as a built-in relational form (user direction). v1's partner-copy diagnostic already gives
  its upper bound for synonyms (0.99–1.00). MAML-style meta-learning through the test-time fit is the other route the
  note names. It is not chosen because it unrolls 500-step fits; amortization is cheaper at test.
- **Beyond this study:** the note's query-conditioned reader over statements and partner facts (answering without
  forming one code) is a different architecture, left for later.
- **What this adds:** whether arbitrary relations among relations, learned only through practice in applying them,
  can introduce a new relation in a purely neural system, and whether virtual relations prevent memorization.

## Scale

An introduction costs one encoder pass over |D(X)| statements: O((t + m + c) · h), where t is the number of T
statements, m the sampled F statements (fixed at 64) and c the category statements. No per-relation optimization is
needed, against v1's 500-step fit over all of X's statements. With N known relations, the number of T statements
grows with how related X is to others, not with N. F statements are sampled, so the cost does not grow with N. At
scale, a relation's description would come from memory (statements heard about it) rather than from the truth
function, which is this world's data generator, O(N² · |extension|), and not part of the system. Enc is shared by all
relations, so its size grows with nothing. More relations give more practice episodes.

## Computation (declared)

- **Statements per description:** about 90 (22 category, 64 F, a few T, self).
- **Per training step:** about 27 natural and 16 virtual introductions, so about 3.9 k encoder elements at about 30 k
  multiply-accumulates each: about 0.12 GMAC forward. The reader batch is 8,192 statements at about 25 k MAC each,
  about 0.2 GMAC. Virtual instances add 1,024 statements. ρ-facts leave the batch.
- **Estimates:** 8–15 CPU min training per run at six parallel workers. Test: 15 relations × 2 fitted conditions × 500
  steps, plus 15 intuition fits; encoder-only reads are negligible. About 2 CPU min.
- **Totals:** 12 runs (3 arms × 4 seeds) on 6 workers, about 25–40 min wall.
- **Smoke check (100 steps):** 12–13 CPU s training and 2.5 s test per run, about 6 s world build (descriptions
  included). That extrapolates to about 6–7 CPU min training per run alone.
- **Parameters:** 110,668, of which 35,016 are Enc's. One checkpoint per run, not tracked.
- **Tuning:** none. Every constant is fixed here.
- **Resumability:** per run, with `experiments/durable.py`. The runtime JSON follows `computation-reporting.md`:
  world build (including virtual descriptions), training, and test, by phase.

## Commands

    EV_WORKERS=6 .venv/bin/python -m experiments.amortized_introduction       # rerun to resume
    .venv/bin/python -m experiments.amortized_introduction summary

## Results (2026-11-07)

Run at revision `1e9100d`, 12 runs, 6 workers, 23 min wall. Means over seeds 450–453 and over the held-out relations
of each type. Balanced accuracy, U counted wrong (chance 0.5).

### Training and costs

Trained synthetic relations through stored codes 0.98–0.99 in every arm. World T/F tests (logic, mean) 0.631
(`episodes`), 0.634 (`virtual`), 0.624 (`regress`) against v1 `nomap` 0.633. Trained relations introduced through the
encoder reach 0.91–0.94 (stored codes 0.93–0.95); trained inverses and synonyms 0.95–1.00 in both sets.

### Held-out relations, logic, from the description alone (k = 0) and with 4 pairs

| Held-out type | arm | `rho` k0 | `shuffled` k0 | `none` k0 | `rho` k4 | `none` k4 | intuition k4 |
|---|---|---|---|---|---|---|---|
| synonym (3) | `episodes` | **0.70** | 0.56 | 0.58 | 0.78 | 0.69 | 0.65 |
| | `virtual` | **0.73** | 0.61 | 0.60 | 0.76 | 0.68 | 0.65 |
| | `regress` | 0.57 | 0.54 | 0.55 | 0.66 | 0.65 | 0.65 |
| inverse (4) | `episodes` | 0.50 | 0.50 | 0.50 | 0.51 | 0.51 | 0.53 |
| | `virtual` | 0.50 | 0.51 | 0.50 | 0.51 | 0.50 | 0.53 |
| | `regress` | 0.50 | 0.50 | 0.50 | 0.51 | 0.50 | 0.53 |
| real inverse (`child-of`, `darker-than`) | `episodes` | 0.43 | 0.40 | 0.40 | 0.53 | 0.54 | 0.55 |
| | `virtual` | 0.52 | 0.36 | 0.43 | 0.66 | 0.61 | 0.55 |
| broad (2) | `episodes` | **0.77** | 0.70 | 0.63 | 0.76 | 0.69 | 0.63 |
| | `virtual` | 0.76 | 0.63 | 0.61 | 0.72 | 0.66 | 0.63 |
| narrow (2) | `episodes` | 0.71 | 0.58 | 0.59 | 0.71 | 0.67 | 0.62 |
| | `virtual` | 0.69 | 0.65 | 0.64 | 0.70 | 0.68 | 0.62 |
| symmetric (2) | every arm | 0.48–0.52 | | | 0.46–0.56 | | 0.55 |

Entailed subsets (`episodes`, k = 0): broad, T recall where the narrow partner holds, 0.94 (`none` 0.54); narrow, F
recall where the broad partner fails, 0.92 (`none` 0.96: an unknown relation defaults to F). Symmetric reverses at k
= 4: 0.43–0.54, so a stated self-inverse is not used either.

Applicability, `rho` k0, synthetic inverse + synonym: recognized 0.84 and kept meaningful 0.83 (`episodes`), 0.84 and
0.81 (`virtual`); `none` 0.86 and 0.23–0.26 (a default toward meaningless). Broad and narrow 0.85–0.90 and 0.89–1.00.
Real inverses keep only 0.44–0.49.

### Virtual relations: copying generalizes, reversing does not

| Arm | copy, set B (`rho` / `shuffled`) | copy, set A | inverse, set B | inverse, set A |
|---|---|---|---|---|
| `episodes` | 0.82 / 0.68 | 0.87 | 0.64 / 0.55 | 0.65 |
| `virtual` | 0.85 / 0.70 | 0.95 | 0.68 / 0.57 | 0.90 |
| `regress` | 0.72 / 0.64 | 0.80 | 0.60 / 0.54 | 0.62 |

A diagnostic after the run splits inverse-of-Q by whether its description also has a "same pairs" (ρ₃) partner,
because Q's own inverse is a trained relation. That partner makes copying enough. Seed 450 descriptions, four seeds of
reads:

| Arm | set B, with ρ₃ partner (23) | set B, without (29) | set A, with (17) | set A, without (22) |
|---|---|---|---|---|
| `episodes` | 0.81 | 0.51 | 0.86 | 0.50 |
| `virtual` | 0.84 | **0.55** | 0.97 | **0.84** |
| `regress` | 0.72 | 0.50 | 0.77 | 0.50 |

Reversal is learned only for the partners practised (set A, `virtual`, 0.84) and not carried to new ones (set B 0.55,
held-out synthetic inverses 0.50). This is the memorization failure the contract named, for inverses only.

### Against the predictions

1. **Held.** Symmetric held-out relations at chance (0.48–0.52).
2. **Failed.** `virtual`, inverse + synonym, k = 0: 0.60 against `none` 0.54 (predicted ≥ 0.75, ≥ 0.2 gain). Synonyms
   alone 0.73 against 0.60; inverses 0.50.
3. **Direction held, threshold failed.** Shuffling costs synonyms 0.12–0.14, broad 0.07–0.13, narrow 0.04–0.13
   (predicted ≥ 0.15).
4. **Failed.** Set-B copies 0.85, at the threshold, within 0.10 of set A; set-B inverses 0.68 against set A 0.90.
   Without the copy route, 0.55 against 0.84: memorized.
5. **Mostly held.** `episodes`: synonyms 0.70 (predicted ≥ 0.75), inverses 0.50 (< 0.65), gap between trained
   inverses introduced through the encoder (0.96–1.00) and held-out ones (0.50) ≥ 0.2.
6. **Failed on the stated measure, held on synonyms.** `regress` below `episodes` by 0.06 on inverse + synonym
   (predicted ≥ 0.1); by 0.13 on synonyms, 0.10 on set-B copies, 0.11 on broad.
7. **Held for synthetic relations, failed for real inverses.** Recognized 0.84 and kept 0.81–0.83; real inverses keep
   0.44–0.49.
8. **Failed.** k = 4: `rho` + fit 0.62 against `none` 0.58–0.59 on inverse + synonym (synonyms 0.76–0.78 against
   0.68–0.69).
9. **Held.** World tests within 0.01 of v1.

### What this means

- **Practising application works where the application is a pass-through.** A new relation described as the same as
  (ρ₃) or contained in (ρ₂) a known one now behaves like it: synonyms 0.70–0.73 from the description alone, broad
  relations' entailed instances 0.94, applicability judged instead of defaulted. v1 had none of this (0.52 with or
  without relation facts). Training on behaviour beats regressing onto stored codes for these relations (0.13 on
  synonyms).
- **Reversal does not generalize in code space.** An inverse needs the partner's code turned into a different code
  that answers (b, a) as the partner answers (a, b). The encoder learned that only as a lookup for partners it
  practised on, even with 78 virtual relations. Nothing in the reader makes "swap the arguments" a map that applies to
  codes it has not seen. Copying needs no transform, so attention passes the partner through.
- **This points at applying rules to facts, not compressing them into a relation code.** At the level of facts, the
  inverse rule is trivial: to answer C(a, b), check P(b, a). It is a variable binding, not a code transform. The agreed
  next study (→ and modus ponens with variables, trained on 1–2 steps and tested on longer chains) works at that
  level.

### Computation (measured)

`reports/amortized-introduction-v1-runtime.json`. Per run: world build with descriptions 6–7 CPU s; training 605–722
CPU s (`virtual` the most, about 10% above `episodes`), above the declared 6–7 min alone because six workers ran in
parallel; test 13–15 CPU s (45 fits, 407 encoder reads). Encoder inference: one pass per relation, versus v1's
500-step fit (101–122 CPU s for 135 fits). 110,668 parameters (35,016 encoder); peak RSS 0.51 GB per worker. Total
8,081 CPU s (2.2 CPU h), 23 min wall, within the declared 25–40 min. The post-run split took under a minute. No
tuning.
