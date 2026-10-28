# Number systems meta v1: training for learning efficiency

Status: **draft, 2026-10-28. Pilot design, not frozen.** Builds on [number systems
v1](number-systems-v1-spec.md) and its [pilot results](number-systems-v1-pilot-results.md),
including Stage A. Decisions recorded on 2026-10-28: real-valued network as before; three
representations moved from source training to development.

## Question

Stage A showed two things:
- source experience raised the accuracy reached with limited examples, but compatible
  order added no detectable benefit over private orders;
- adapting the shared image interface halved the sources' accuracy.

Ordinary training rewards only the current label and relation tasks. Nothing in it
favours parameter changes that make later learning easier (external review).

This study adds that signal directly. A bounded adaptation episode is put inside
training, and the shared parameters are updated from

> E_learning = L_new (different examples, after B adaptation updates) + λ L_preservation
> (source tasks after the same adaptation).

## Representations

| Role | Representations |
| --- | --- |
| Final evaluation (never trained, never used for choices) | tree, german, thai_digits, hand |
| Development (episodes, the data-matched control, and tuning only) | banana, french, devanagari_digits |
| Sources (sparse complementary labels and relations, as in v1) | mnist, dice, dots, tally, egyptian, roman, chinese, arabic_digits, english, hindi, morse |

- Development covers three families: grouping (size 4), names, and distinct symbols.
- The sequence medium has only Morse and Hand, so it gets no episodes. Hand tests whether
  any effect carries to a medium never trained this way.

## Episode (first-order MAML)

1. Draw a development representation, a support set of k = 8 examples per number (all
   nine labelled), and its within-representation relations.
2. Make B = 10 updates to the **shared image satellite** (front end, input module and
   translations), with Adam at learning rate 1e-2. This is the deployment setting: one
   satellite per medium.
3. Compute the outer loss at the adapted parameters:
   - L_new: labels and relations on query examples of the same representation, drawn
     from outside the support set;
   - L_preservation: the ordinary source loss for 2 image sources, sampled per episode,
     encoded through the adapted satellite;
   - with λ = 1.
4. Restore the satellite to its pre-episode values. Apply the outer gradient (taken at the
   adapted parameters, the first-order approximation) to all parameters through the main
   optimizer.

## Arms (equal budget)

The budget is counted in training encodings: 3,000 ordinary steps × 4 representations ×
448 encodings each = 5.376 M. Training CPU time is also measured and reported.

| Arm | Training |
| --- | --- |
| ordinary | source steps only |
| ordinary + dev | source and development representations sampled together, 4 per step; development fully labelled, as in episodes |
| meta | every 4 source steps, one episode (≈ 3.25 source steps of encodings) |
| meta-I | as meta, with every source's and development representation's relation order a private permutation |

The ordinary + dev arm sees the same development data under the ordinary objective. It
separates the new objective from the extra data.

## Evaluation

The deployment adaptation matches the episode: interface mode (shared image or sequence
satellite), Adam at learning rate 1e-2. Settings:
- k ∈ {2, 8, 32} examples per number, all labelled, with relations;
- up to 200 updates, scored at 0, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150 and 200;
- 100 test images per number;
- retention measured through the shared interface: mean accuracy of each source on its
  own labelled numbers, on 50 test images per number.

The same evaluation runs from random initialization (no source).

- **Criterion, fixed in advance:** for each held-out representation, the no-source accuracy
  at k = 32 after 200 updates. Every arm can therefore reach it in principle.
- **Cost to criterion:** updates, examples (9k) and training CPU time. It is reported both
  without and with retention ≥ 0.9 × before. Runs that never reach it are censored at 200.
- **Also reported:** best accuracy within the budget.
- **Source state at the end of training:** label accuracy, complementary transfer, and the
  cross-representation number probe (from v1), to show what the episodes cost the sources
  and whether they change sharing.

## Predictions

1. If the learning-efficiency error works, meta reaches the criterion with fewer
   examples or updates than both ordinary arms, while meeting the retention requirement
   more often.
2. If meta beats ordinary + dev, the gain comes from the objective, not the data.
3. If meta beats meta-I, the gain uses the shared number order.
4. If meta matches ordinary + dev, the episodes add data, not a useful signal, at this λ,
   B and budget.

## Not decided yet

- λ, B, k and the inner learning rate are defaults. Tuning uses development
  representations only, and counts as selection cost.
- Seeds: the pilot runs one (100).
- Prior work: [training for learning
  efficiency](prior-work.md#training-for-learning-efficiency-number-systems-meta), covering
  MAML, first-order MAML, Reptile, OML, ANML, ANIL and Chollet. Abstract-level so far.
  - ANIL's feature-reuse finding predicts meta ≈ ordinary + dev.
  - With only three development representations, meta-overfitting is the main risk.

## Computation

- Encodings per ordinary step: 4 × (64 + 6 × 64) = 1,792.
- Per episode: (B + 1 + 2) × 448 = 5,824.
- Factorizations per step: one per medium present. Each episode adds B + 2.
- Evaluation: 5 starting points × 4 held-out × 3 k = 60 adaptation runs of 200 updates,
  each with 12 scoring passes.
- Measured costs go into `-runtime.json`, following [computation
  reporting](computation-reporting.md).

## Episode audit (development only, added 2026-10-28 after the gradient diagnostic)

Before judging the learning-efficiency gradient, the inner adaptation must work. Fixed
before running:

- **Checkpoints:** ordinary and meta from the pilot.
  - Ordinary never saw the development representations, so it is the clean test.
  - Meta's outer loss used development queries, so they are not new to it.
- **Episodes:** 3 development representations × 2 episodes; support of 8 examples per
  number, all labelled, with relations. Architecture unchanged (real-valued, ReLU).
- **Inner procedures:**
  - Adam at {1e-4, 3e-4, 1e-3, 1e-2}, or plain SGD at {1e-3, 1e-2, 3e-2, 1e-1};
  - adapted parameters: the whole image satellite, its SRI part only (W, V, T), or its
    front end only;
  - up to 50 updates, scored at 0, 5, 10, 25 and 50.
- **Measures, before and after each budget:**
  - support accuracy (the 72 support images);
  - query accuracy (50 test images per number);
  - image-source retention (accuracy on each source's labelled numbers, 20 test images
    per number, mean over the 10 image sources);
  - divergence.
  
  Interpretation follows the review:
  - no support gain: ineffective adaptation;
  - support gain without query gain: fitting without generalizing;
  - both: useful adaptation.
- **"Functioning" rule:** mean query gain ≥ 0.15 over the episodes, with image-source
  retention ≥ 0.9 × before. Among functioning procedures, prefer the largest query gain,
  then the fewest updates. Only a functioning procedure goes on to the gradient audit and
  to an equal-compute first-order versus exact comparison.

### Decision after the episode audit (2026-10-28)

No procedure met the functioning rule above: every procedure that learned also forgot.
The researchers chose **option 1**:
- "functioning" means stable improvement in both support and query;
- retention becomes the quantity the outer loop must improve, not a precondition;
- the architecture stays fixed.

**Selected inner procedure.** On the ordinary checkpoint (the one that never saw the
development representations), among non-diverging procedures with mean query gain
≥ 0.15, take the fewest updates, then the highest retention:
- Adam at 1e-3, whole image satellite, **25 updates**;
- audit values: support +0.54, query +0.16, retention 0.73.

**Gradient audit through that procedure** (development only):
- Adam unrolled differentiably, using sqrt(v̂ + 1e-16) so the backward pass stays finite
  where v̂ = 0;
- float64; ordinary and meta checkpoints; 2 episodes per development representation;
- exact versus first-order gradient: cosines, and finite differences at several step
  sizes;
- unit-norm outer steps η ∈ {0.003, 0.01, 0.03} along each, scored on 6 fresh episodes:
  - support and query accuracy before and after adaptation;
  - image-source forgetting, measured as accuracy before adaptation minus after;
  - the outer step's own effect on source accuracy before adaptation, kept separate.
- The measured CPU ratio of exact to first-order gradients sets the episode counts for
  the equal-compute training comparison. That comparison is fixed in this spec only
  after the audit.
