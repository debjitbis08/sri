# Transition pilot v1: a generic world model as the arranging pressure

Status: **pilot, 2026-10-29; not a frozen contract.** Follows
[generic relations v1](generic-relations-v1-results.md). Without relation-specific heads,
taught relations survived but untaught transfer collapsed, in both the standard network
and SRI. The interpretation: the heads' fixed forms arranged the shared code, and a
generic comparator does not.

This pilot tests the researchers' chosen generic replacement for that pressure: **one
action-conditioned transition network trained to predict the next item's code.**

## Design

- **Network** (`src/generic_net.GenericTransitionNet`): the generic network of
  generic relations v1 (standard perception, one comparator `g(a, b, relation)`), plus
  `T(c, action)`.
  - `T` is a two-layer MLP (24 → 64 → 64 → 16) on a code and a learned action token
    (*add*, *remove*).
  - One network serves both actions; no action has its own form.
  - The shared parameters start from the same values as in the generic pilot.
- **Training:** the generic pilot's losses on the same batches and draws (Part D:
  matching, *next*, *more*, all through `g`), plus a **prediction loss.**
  - For each counting transition u → v in the batch (v one count above u), `T(c_u, add)`
    must pick the collection seen at v's step among one collection per count. The
    candidates are the batch's shared ones, with the true slot replaced by that
    collection.
  - `T(c_v, remove)` must likewise pick u's collection.
  - Logits are −exp(s)·‖T(c, a) − candidate‖², with s learned.
  - Sources are any kind, withheld forms included. Targets are visible collections only.
    A withheld form's collection is never shown, as before.
  - The loss draws no random numbers, so every batch is identical to the generic pilot's.
- **Readout:** unchanged. Every answer is read from `g`. The squared distance appears
  only in the prediction loss: it is the one metric of the code space that the world
  model predicts in, not a relation's form.
- **Arms and seeds:** D-base, D-rehearsal and D-varied+rehearsal; seeds 100–102. Paired
  with the stage 1 standard network (with heads) and the generic pilot.

## Measures

- As in generic relations v1: withheld forms, untaught compositions, taught relations,
  retention, judgment revision, next-judgment coherence, Thai.
- **New: composition.** `T` is applied k = 1–3 times (*add* or *remove*) to an item's code.
  The result is matched among the 9 collections of the same evaluation item, by `g(same)`
  and by distance. Kinds: collections, taps, and the forms of each language.

## Prior work

2026-10-29, from memory [familiar], not yet read for this pilot.
- **Contrastive predictive coding.** [van den Oord et al. 2018](https://arxiv.org/abs/1807.03748).
  Predicting future codes contrastively against negatives learns representations without
  collapse. The prediction loss here is that form, with an action input.
- **Action-conditioned world models and latent dynamics:** for example Ha and Schmidhuber
  2018; Hafner et al. 2019. A transition network over a learned latent is a standard
  component.
- **Collapse in latent prediction.** Predicting a target in a learned space can collapse
  every code to a constant. Contrastive negatives (as here) or variance regularization
  (VICReg) prevent it.
- **Earlier result in this line:** the affine successor `S` with a distance judge
  produced short steps unless trained on multi-step targets
  ([multi-step results](number-construction-v1-multistep-results.md)). An unconstrained
  `T` may also compose poorly. Composition is measured, not assumed.

**Prediction, written before the results:**
- The prediction target in code space should restore much of the **placement of withheld
  forms.** A withheld form is a source of *add* and *remove* predictions toward the
  visible neighbouring collections, which constrains where its code sits.
- **Untaught cross-sense matching** should recover partly. Taps and forms are both
  pulled toward collection positions.
- **Untaught comparison on forms** (*more*) may stay low. Nothing trains `g`'s *more*
  judgment to read positions that `T` arranges, unless the arrangement is linear enough
  for `g` to generalize.

**What the pilot adds:** whether a generic predictive objective, rather than
hand-shaped relation forms, supplies the pressure that makes relations transfer.

## Computation

- **Parameters added:** `T` 6,800 (24 × 64 + 64 + 64 × 64 + 64 + 64 × 16 + 16), action
  tokens 16, scale 1, for 6,817 in total.
- **Added cost per update:** one `T` evaluation per transition pair and action, and a
  9-candidate distance per prediction: O(P · (24·64 + 64·64 + 64·16 + 9·16)), with P
  pairs in the batch.
- **Runs:** 9, one wave on 11 workers.
- **Selection and tuning:** none. The sizes match the comparator, and the loss weight is
  1, set once.
- **Timing** is in the runner's and the analysis's runtime files.

## Commands

- `NSY_WORKERS=11 .venv/bin/python -m experiments.transition_pilot pilot`
- `.venv/bin/python -m experiments.transition_pilot_analysis pilot`
