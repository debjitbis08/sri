# Sequence prediction pilot v1: next-item prediction without action tokens

Status: **pilot, 2026-10-29; not a frozen contract.** Follows
[transition pilot v1](transition-pilot-v1-results.md). Researchers' decision: **no
add/remove tokens.** The transition network's action tokens named the operations, which is
the kind of hand-given structure the generic design removes.

## Design

- **Network** (`src/generic_net.GenericSequenceNet`): the generic network (standard
  perception, one comparator `g(a, b, relation)`) plus one next-item predictor
  `P(c_prev, c_cur)`.
  - `P` is a two-layer MLP (32 → 64 → 64 → 16) on the codes of the two most recent items
    of a sequence.
  - There is no action or relation input. Whether an episode counts up or down (30% are
    descending) has to be inferred from its history.
  - The shared parameters start as in the generic pilot.
- **Training:** the generic pilot's losses on identical batches and draws, plus a
  prediction loss.
  - For every three consecutive steps t−1, t, t+1 of an episode whose step t+1 shows a
    collection: for each item kind present at both t−1 and t (forms, taps or
    collections), `P(c₍t−1₎, c_t)` must pick the collection seen at t+1 among one
    collection per count.
  - Logits are −exp(s)·‖P − candidate‖², with s learned.
  - Withheld forms take part as history. Their own collections are never shown.
- **Readout:** unchanged; every answer is read from `g`.
- **Arms and seeds:** D-base, D-rehearsal and D-varied+rehearsal; seeds 100–102. Paired
  with the heads network and the generic network.
- **Composition:** from two real items of counts (n−1, n) for upward or (n+1, n) for
  downward, `P` is rolled forward k = 1–3 steps. The result is matched among 9
  collections, by `g(same)` and by distance.

## Differences from the transition pilot

| | Transition pilot | This pilot |
| --- | --- | --- |
| Conditioning | current code + action token (*add* / *remove*) | the two most recent codes, no tokens |
| Training rows | every pair u → v, both directions, as add and remove | every triple t−1, t → t+1, in presented order |
| Withheld forms | sources of predictions | history items of predictions |
| Rows per batch | about 2 per transition | about 1 per step after the second, so fewer |

## Prior work

From memory [familiar]:
- Next-step prediction from context is the standard self-supervised sequence objective:
  [contrastive predictive coding](https://arxiv.org/abs/1807.03748), and autoregressive
  sequence models.
- Inferring the direction of a sequence from its history is what a context window
  provides.
- **Known risk:** fewer and longer-range training rows than the transition pilot. Two
  steps of history must also disambiguate the direction, so the arranging pressure may be
  weaker.

**Prediction, written before the results:** placement and untaught transfer recover less
than in the transition pilot. There are fewer prediction rows, and withheld forms appear
only as history. They should still recover more than in the generic network, because the
predictor still targets collection positions in code space.

## Computation

- **Parameters added:** `P` 7,312 (32 × 64 + 64 + 64 × 64 + 64 + 64 × 16 + 16) and the
  scale, for 7,313 in total.
- **Added cost per update:** one `P` evaluation and a 9-candidate distance per training
  row.
- **Runs:** 9, one wave on 11 workers.
- **Selection and tuning:** none.

## Commands

- `NSY_WORKERS=11 .venv/bin/python -m experiments.sequence_pilot pilot`
- `.venv/bin/python -m experiments.transition_pilot_analysis pilot sequence`
