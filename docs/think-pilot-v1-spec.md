# Thinking-loop pilot v1: answer after K steps of recurrent relational reasoning

Status: **pilot, 2026-10-29; not a frozen contract.** Researchers' request: a loop-like
(RRN-like) structure, so that predictions are tried before the actual output. Chosen
reading (researchers, 2026-10-29): **think before answering.** The system loops internally,
and the answer comes from the loop's final state. The number of steps is the thinking
effort.

## Design

- **Base:** the [sequence pilot](sequence-pilot-v1-results.md)'s network: standard
  perception, the next-item predictor `P(c_prev, c_cur)`, no action tokens. Its prediction
  loss still arranges the code.
- **Answerer** (`src/generic_net.ThinkingSequenceNet`): a recurrent relational network
  ([Palm, Paquet and Winther 2018](https://arxiv.org/abs/1711.08028)) replaces the one-pass
  comparator `g`.
  - A question is a set of nodes:
    - *same* (matching): the query and its 9 candidates;
    - *next* and *more*: the two items.
  - Each node starts from `x_i = W[code_i, role_i, question token]`. The roles are query,
    candidate, first and second; the question tokens are *same*, *next* and *more*, as
    before.
  - For K steps, each node sums messages `f(h_i, h_j)` from all other nodes and updates
    its state with a GRU cell on `[x_i, m_i]`. An answer logit is read from every node at
    every step: from the candidates for matching, otherwise from the second item.
  - Hidden size 32; 13,665 parameters.
  - **Matching now sees all candidates together.** The loop can weigh candidates against
    each other, which the hubness correction did by hand.
- **Training:** the sequence pilot's losses and draws. Every relational loss is
  averaged over **K = 4** steps, so an answer is trained at every step (deep supervision,
  as in RRNs).
- **Evaluation:** the sequence pilot's measures at K = 4. The checkpoints are then
  re-evaluated at **K = 1, 2, 4 and 8** (the thinking-effort curve).
- **Arms and seeds:** D-base, D-rehearsal and D-varied+rehearsal; seeds 100–102. Paired
  with the sequence pilot (identical batches; the shared parameters start equal) and
  with the heads and generic networks.

## Prior work

- **Recurrent relational networks.** [Palm, Paquet and Winther 2018](https://arxiv.org/abs/1711.08028)
  [familiar].
  - Message passing over a graph for many steps, with a loss at every step. It solves
    tasks that need many steps of relational inference (Sudoku, bAbI) that one-pass
    relation networks do not.
  - Accuracy rises with the number of steps.
- **Adaptive and iterative computation:** ACT (Graves 2016), PonderNet (Banino et al.
  2021) [familiar]. More internal steps for harder inputs.
- **Expected limitation here:** matching against 9 candidates and two-item judgments need
  little multi-step inference. The loop may add little at test unless the answer depends
  on comparing candidates (hubness-like cases) or on context. Context items are not yet
  nodes in this pilot.

**Prediction, written before the results:**
- Taught measures similar to the sequence pilot.
- Cross-sense matching (taps → form) may improve, because candidates are compared jointly.
- The effort curve rises from K = 1 to K = 4 and flattens or falls at K = 8, beyond the
  trained number of steps.
- Placement of withheld forms is not expected to change: the loop reads the code, and the
  code is shaped by the same losses.

## Computation

- **Per question:** N nodes, N(N − 1) messages per step, K steps.
  - Matching: N = 10, 90 messages per step.
  - Judgments: N = 2.
  - Each message costs O(64 · 32 + 32 · 32), and each GRU update O(3 · (64 · 32 + 32 · 32)).
- **Training costs about twice the sequence pilot per update** (measured in the smoke run).
- **Runs:** 9, one wave on 11 workers.
- **Effort curve:** 3 checkpoints × 4 values of K, evaluations only.
- **Selection and tuning:** none.

## Commands

- `NSY_WORKERS=11 .venv/bin/python -m experiments.think_pilot pilot`
- `.venv/bin/python -m experiments.transition_pilot_analysis pilot think`
- `.venv/bin/python -m experiments.think_pilot_effort pilot`
