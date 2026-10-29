# Generic relations v1: the standard network without relation-specific heads

Status: **pilot, 2026-10-29; not a frozen contract.** Follows the researchers' decisions
in the [architecture audit](architecture-audit.md#researchers-decisions-2026-10-29):
- relation-specific heads were a mistake;
- answers should come from an output module, not from code distances;
- the standard network is the default.

Run on the Part D design, so it pairs with [stage 1](architecture-audit-stage1-results.md)'s
standard network by seed.

## What changes

| Standard network (stage 1) | Generic network (`src/generic_net.py`) |
| --- | --- |
| *same*: `b − softplus(s)·‖a − b‖²`, a fixed distance form | one comparator `g(a, b, r)`: a two-layer MLP (40 → 64 → 64 → 1) on two codes and a learned relation token (8-dim) |
| *next*: distance to `S(a) = A a + t`, an affine step map | the same `g` with the *next* token |
| *larger*: `w · (a − b)`, a fixed order vector | the same `g` with the *more* token |
| matching reads code distances | matching reads `g`'s output (the *same* token) |

- **Unchanged:** perception (CNN front ends, one linear projection per sense, 16-dim
  code), data, losses, curriculum, rehearsal, seeds. The projections start from the same
  initial values as the standard network's.
- The relation names remain as **input tokens**, the way a question names what it asks.
  No relation has its own parameters or functional form.

## What is lost, and what replaces it

The removed heads supported explicit calculations. Without them:

| Old mechanism | In this pilot |
| --- | --- |
| Revision by least squares through `S` | **Judgment-combined revision:** each candidate collection is scored by `g(same, form, candidate)` plus the *next* judgments that place it after the grounded previous collection and before the grounded next one |
| Coherence by successor residuals ‖c₍k+1₎ − S(c_k)‖² | **Next-judgment coherence:** a counting run is scored by its summed *next* logits |
| Composition by iterating `S` (*add k*), multi-step objective | **not possible:** there is no transition map to iterate |
| Imagery from `S(p)` or gap states | not possible (and only a probe) |
| Comparison-direction diagnostics (`A`, `t`, `w`) | not defined |

The two generic measures use only `relation_logits`. The analysis therefore applies them
to the standard network's checkpoints too, and both networks are scored by identical
procedures.

**Decision deferred to the researchers.** Composition needs some way to *transform* a
state ("what would this be after adding one?"). One option is a generic, action-conditioned
transition network `T(c, action)`: one network, with the action as an input token. That is
a standard world-model component and not relation-specific. Whether it belongs in a
"perfectly generic" design, or whether composition should be posed only as questions to
`g`, is the researchers' call. This pilot leaves it out.

## Measures

All on D-varied+rehearsal at the end of training unless noted. Seeds 100–102, paired
with the standard network's stage 1 runs.
- **Acquisition:** withheld Chinese and Devanagari forms at the end of their own blocks.
- **Retention:** Chinese paired at the end, for D-rehearsal and D-base.
- **Taught relations:**
  - taps → collection;
  - comparison on collections;
  - *next* succession on collections, pairwise (the true next scored above other
    counts).
- **Untaught compositions (the key test):**
  - taps → form and form → taps for withheld forms;
  - comparison on forms, which was never trained;
  - succession on forms.
- **Judgment-combined revision**, withheld forms: isolated, revised, and whether the
  answer follows a swapped context.
- **Next-judgment coherence** AUROCs.
- **4-shot Thai** after 200 updates.

Outcome labels use the audit's proposed margins (0.05 for accuracies, 0.03 for AUROCs):
paired difference, generic − standard.

## Prior work

2026-10-29, from memory [familiar]; not yet read for this pilot.
- **Relation Networks.** [Santoro et al. 2017, *A simple neural network module for relational reasoning*](https://arxiv.org/abs/1706.01427).
  - One shared MLP `g(o_i, o_j, q)` over object pairs, conditioned on the question,
    answers many relational questions.
  - This pilot's comparator is that design restricted to two objects, with the
    relation as the question.
- **Learned comparators in few-shot learning.** [Sung et al. 2018, *Learning to Compare*](https://arxiv.org/abs/1711.06025).
  - A learned relation module replaces a fixed distance for matching queries to
    support examples.
  - Prototypical networks (Snell et al. 2017) with a fixed distance are a strong
    baseline for it [familiar].

**Known failure modes:**
- A learned MLP comparator has no built-in symmetry, transitivity or ordering. A fixed
  form (distance, linear order vector) gives those for free.
- MLPs often fail to apply a relation systematically to inputs outside the trained
  distribution (Lake and Baroni 2018 [familiar]).

**Prediction, written before the results:**
- The taught judgments should survive.
- The **untaught transfers are at risk.** The old linear order vector gave every point of
  the code space an ordinal score, so comparison transferred to forms automatically. A
  learned `g` must instead generalize from collection and taps codes to form codes.
- For the same reason, untaught comparison on forms and succession on forms are the
  measures most likely to fall.

**What the pilot adds:** whether the line's results need hand-structured relation forms,
or follow from a shared code plus a generic comparator.

## Computation

- **Parameters:**
  - comparator: (40 × 64 + 64) + (64 × 64 + 64) + (64 + 1) = 6,849, plus 24 for the tokens;
  - heads removed: 294 parameters.
- **Cost per judgment:** O(40·64 + 64·64) for the comparator, against O(16²) for the
  affine step and distance. Matching scores 9 candidates per query.
- **Runs:** 9 (3 arms × 3 seeds), one wave on 11 workers.
- **Analysis:** recomputes the generic measures on 6 end-of-training checkpoints.
- **Selection and tuning:** none. The comparator size and token size were set once, not
  searched.
- **Timing** is in the runner's and the analysis's runtime files.

## SRI variant (added 2026-10-29, researchers' request)

The same replacement on the SRI network (`src/generic_net.GenericSRI`):
- the SRI modules, translations and energy are unchanged and initialized as in the Part D
  pilot;
- the three heads are replaced by the same comparator on the core-inner code;
- paired by seed with the SRI Part D pilot (with heads).

It asks whether SRI's coupling supplies the arranging pressure that the heads supplied in
the standard network. The comparator reads the core code, where the heads were, not the
output module; an output-module readout would be a different design.

## Commands

- `NSY_WORKERS=11 .venv/bin/python -m experiments.generic_relations pilot [standard|sri]`
- `.venv/bin/python -m experiments.generic_relations_analysis pilot [standard|sri]`
