# OP-03B — Stability–expressivity trade-offs

Status: COMPLETE within the declared Boolean search contract. The outcome is a
bounded impossibility result, plus a restricted analytic trade-off theorem.
It does not rule out richer state spaces, different rules, other disturbance
models, or alternative schedule assumptions. No claim of useful inference follows.

## Contract and scope

The [v0.1 contract](op-03b-contract.md) fixed the gates before inspecting the
search results. This is an in-session specification, not an external
preregistration. No failing gate was relaxed to obtain positive examples.

Use the existing eight OR/XOR rule triples and 16 star edge masks. Each candidate
response family consists of two distinct persistent evidence pairs chosen from
`00,01,10,11`, giving six families per network and **768 families per schedule
regime**. Initialize internal state to `000`; arbitrary initial-state robustness
is not required. Evidence remains clamped. The observation is the ordered pair
of satellite states; core cycling is allowed only when that pair is constant.

The regimes are reported separately:

- `all_fair`: every node-fair single-node schedule, using OP-03's exact SCC analysis.
- `fixed_sweeps`: each of the six repeated ordered sweeps, with response agreement
  across all six. The order stays fixed within a run. After a state or structural
  perturbation, any fixed order may be chosen again. Atomic phases are observed,
  not just sweep boundaries; the execution state includes the schedule phase.

The second regime excludes adversarially changing orders, but is not the class
of all bounded-delay schedules. Neither regime includes synchronous simultaneous
updates. No probabilities or typical-CPU scheduling claims are introduced.

## Five gates

For both inputs, require a stable pair from zero; the two pairs must differ.
The isolated A and B observations must also be well-defined at both inputs, with
at least one joint pair differing from its independent pair. This is contextual
interaction, not the stronger OP-02B recurrence-necessity criterion.

For recovery, take **every** recurrent state reachable from zero under the regime,
flip one of the three internal bits, and require the original satellite pair
under every admissible continuation. Only one disturbance is applied. Hidden
core-cycle phases and schedule alternatives cannot be discarded.

For structural tolerance, at least one common present edge must be deletable
for both inputs while preserving their original pairs, both from zero and from
every established recurrent state. Report all qualifying deletions. This can
include removal of a redundant connection; it is not learning, and the modified
network is not required to satisfy the recovery gate itself. Repeated edits are
outside the contract.

## Exact result: recovery eliminates all distinguishable families

Run `python -m experiments.response_family` to regenerate
[response-family.json](../reports/response-family.json). The report contains all
128 networks, four input assessments and six family assessments per network,
for each regime, including explicit recovery counterexamples.

| Successive gate | All node-fair schedules | Six fixed sweeps |
| --- | ---: | ---: |
| Candidate two-input families | 768 | 768 |
| Stable at both inputs | 217 | 270 |
| Also distinguishable | 208 | 250 |
| Also contextual | 14 | 34 |
| Also recoverable | 0 | 0 |
| Also structurally tolerant | 0 | 0 |

For fully reciprocal stars alone, 48 families are tested per regime. Their
successive counts are `(7,4,2,0,0)` and `(16,10,6,0,0)` respectively. There is
therefore no qualifying full-star family either.

The obstruction is stronger than the gate order suggests. Even **without** the
contextual or structural gate, no distinguishable family has two recoverable
responses. Across the 512 network/input cases per regime, 40 are recoverable
under `all_fair` and 48 under `fixed_sweeps`; every one has the same observed pair
`11`. Any two recoverable responses therefore fail distinguishability.

This is an exact bounded-enumeration result, not a general algebraic theorem
about OR/XOR networks of arbitrary size. Each stable input that fails recovery includes a
recurrent source state, flipped coordinate, and recurrent post-disturbance
states exhibiting failure to restore the pair. Undefined/oscillating responses
are not counted as distinct outputs or as successful interpretations.

Structural tolerance is not independently impossible: omitting only the recovery
gate leaves 10 and 22 stable, distinguishable, contextual families with a common
preserving deletion. Thus the final zero in the table must not be read as zero
successful edge-preservation cases. The decisive obstruction here is the
incompatibility of recovery with distinct outputs under this contract.

### A simple failure retained as a control

An all-OR path B→K→A from zero maps evidence `00` to response `00` and `01` to
`11`. This is stable, distinguishable, and contextual: at `01`, isolated A stays
0 while coupled A becomes 1. But under evidence `00`, flipping the core from
0 to 1 permanently activates A. The response cannot recover. The stable `11`
response, in contrast, survives single-bit disturbances. This is latching,
not a robust family of two interpretations.

For `(OR,OR,XOR)` at evidence `01`, restricting to fixed sweeps removes the
OP-03 timing counterexample and yields response `11` consistently, but does
not restore single-bit recoverability. Relaxing the schedule class alone does
not solve the present obstruction.

## A restricted monotone trade-off theorem

Restrict to all-OR rules, the same three-node star, zero initialization, persistent
Boolean evidence, and the same satellite-pair observation. For edge mask E define:

- D(E): number of distinct response pairs across all four inputs.
- R(E): number of those inputs whose response survives every single internal-bit
  disturbance from its zero-start recurrent state under every fair schedule.

Let p indicate the path A→K→B, and q indicate B→K→A. They are Boolean indicators,
not coupling strengths. Directed reachability gives

```
a = x_A OR (q AND x_B)
b = x_B OR (p AND x_A)
```

**Proposition:** `D(E) = 4 - p - q`, `R(E) = 1 + p + q`, hence

```
D(E) + R(E) = 5.
```

**Proof of distinguishability.** Input `00` yields `00`, and `11` yields `11`.
Input `10` yields `10` if p=0 and `11` otherwise. Input `01` yields `01` if q=0
and `11` otherwise. Counting distinct pairs gives the stated D.

**Proof of recovery.** If a target satellite bit is 0, flipping that bit to 1
cannot be undone by any OR update: self-retention keeps it 1. Thus recovery
requires target pair `11`. Conversely, when the zero-start response is `11`,
both satellites are reachable from persistent evidence sources. After any one
internal-bit flip, fair propagation from those sources restores both satellite
ones; extra activation cannot change an already maximal pair. Therefore exactly
inputs `11`, and `10` when p=1, and `01` when q=1, are recoverable. This gives R.

Under edge addition E subset E', neither p nor q can decrease. Consequently R
cannot decrease, D cannot increase, and **any strict increase in R forces an
equal strict decrease in D**. The certificate verifies all 16 masks and all 65
strict nested-mask comparisons. The same formulas hold for the six fixed sweeps,
since they converge to the same OR reachability closure.

This answers the proposed trade-off question only under these exact assumptions
and this definition of robustness. It does not imply that robustness generally
costs expressivity. The metric R counts recoverable *inputs*, not basin volume,
structural tolerance, accuracy, or memory capacity. The distinction count uses
all four inputs from zero, not every possible initial state.

## Verification and interpretation

Tests independently check the OR closed forms, replay atomic phases of each
fixed sweep, verify the difference between preserved latches and reconstructed
responses, exercise the common-deletion gate, and regenerate the report exactly.
The existing OP-02/02B/03 certificates remain unchanged.

The Boolean control has now served its narrow purpose. It supplies a precise
failure mechanism: retention makes zero-valued distinctions vulnerable to false
activation, while the mixed OR/XOR alternatives in this finite family do not
supply a second recoverable response. The strongest existing positive examples
are therefore not a family of robust distinct interpretations.

## Transition to graded coupling and layer-resolved modes

Stop expanding Boolean truth tables here. The next experiment should ask whether
a specified layer-resolved state and graded coupling law can restore at least
two distinct response signatures after perturbation, while retaining contextual
interaction under an explicitly chosen schedule policy. Define the observable,
allowed perturbation size, and coupling-edit metric before the implementation.
Keep the Boolean impossibility result and OR closure as controls.

Merely replacing binary edge flags with weights does not automatically resolve
the OR self-retention failure: the new local law must explain how a spurious
active state can be corrected without collapsing every input to the same output.
This is the identifiable question for adding richer state and coupling.

STM remains separate. After a qualifying family exists, define admissible
structural configurations C and update maps `L: C × X → C`, with explicit
preserved response properties and allowed forgetting. Single-edge tolerance
does not prove closure under a sequence of edits. The full-star all-controls-
stable target from OP-02B remains open and has not been weakened or replaced.
