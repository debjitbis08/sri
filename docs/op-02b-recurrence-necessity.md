# OP-02B — Recurrence necessity under matched edge deletion

Status: PROVED (scoped) witnesses and minimal edge count; EXHAUSTIVE (bounded)
classification. OP-02B remains PARTIAL as an architectural research question.

This iteration advances the [first recurrent experiment](op-02b-recurrent.md)
without changing its report or the original OP-02 artifacts. It establishes a
feedback-dependent stabilization effect, not useful inference or irreducibility
to every independent implementation.

## Fix the observable before comparing controls

State, local rules, edge order, and schedules are unchanged. Evidence is clamped;
all three internal nodes retain state. Here **acyclic** means no *inter-node*
directed cycle. Each node's local state dependence remains in every control.

For an exact eventual cycle C and coordinate set J, define P_J(C) as the single
projection shared by **every** state of C, if there is one, and undefined otherwise.
This detects stable component outputs even when hidden coordinates keep cycling.
It never averages phases or treats one phase as the output.

The joint system must reach a full-state fixed point. Its declared response is
the ordered satellite pair `(a,b)`. For the independent response, take the stable
A projection in A-only and the stable B projection in B-only. The A-only mask
retains only A/core edges actually present in the candidate; B-only is symmetric.
The unused satellite stays in the state representation, so it may oscillate
without invalidating the observed isolated component's response.

A candidate passes the scoped recurrence test exactly when:

1. The joint system reaches a fixed point.
2. Both isolated satellite observations are stable.
3. The joint satellite pair differs from the ordered pair of isolated outputs.
4. No acyclic edge submask produces that same stable satellite pair.

Every run uses identical rules, evidence, initial state, and schedule. For the
full star there are nine acyclic submasks: neither reciprocal pair may be intact,
so each of the two pairs has three possible choices (neither edge or either one).
The empty mask, both feed-forward edges, and both oriented paths are included.

We separately count the stronger condition that **all** acyclic satellite-pair
observations are stable. This distinguishes differing stable outputs from a
contrast that depends on a control continuing to oscillate. A cycling core with
constant satellite outputs counts as a stable observation, not a failure to settle
at the observation level.

## Why the projection detail matters

Two full-star cases pass a weaker test that accepts only full-state fixed points
in acyclic controls, but fail the criterion above. One uses rules `(XOR,OR,XOR)`,
evidence `00`, initial `001`, and synchronous updates. The full star ends at `011`.
Its acyclic B→K→A control cycles between `011` and `111`: the core oscillates but
the satellite pair is constantly `11`, exactly matching the full star. The mirror
case exchanges A and B. The report retains the first counterexample in full.

Conversely, requiring the *entire* A-only bookkeeping state to settle wrongly
rejects cases where disconnected B oscillates but A has a stable output. The
new report uses projection semantics consistently for both kinds of controls.
The original report's whole-state-fixed counts are preserved as historical results;
its counts and the new contextual counts therefore answer different questions.

## Exhaustive result

Run `python -m experiments.recurrence_necessity` to regenerate
[recurrence-necessity.json](../reports/recurrence-necessity.json). The analyzer
uses exact first-repeat detection with the same eight-state termination bound.
There is no cap, approximation, or sampling.

For the full four-edge star: 8 OR/XOR rule triples × 7 schedules × 4 evidence
pairs × 8 initial internal states = 1,792 candidate cases.

| Successive condition | Cases |
| --- | ---: |
| Full-state fixed point | 961 |
| Also stable isolated A and B observations | 841 |
| Also contextual satellite pair | 224 |
| Also no matching acyclic satellite response | 80 |
| Of those, zero internal initial state | 22 |
| Of those, all acyclic satellite observations stable | 0 |

These are labeled cases, not distinct mechanisms or isomorphism classes. The
report lists all 80 full-star witnesses and counts by rule and schedule.

The extended search treats all 16 edge masks as candidates, comparing each with
only its own acyclic submasks. This covers 28,672 candidate cases. The controls
are additional evaluations, not additional candidate cases.

| Candidate edges | Cases | Passing recurrence test | Passing with all acyclic observations stable |
| --- | ---: | ---: | ---: |
| 0 | 1,792 | 0 | 0 |
| 1 | 7,168 | 0 | 0 |
| 2 | 10,752 | 0 | 0 |
| 3 | 7,168 | 82 | 2 |
| 4 | 1,792 | 80 | 0 |

The three-edge cases are ablation controls, not fully reciprocal two-satellite
architectures. The zero in the last row is a bounded negative result: none of
these full-star cases yields a distinct stable pair while every acyclic control
also has a stable pair. It is not a theorem about arbitrary finite networks.

## A full-star witness: feedback stops an oscillating satellite

Use rules `(OR,OR,XOR)`, evidence `(0,1)`, zero internal initialization, and
synchronous updates:

```
k' = k OR a OR b
a' = a OR k
b' = b XOR k XOR 1

000 → 001 → 100 → 110 → 110
```

A-only has stable A output 0 (disconnected B oscillates); B-only has stable B
output 0. The independent pair is therefore `00`, while the joint pair is `10`.
Once k becomes 1, the two terms `k XOR 1` cancel, and B retains its current bit.
The same k activates A. This is a transparent feedback stabilization mechanism.

**Why every acyclic deletion fails.** Without K→B, B toggles forever because
`b'=b XOR 1`. If K→B is kept, acyclicity excludes B→K. With zero initialization
and A's zero evidence, k and a remain zero, even if A→K is present: K→A cannot
also be present in an acyclic mask. B therefore toggles in this case too. Thus
all nine controls have oscillating satellite pairs, while the full star settles.

At this evidence the full star has two fixed points, `110` and `111`, with basin
sizes five and three respectively under synchronous updates. The chosen initial
state selects `110`. The same initial state under each of the six ordered sweeps
selects `111`; it still passes the recurrence test, but the response value is
schedule-dependent. The fixed point need not restore itself after a perturbation.
This is neither a unique global interpretation nor a robustness guarantee.

## Minimality under edge deletion

Removing A→K from the witness leaves three edges, B↔K and K→A, and preserves its
trace and passing result. Three is the minimum edge count in the declared star
family. An acyclic candidate is its own matching control, so cannot pass. A cyclic
candidate with at most two edges consists only of one core/satellite reciprocal
pair and a disconnected other satellite. Each satellite then behaves exactly as
in its isolated control, so cannot satisfy contextual change. The three-edge
witness achieves the lower bound. The enumeration independently checks it.

The stipulated fully reciprocal architecture requires four edges by definition;
the three-edge reduction shows that both reciprocal pairs are not necessary for
this effect. There are always three mutable nodes in this search. The uniform
binary alphabet is the smallest nonconstant alphabet, but heterogeneous alphabets,
node counts, and alternative graphs were not enumerated. No universal architectural
minimality claim follows.

## A stricter three-edge boundary case

Rules `(XOR,OR,XOR)`, evidence `00`, initial internal `001`, synchronous updates,
and edges B↔K plus K→A give:

```
001 → 101 → 010 → 010
```

The core/B recurrence clears their initial activity after A has latched it. The
joint satellite pair is `10`; the isolated pair is `00`. All six acyclic submasks
have stable satellite outputs, either `01` or `11`, so none reproduces `10`.
Two of these controls have cycling cores; their satellite observations are still
stable. The report retains their exact cycles rather than calling them fixed points.

This case and its A/B mirror are the only two stricter cases in the enumeration.
They require a nonzero initial internal state and lack one reciprocal satellite
edge. Adding that missing edge makes the first case fail the recurrence test
through the hidden-core-cycle counterexample above. Adding coupling therefore
need not strengthen this particular property.

## Interpretation and next obligation

The scoped test now has positive witnesses with exact proofs and minimal edge
counts. It establishes that deleting all inter-node recurrence, while keeping
rules and initial conditions fixed, can destroy a stable contextual response.
It does **not** rule out an acyclic circuit with different rules, a lookup table,
or a model that reconstructs the result from the complete isolated information.
It also does not establish topology-dependent semantic modes, learned abstraction,
accuracy, efficiency, or a capability unavailable to a suitable independent model.

The strongest fully reciprocal result here is feedback-dependent stabilization;
none passes the all-controls-stable comparison. Before changing the local-rule
family, specify which additional property is needed: for example, a full-star
response distinct from stable acyclic responses, with an explicit initialization
and schedule-robustness requirement. Preserve the positive stabilization witnesses,
the three-edge boundary, and the projection counterexample as regression controls.
A capability claim still requires a separately specified inference task.
