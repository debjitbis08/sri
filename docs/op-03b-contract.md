# OP-03B search contract — v0.1

This contract fixes the search before inspecting its results. It is an in-session
specification, not an external or time-stamped preregistration. Do not relax a
failed gate within this iteration.

## Bounded candidate family

Use the existing 8 Boolean OR/XOR rule triples and all 16 star edge masks.
Consider every unordered pair of distinct clamped evidence inputs from
`00,01,10,11`: six two-input families per network, 768 candidate families.
A run starts from internal `000`. No arbitrary initial-state guarantee is sought.
All cases, including acyclic controls, are reported; fully reciprocal results
are counted separately. No new local rules, graded weights, or learning updates.

## Two separately reported schedule assumptions

1. `all_fair`: arbitrary single-node schedules selecting every node infinitely
   often, as in OP-03. No delay bound.
2. `fixed_sweeps`: one of the six ordered node sweeps repeats forever. The response
   must agree for all six choices. Orders cannot change within a run. After a
   declared state/edge perturbation, any one fixed order may be chosen anew.

Observations include every atomic update phase, not just sweep boundaries.
Neither regime includes simultaneous synchronous updates. The second is a
restricted scheduling policy, not a replacement for the stronger first result
and not an analysis of all bounded-delay fair schedules.

## Five gates for a two-input response family

- Stable: from `000`, each input has a unique eventual satellite pair across all
  admissible schedules. Hidden core cycles are allowed only if the pair is
  constant in every recurrent phase.
- Distinguishable: the two inputs have different satellite pairs.
- Contextual: the isolated A and B responses are well-defined at both inputs;
  at least one joint pair differs from the ordered pair of isolated responses.
  Isolation removes the other satellite's core edges, retaining the candidate's
  remaining rules, evidence, and initialization.
- Recoverable: for each input, from every recurrent state reachable under an
  admissible schedule from `000`, flip any one of the three internal bits once.
  Every admissible continuation must eventually restore the original pair.
  This is a single-disturbance test, not tolerance of continuing noise.
- Structurally tolerant: at least one **common present edge** can be permanently
  deleted for both inputs while preserving each original pair both from `000`
  and from every established recurrent state. The modified structure need not
  itself pass the recovery gate. No sequence of edits is claimed safe.

Structural tolerance is existential over a declared set of four possible
single-edge deletions. Report all passing edges, including whether the edit
simply removes a behaviorally redundant connection. Passing is not learning.

## Monotone trade-off question and stopping rule

Within the all-OR subclass, define D as the number of distinct satellite pairs
across all four inputs from zero. Define R as the number of those inputs whose
response passes the single-bit recovery test. Compare every edge-addition pair
E subset E'. Ask whether R(E') > R(E) necessarily entails D(E') < D(E), and
retain the exact theorem or a counterexample with its assumptions.

Stop after this bounded search and the OR analysis. If no candidate passes all
five gates, report bounded impossibility under these assumptions and identify
the failing gate(s). Do not search larger Boolean truth tables in this iteration.
The next stage should address graded coupling and layer-resolved modes with an
explicit question, not reopen Boolean enumeration indefinitely. The unresolved
full-star all-controls-stable target from OP-02B is retained unchanged.
