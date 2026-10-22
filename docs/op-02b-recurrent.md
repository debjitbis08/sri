# OP-02B — Minimal recurrent structural interference

Status: PARTIAL. This first bounded dynamics experiment is complete. Its OR
witness reduces to reachability. The [second iteration](op-02b-recurrence-necessity.md)
now establishes scoped feedback-dependent stabilization and minimal edge counts.
Useful inference and stronger architectural claims remain open. The original
OP-02 certificate and the first recurrent report remain unchanged.

## Question and acceptance criteria

Can independently simple components, through reciprocal algebraic influence,
develop stable collective response modes not reducible to their isolated responses?

- [x] Two satellites and one stateful core, with bidirectional coupling.
- [x] Input persists as explicit evidence in the collective state.
- [x] Exact synchronous and ordered sequential update semantics.
- [x] Exhaustive fixed-point, cycle, and basin analysis in a bounded family.
- [x] Matched isolated, uncoupled, and feedback-removed controls.
- [x] Explicit separation of interaction, organization, and capability.
- [x] A stable contextual witness requiring recurrence under matched edge deletion; see the second iteration.
- [x] Scoped minimum edge count; see the second iteration.
- [ ] Fully reciprocal witness with a distinct response while all acyclic observations are stable.

No layers, topology variants beyond edge ablations of the star, graded weights,
learning, or dreaming are introduced.

## State and transition semantics

The state is `(x_A, x_B, k, a, b)`, five Boolean coordinates. The two evidence
bits are clamped throughout a run: `x'_A=x_A`, `x'_B=x_B`. The other three bits
are mutable node states, including genuine retained core state `k`. Evidence
is separate from interpretation and is read on every satellite update. This
first model does not let the system rewrite its external evidence.

Four directed edges, in order, are `A→K, B→K, K→A, K→B`. The full model includes
all four. Deleted edges transmit 0. Each node independently chooses one of two
three-argument rules: OR, or parity (XOR). For edge indicators e:

```
k' = H_K(k, e_AK a, e_BK b)
a' = H_A(a, e_KA k, x_A)
b' = H_B(b, e_KB k, x_B)
```

The products above are Boolean gates for deleting edges, not graded weights.
Both rules have identity 0; OR outputs 1 if any argument is 1, XOR outputs 1
if an odd number of arguments are 1. These formulas fully specify their tables.
They are elementary controls, not proposed new algebras.

Under `sync`, all three right-hand sides read the previous state. Under one
of `kab, kba, akb, abk, bka, bak`, nodes update in that order, reading the most
recent values. One recorded transition is a full sweep. The same order repeats
forever. Random, changing, and arbitrary fair asynchronous schedules are outside
the enumeration. A sweep fixed point is also a local fixed point: every node
is written only once, so no changed coordinate can be restored later in the sweep.

## Observable and controls

The primary observable is the exact eventual periodic orbit of the whole state,
plus transient, period, and basin size. Cycles are canonicalized only by rotation,
not by numeric output encoding. A cycle of period greater than one is a complete
analysis result, never a settled response.

For the first contextual test, compare A's final Boolean state between the full
star and the A-only control, requiring both runs to reach fixed points. Use the
same evidence, initial states, rules, and schedule. A-only removes both B/core
edges, keeping A/core reciprocity. B remains in the bookkeeping state but cannot
affect A or the core. Define the symmetric B test with B-only. No periodic
response is silently collapsed into a single output; contextual cycle comparison
is deferred.

Additional controls remove every inter-node edge (uncoupled) or both core-to-
satellite edges (feed-forward). The witness also retains all 16 edge ablations.
The isolated satellite-pair response is the ordered pair of A's output in A-only
and B's output in B-only. This preserves component identity and multiplicity;
it is not OP-02's set-union observable. All comparisons hold the initial state
fixed, since different basins can produce different outputs.

## Exhaustive scope and stopping theorem

`python -m experiments.recurrent` writes [recurrent.json](../reports/recurrent.json).
The experiment covers 8 OR/XOR rule triples × 16 edge masks × 7 schedules ×
4 evidence pairs = 3,584 deterministic maps. Every map is analyzed from all
8 internal states: 28,672 trajectories. This is exhaustive for these rules,
not for all Boolean functions, finite algebras, or recurrent architectures.

With fixed evidence there are eight possible internal states. A deterministic
orbit must repeat after at most eight transitions. On its first repeat, the
prefix is transient and the remainder is the exact cycle. Enumerating every
initial state therefore finds all attractors and partitions each map's eight
states into their basins. This proves termination of the analyzer, not convergence
of the network. There is no cap, randomized sampling, or truncated result.

The report aggregates trajectory periods and distinct attractor periods per
rule/schedule across all edge masks and evidence. Attractors from different maps
are counted separately. It includes complete traces and basin listings for the
stable witness's controls, plus explicit cycle and schedule counterexamples.
Stable-context counts use only the full star's 32 evidence/initial-state cases
per rule/schedule. The tests verify cycle transitions, direct fixed points,
basin totals, the eight-transition bound, and byte-identical report generation.

## Stable contextual witness — and its failure as a stronger result

Use OR at all nodes, full coupling, synchronous updates, evidence `(0,1)`, and
initial internal state `000`. Omitting the unchanged evidence in this trace:

```
000 → 001 → 101 → 111 → 111
```

A ends at 1 with B present, and at 0 in the matched A-only control. The coupled
satellite output is `(1,1)` versus independent `(0,1)`.

| Control | Final internal state from 000 | A's response |
| --- | --- | --- |
| Full reciprocal star | 111 | 1 |
| A-only | 001 | 0 |
| B-only | 101 | 0 |
| Uncoupled satellites/core | 001 | 0 |
| Feed-forward to core | 101 | 0 |
| Acyclic path B→K→A | 111 | 1 |

The full star's `111` fixed point attracts all eight internal initial states
at this evidence. However, `111` also exists in isolated controls for other
initial states. This is a change in basin selection and response, not a newly
created attractor absent from every independent configuration.

More decisively, deleting A→K and K→B leaves the acyclic path B→K→A, which
reproduces the same final response. Thus recurrence is present in the full
witness but is not necessary for its contextual effect.

**Reachability reduction (all-OR class).** Every node retains its previous 1,
and a node becomes 1 only from evidence or a predecessor already at 1. Hence
the eventual active set is exactly the directed reachable closure of initial
active nodes and evidence-driven satellites. Each nonstationary sweep adds at
least one of three internal nodes. All seven schedules therefore reach the
same closure after at most three strict expansions and one no-change check.
Tests compare every mask/evidence/initial state/schedule to an independent graph
closure calculation. At full coupling and zero evidence, both `000` and `111`
are fixed, demonstrating initial-state dependence rather than universal uniqueness.

This negative result is useful: contextual dependence plus fixed-point stability
does not by itself distinguish structural resonance from propagation.

## Cycles and schedule dependence

Take rules `(OR, OR, XOR)` and full coupling. At evidence `(0,0)`, synchronous
updates from `001` give:

```
001 → 101 → 110 → 111 → 110 → ...
```

The period-two attractor refutes unconditional settling in this family.

With the same rules, evidence `(0,1)`, and initial `000`, synchronous updates
settle at `110`, while repeated `kab` sweeps settle at `111`. Both are fixed
points; changing the schedule changes basin selection. Neither result licenses
an interpretation as inference correctness or a useful competing hypothesis.

## First-iteration limits and follow-up

Three distinct mutable nodes and four directed edges are the architectural
minimum *by the requirement* of a core reciprocally connected to two satellites.
Binary states are the smallest nonconstant state alphabet. These observations
do not prove minimality of recurrence-essential interference. The first OR
witness fails that stronger condition. The initial experiment did not search
for a stronger witness; the second iteration now does so with explicit criteria.

Interaction is established in a controlled dynamical sense. Stable organization
is established only as attraction to fixed configurations; in the positive
example it reduces to reachability. Useful inference has not been tested: there
is no task, correctness criterion, held-out evaluation, or capability advantage.

The follow-up now defines the component-level recurrence-necessity criterion,
exhausts the existing OR/XOR family against acyclic controls, and retains this OR
witness as a negative control. See [the second iteration](op-02b-recurrence-necessity.md)
for proofs, exact counts, and the distinction between feedback stabilization and
useful inference. The original whole-state-fixed contextual counts above are
historical; the follow-up consistently allows stable observed coordinates within
otherwise cycling control systems.
