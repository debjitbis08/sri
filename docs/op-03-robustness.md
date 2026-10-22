# OP-03 — Robustness of collective response

Status: PARTIAL. This first iteration exactly characterizes fair single-node
scheduling, one-bit state recovery, and one-edge deletion for the existing
Boolean model. OP-02B is a successful, completed **scoped recurrence-necessity**
checkpoint; its stronger full-star all-controls-stable target remains unresolved.
No graded couplings, learning rules, or new Boolean functions are introduced.

## Three different properties

| Property | Test used here |
| --- | --- |
| Fixed-point stability | Every local update leaves the state unchanged |
| Attractor robustness | After one internal-bit flip, every fair execution eventually returns to the same state or satellite response |
| Structural robustness | After permanently deleting one edge at an established fixed point, every fair execution eventually preserves the satellite response |

The response is still the ordered satellite pair `(a,b)`; no semantic
interpretation is assigned. State recovery and response recovery are reported
separately. Returning to the same pair does not require the hidden core to return
to its original state or even to stop changing. These are discrete, scoped tests,
not a claim of continuous structural stability or resistance to arbitrary noise.

## Fair schedules beyond seven fixed orders

Use the same state `(x_A,x_B,k,a,b)`, rules, and edge masks as OP-02B. Evidence
remains clamped. At each step exactly one of `k,a,b` updates from the current
state; the other two retain their values. A schedule is **node-fair** if each
node is selected infinitely often. There is no maximum delay between selections.
No-op selections count as updates. The model's `schedule` field is not used by
the asynchronous analyzer; it builds all three possible atomic transitions.

This class includes arbitrary changing orders, repeated selections of a node,
and arbitrarily long finite delays. It includes the six repeated ordered sweeps,
but not simultaneous synchronous updates. The original synchronous dynamics are
retained as a separate comparator, not silently interpreted as asynchronous.
Fairness here is about selecting nodes, not about taking every enabled transition
from each individual state; stronger fairness assumptions are not implied.

### Exact graph criterion

For each fixed evidence pair there are eight internal states and 24 labeled
transitions (including no-ops). Compute all reachable states and all maximal
strongly connected components (SCCs). An SCC can support a fair infinite execution
exactly when it contains an internal edge with **each of the three node labels**.
Such an SCC need not be a bottom component: a fair execution can avoid an exit.

**Proof.** In any infinite execution, the states recurring infinitely often
form a strongly connected set. Node fairness ensures an infinitely recurring
edge of each label in that set. Its enclosing maximal SCC therefore has all
three labels. Conversely, inside any strongly connected component with all
three labels, concatenate internal paths that visit every state and one edge
of every label, then return to the start. Repeating that finite tour is fair.
A finite reachable prefix connects any eligible initial state to the tour.

Consequences for a given initial state:

- Every fair execution settles to some full-state fixed point iff every reachable
  fair SCC is a singleton. A singleton fair SCC has all three no-op edges.
- Every fair execution eventually has a specified satellite response iff **every
  state in every reachable fair SCC** has that projection.
- A fixed point's possible state basin contains starts with a path to it: the
  finite path followed by repeated updates of all nodes is a fair execution.
- Its guaranteed state basin contains starts from which every fair execution
  reaches it. The guaranteed response basin may be larger.

This decides unbounded-schedule properties by finite graph analysis, not by a
finite rollout budget. Fair convergence need not have a uniform time bound when
update delays are unbounded. The analyzer itself always terminates on this finite
graph. No floating-point tolerance or randomized sampling is involved.

## Exact scope and reproducibility

Run `python -m experiments.robustness` to regenerate
[robustness.json](../reports/robustness.json). The analyzer builds all
8 rule triples × 16 edge masks × 4 evidence pairs = **512 labeled graphs**.
The report gives detailed basin/robustness results for the 32 full-star graphs,
covering all 256 internal initial-state cases. Edge-deletion checks use the same
cached graph family. The remaining masks are supporting controls, not additional
full-star cases in the summary counts.

For each full-star fixed point, the report records all three one-bit disturbances
and all four single-edge deletions. Evidence bits are not disturbed. A deletion
happens once, permanently, after the system has reached that fixed point; the
post-deletion state is initially unchanged. A second field restarts the modified
network from `000`, separating preservation of an established response from its
reconstruction. Combined state/edge changes, repeated edits, additions to an
already full star, changing evidence, and graded weights are outside this test.

| Full-star result | Count |
| --- | ---: |
| Evidence/initial-state cases | 256 |
| Every fair execution reaches some fixed point | 100 |
| Every fair execution has the same stable satellite pair | 88 |
| Fixed points across rule/evidence combinations | 43 |
| Fixed points recovering the response after every one-bit disturbance, under every fair schedule | 6 |
| Fixed points preserving the response after every single-edge deletion, under every fair schedule | 20 |

The 100 and 88 count different properties and must not be subtracted to infer
an overlap. A stable observed pair may coexist with core cycling. The 6 and 20
are counts of labeled fixed points, not distinct mechanisms or learned abilities.

Tests check the atomic semantics against all six ordered sweeps, replay fair
lassos across all graphs, compare SCC results with exhaustive recurrent-state
subsets for the full stars, check the OR class against graph reachability, and
verify byte-identical report regeneration.

## The main OP-02B witness is not fair-schedule robust

Keep `(OR,OR,XOR)`, evidence `(0,1)`, full coupling, and start `000`. All seven
old schedules settle: synchronous updates give `110`, while every repeated
ordered sweep gives `111`. But the following schedule is also fair:

```
repeat: B, B, core, A

000 --B--> 001 --B--> 000 --core--> 000 --A--> 000 --B--> ...
```

The core never observes B's pulse. Each node updates infinitely often, yet the
satellite response keeps oscillating. The two-state set `{000,001}` is a fair SCC
with an exit to other states; a bottom-SCC-only analysis would miss it.

From zero initialization, the reachable fair SCCs are `{000,001}` and `{111}`.
Thus fair atomic updates may fail to settle or may settle at `111`; `110` is not
reachable from `000` under these atomic semantics. That does not contradict the
synchronous result, since simultaneous updates define a different transition
relation. The report includes an executable fair lasso and the short schedule.

### Basins and recovery

| Fixed point | Synchronous basin size | Possible atomic fair basin | Guaranteed atomic fair basin |
| --- | ---: | --- | --- |
| 110 | 5 | 010, 011, 100, 110 | 100, 110 |
| 111 | 3 | 000, 001, 010, 011, 101, 111 | 101, 111 |

Possible basins overlap because the update schedule can select different outcomes.
Guaranteed response basins equal guaranteed state basins for these two points.

From either fixed point, flipping A is repaired under every fair schedule. Flipping
B moves directly to the other fixed point, changing the response permanently.
Flipping the core does not guarantee recovery either. These fixed points therefore
fail the all-one-bit recovery test despite their nontrivial synchronous basins.

### Structural preservation differs from reconstruction

At either established fixed point, deleting A→K, B→K, or K→A preserves that
fixed state. In contrast, deleting K→B makes `b'=b XOR 1`, so B toggles whenever
selected and cannot retain its response under a fair schedule. The witness is
not robust to all single-edge deletions.

Restarting from zero after any one of the four deletions does not guarantee the
original target pair across all fair schedules. Persistence after an edit can
therefore overstate the ability to reconstruct an interpretation after reset.
The preservation result alone is insufficient as a structural-learning guarantee.

## Positive controls and limits on distinguishability

All-OR networks converge under every fair atomic schedule to the same directed
reachability closure. Inflationarity permits only finitely many strict bit
changes; fairness eventually propagates every available activation. This gives
an analytic positive control independent of the SCC machinery.

Four of the six response-recovery successes are all-OR fixed points `111`, one
per evidence pair. Their guaranteed state basins have size seven for evidence
`00`, and eight for each nonzero evidence pair. The other two successes use
`(XOR,OR,OR)` with evidence `11`, at states `011` and `111`: the satellite response
is robust, but flipping the core moves to the other fixed point rather than
restoring the original state. This illustrates why full-state and observed
recovery must be distinguished.

From zero initialization, all-OR robustly distinguishes zero evidence (pair `00`)
from nonzero evidence (pair `11`) across all fair schedules. It does not distinguish
`01`, `10`, and `11`. Moreover, at zero evidence a nonzero initial internal state
can permanently activate the system, so this distinction is not initialization-
robust. The report lists certified zero-start responses for all four evidence
pairs and every rule triple; an uncertified entry is not treated as a response
category or counted as a useful distinction.

These results establish neither a task nor a capability advantage. Robust latching
and propagation remain controls, not the sought learned structural organization.

## Next obligation

The recurrence result is retained, but its principal witness fails broad fair
scheduling, one-bit response recovery, and universal single-edge preservation.
Before expanding the algebra, specify which robustness conditions an SRI candidate
must satisfy together: a stable, distinguishable response family; an explicit
initial-state domain; allowed schedules; and allowed structural modifications.
Keep the full-star all-controls-stable criterion as an unresolved stronger target,
not something to relax to obtain more witnesses.

Graded coupling should be introduced only with a stated perturbation question
and metric. Dynamic structural learning remains untested: a one-time deletion
of a fixed binary network does not show that repeated STM modifications preserve
responses during ongoing inference.
