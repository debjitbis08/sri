# Open Problems Register

**Version:** 0.1 · **Status vocabulary:** OPEN, PARTIAL, PROVED (scoped), REFUTED, DEFERRED.  
For each problem, distinguish a mathematical theorem from a bounded enumeration and from randomized evidence. Update the status only with a linked proof, reproducible experiment or counterexample.

## A. Algebra and semantics — highest priority

### OP-01 — What constitutes a response mode? [OPEN, critical]
Can modes be defined structurally (e.g., invariant configurations or relations) without choosing a linear spectral basis? Which definitions extend naturally from finite graphs to cell complexes?

**Tests:** construct non-isomorphic small systems with matched mode inventories; compare what the definition captures.  
**Failure mode:** rebranding arbitrary hidden states as 'modes'.

### OP-02 — Minimal algebraic interference witness [PROVED (scoped); restricted checkpoint complete]
Find a finite algebra and coupled core/satellite instance whose combined observable cannot be explained by a declared independent baseline. Determine smallest states/nodes/edges supporting the effect.

**Current evidence:** [First bounded certificate](op-02-interference.md): XOR cancellation and ternary saturation differ from a declared independent set-union control. Exhaustive search covers commutative monoids on 1–3 states and four feed-forward edge masks. Minimality is scoped to that class. This checkpoint is complete within its declared class; intrinsic nonadditivity was not established. The original certificate remains unchanged as a historical baseline, including its original PARTIAL label. Further structural investigation proceeds under OP-02B.

**Proof target:** exact witness and exhaustive minimality under stated size bounds.  
**Failure mode:** fake interference caused by an arbitrarily chosen numeric output encoding.

### OP-02B — Minimal recurrent structural interference [PROVED (scoped); recurrence checkpoint complete]
Can independently simple components, through reciprocal algebraic influence, develop stable collective response modes not reducible to their isolated responses?

**Scope:** two stateful satellites and one stateful core, bidirectional coupling, persistent evidence, exact schedules, exhaustive bounded attractor analysis, and matched independent controls. Layers, graded weights, structural learning, and dreaming are deferred.

**Current evidence:** [First recurrent experiment](op-02b-recurrent.md) exhausts 3,584 Boolean maps / 28,672 trajectories. The [matched-ablation follow-up](op-02b-recurrence-necessity.md) finds 80 full-star feedback-dependent stable contextual cases, with a scoped three-edge minimum in the expanded ablation family. None of the full-star cases has all acyclic satellite observations stable. Mixed OR/XOR rules exhibit cycles and schedule-dependent fixed-point selection.

**Completed scope:** recurrence necessity and edge minimality under matched deletions. The historical certificates remain unchanged, including their original PARTIAL labels. The stronger fully reciprocal all-controls-stable target remains OPEN; useful inference is not established. Robustness work now proceeds under OP-03.

### OP-03 — Robustness of collective response under perturbation and structural modification [PARTIAL, critical]
Which responses survive variation in initial state, all declared fair update schedules, internal disturbances, and structural edits? Preserve the original well-defined-response obligations: convergence, cycles, uniqueness, and honest stopping semantics.

**Current evidence:** [Exact Boolean robustness analysis](op-03-robustness.md) covers 512 asynchronous transition graphs, with detailed full-star basins and single-bit/edge tests. The principal OP-02B witness settles under the seven old schedules but fails under the fair repeating schedule `B,B,core,A`. Only A-bit flips are guaranteed to recover its response; deleting K→B destroys stabilization.

**Proof target:** distinguish fixed points, attractor recovery, and structural preservation. Specify a jointly robust and distinguishable response family with explicit schedule/initialization/edit assumptions before adding graded coupling or learning.

### OP-04 — The meaning of overlap [OPEN]
When do distinct satellites have a shared mode? Is matching labels insufficient; must a mapping preserve composition, constraints or observable responses?

**Proof target:** a behavioral equivalence/compatibility definition that survives re-encoding of states.

### OP-05 — Structural versus literal resonance [OPEN]
Which algebraic properties justify the term resonance when no periodic oscillation exists? Can one define an invariant collective response, and which conclusions need dynamics?

**Risk:** an attractive analogy with no measurable consequences.

## B. Nested shape, topology and filtering

### OP-06 — Does a nested representation add expressive power? [OPEN]
Compare flat and layered finite systems with the same total states, connections and update rules. Is layering useful beyond bookkeeping?

### OP-07 — Do topological holes matter? [OPEN]
Which candidate response operators can detect noncontractible cycles or higher homology? Compare matched no-hole/cyclic/multi-hole cases. A Betti number is not itself a count of semantic capabilities.

### OP-08 — Diverse exterior, stable but discriminative interior [OPEN]
Characterize response maps that attenuate nuisance perturbations while preserving useful distinctions. How does reconstruction vary with resource limits?

**Proof target:** sensitivity upper bounds on a nuisance class plus distinguishability lower bounds on a designated relevant class.

### OP-09 — Graded inter-layer coupling [OPEN]
Under what coupling laws are inner-layer influences typically stronger without erasing weak decisive outer modes? What is gained over uniform coupling?

### OP-10 — Weak global influences and safe truncation [OPEN, critical]
Can the total omitted effect be bounded given path multiplicity, nonlinear coupling, cycles and amplification? How do we detect when pruning is unsafe?

**Proof target:** an error bound with explicit topology/degree/feedback assumptions and a sharp counterexample when assumptions fail.

## C. Structural learning and memory

### OP-11 — A genuinely local plasticity law [OPEN, critical]
What intrinsic activity/feedback/compatibility state triggers connection strength, incidence or local-rule changes? Can changes be efficient and still meaningful?

### OP-12 — Structural STM with rapid turnover [OPEN]
Can high-frequency structural modification encode recent perturbations without an external episodic buffer? Distinguish learning rate from structural persistence.

### OP-13 — Preserving existing satellites [OPEN, critical]
Can a new satellite modify shared coupling without destroying old response families? What worst-case trade-offs or impossibility results arise for bounded capacity?

### OP-14 — Learning from outcomes, not self-consistency [OPEN]
What mechanisms prevent coherent yet incorrect collective responses from reinforcing themselves? What feedback remains available when explicit human reward is sparse?

### OP-15 — Structural forgetting and dormancy [OPEN]
When does an old mode become hard to activate yet remain recoverable? How can we distinguish dormancy from irreversible loss using counterfactual perturbations?

## D. Endogenous dream and abstraction

### OP-16 — Dream generator with no history store [OPEN, critical]
Define \(Q_\Theta\) using only the current persistent model, plus bounded transient random state. Can dreams be diverse and reflect previously learned structure without replaying raw episodes?

### OP-17 — Random variation without combinatorial search [OPEN]
What local stochastic rules generate meaningful related and cross-satellite perturbations? Can their cost remain bounded? Random sampling alone does not imply efficient discovery.

### OP-18 — Detecting robust commonality [OPEN]
When do repeated whole-core patterns across independently varied satellite activations constitute stable commonality rather than coincidental recurrence?

### OP-19 — Core abstraction without copying a capability [OPEN, critical]
Can structural change encode common response relations while leaving domain-specific states in satellites? Define this condition in observables, not by labeling nodes 'abstract'.

### OP-20 — Transfer beyond training perturbations [OPEN, critical]
Can endogenous consolidation improve held-out combinations/new-domain accommodation compared with no-dream, memorization and random-consolidation controls?

### OP-21 — Dream-induced mode collapse [OPEN]
How do we prevent internally likely patterns from strengthening merely because the system generates them often? Can weakening/competition maintain diverse useful modes?

## E. Computational cost, limits, and practicality

### OP-22 — Expected active-set growth [OPEN, critical]
Can typical inference work depend on an active substructure rather than total stored structure? Under what distributions and degree bounds? Avoid assuming sparsity just because edges are sparse.

### OP-23 — Thinking effort as approximation [OPEN]
Can layered budgets, thresholds and work limits progressively refine responses without restarting? Are deeper runs demonstrably better on prospective tests?

### OP-24 — Integer/bitwise sufficiency [OPEN]
Which candidate algebras admit lookup-table, XOR, bitset or small-integer implementations without losing the important response properties? Benchmark memory traffic as well as arithmetic.

### OP-25 — Bounded persistent storage [OPEN]
Can shared core abstractions reduce duplication or growth? What lower bounds arise from the number of independent distinctions the system must retain? No finite model can retain unlimited arbitrary information losslessly.

### OP-26 — Capability-level demonstrations [OPEN]
Identify increasingly difficult tasks: relational constraints, analogy, variable binding, composition, novel domains and eventually useful language-mediated assistance. Which capabilities arise from learning rather than handcrafted task rules?

### OP-27 — CPU-only end-to-end budget [OPEN]
Can an honest test including routing, relaxation, plasticity, dreaming and storage fit user-class hardware? Compare against conventional small baselines at comparable accuracy and latency.

## F. Methodological dangers and negative controls

### OP-28 — No hidden abstraction oracle [OPEN]
Does the evaluator, perturbation sampler or consolidation rule hard-code the very generalization credited to the system? Pre-register test families and inspect leakage.

### OP-29 — Completeness labels [PARTIAL]
The starter's monotone baseline has a mathematical closure argument, but experiment outputs need explicit completion/truncation status and provenance.

### OP-30 — Refutation-first experiment design [OPEN]
For each hypothesis, state the smallest counterexample that would overturn it. Prefer exhaustive bounded exploration to broad random parameter search when feasible.

## Prioritized next three questions

1. **OP-03:** Characterize schedule, state-perturbation, and structural robustness before expanding the local rules. OP-02 and OP-02B are completed scoped baselines; the stronger full-star all-controls-stable target remains open.
2. **OP-10:** Derive one certified weak-interaction approximation rule for a restricted class, or find a decisive counterexample.
3. **OP-11 + OP-13:** Define a structural modification law and measure retention before introducing endogenous dreams.

## Evidence ledger template

When a problem is advanced, append an entry such as:

```text
Problem: OP-XX
Claim: [one precise sentence]
Status: PROVED (scoped) | EXHAUSTIVE (bounded) | SAMPLE-SUPPORTED | REFUTED
Assumptions: [finite class / algebra / topology / update schedule]
Proof or witness: docs/...
Code revision: [commit]
Experiment: experiments/... [seed, bounds]
Frozen report: reports/...
Counterexamples/limitations: [...]
Next question: [...]
```

A measured correlation is not a theorem; a verified finite example is not a general solution; a coherent collective response is not proof of intelligence.

## Evidence entry — OP-02, first iteration

- Claim: two states and two edges suffice, and are minimal in the declared two-satellite feed-forward class, for disagreement with the set-union control.
- Status: PROVED (scoped), with EXHAUSTIVE (bounded) certificate.
- Proof and limitations: [OP-02 note](op-02-interference.md).
- Experiment: `python -m experiments.interference`; no randomness.
- Frozen report: [interference.json](../reports/interference.json).
- Code revision: initial repository commit containing this evidence entry and the linked certificate.
- Next question: does any candidate survive stronger independent controls?

## Evidence entry — OP-02B, first iteration

- Claim: the Boolean recurrent star supports stable contextual dependence, cycles, and schedule-dependent basin selection; the selected stable OR witness is reducible to reachability.
- Status: EXHAUSTIVE (bounded) dynamics analysis; PROVED (scoped) reachability reduction and analyzer termination. OP-02B remains PARTIAL.
- Proof and controls: [OP-02B note](op-02b-recurrent.md).
- Experiment: `python -m experiments.recurrent`; all declared cases, no randomness or cap.
- Frozen report: [recurrent.json](../reports/recurrent.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Next question: can a stable contextual response survive matched controls while requiring recurrence?

## Evidence entry — OP-02B, matched-ablation follow-up

- Claim: feedback can stabilize contextual satellite responses not reproduced by matched acyclic edge deletions; three edges are minimal in the declared star-ablation family.
- Status: PROVED (scoped) witness and edge lower bound; EXHAUSTIVE (bounded) classification. OP-02B remains PARTIAL for stronger architectural claims.
- Proof and controls: [Recurrence necessity](op-02b-recurrence-necessity.md), including a hidden-core-cycle false positive and a stricter three-edge boundary case.
- Experiment: `python -m experiments.recurrence_necessity`; 28,672 candidate cases, no randomness or cap.
- Frozen report: [recurrence-necessity.json](../reports/recurrence-necessity.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limitation: 80 full-star cases pass the declared test; zero pass while all acyclic satellite observations are stable. No new capability claim.
- Next question: can a fully reciprocal candidate support a stronger response distinction with stable controls and explicit schedule robustness?

## Evidence entry — OP-03, fair-schedule robustness

- Claim: exact labeled-transition analysis decides all node-fair atomic schedules in the Boolean model; the principal OP-02B witness is not robust to this schedule class, all one-bit disturbances, or all one-edge deletions.
- Status: PROVED (scoped) fair-SCC criterion and counterexamples; EXHAUSTIVE (bounded) basin and perturbation analysis.
- Proof: [OP-03 robustness](op-03-robustness.md).
- Experiment: `python -m experiments.robustness`; 512 graphs, 32 full-star cases with 256 initial states, no randomness or cap.
- Frozen report: [robustness.json](../reports/robustness.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limits: clamped evidence, atomic node fairness, one-time internal-bit flips or edge deletions; no repeated structural learning or graded coupling.
- Next question: which nontrivial response family satisfies an explicitly declared combination of robustness and distinguishability requirements?
