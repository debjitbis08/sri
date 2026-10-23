# Open Problems Register

**Version:** 0.1 · **Status vocabulary:** OPEN, PARTIAL, PROVED (scoped), EXHAUSTIVE (bounded), REFUTED, DEFERRED.
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

### OP-03B — Stability–expressivity trade-offs [EXHAUSTIVE (bounded); scoped investigation complete]
Can the current Boolean family support stable, distinguishable, contextual responses that recover from single-bit disturbances and tolerate a common structural edit?

**Contract:** [Declared two-input response-family gates](op-03b-contract.md): zero initialization, clamped input, two separately reported schedule regimes, exact observations, one-bit disturbances, and one-time edge deletion. The stronger OP-02B full-star all-controls-stable target remains open.

**Result:** [OP-03B analysis](op-03b-response-family.md) exhausts 768 families per regime. None passes: every recoverable output is `11`, so no two recoverable responses are distinguishable. Within the all-OR subclass, the restricted theorem `D + R = 5` links distinguishable responses to recoverable inputs.

**Next obligation:** move to graded coupling and layer-resolved modes with an explicit correction/recovery question. Do not expand Boolean enumeration or introduce STM learning in this stage.

### OP-04 — Comparative modeling before architecture selection [OPEN]
Which definitions of modes, nesting, collective interaction, and core substrate support the intended responses, and what persistence, approximation, and execution guarantees follow?

**First comparison complete (bounded):** [v1 results](comparison-v1-results.md) evaluate spectral, affine F3, graded higher-arity, and cellular cochain representatives in 50 cases under a [declared contract](comparison-v1-contract.md). Driven correction, post-input retention, and released-state recovery separate sharply. Exact bounds, ties, retained-free-coordinate failures, and topology/approximation counterexamples are recorded. None establishes autonomous selective core mediation or learned reusable structure; OP-04 remains OPEN.

**Combined follow-up complete (bounded):** [graded cellular v1](graded-cellular-v1-results.md) tests 60 configurations, 4,860 basin samples and 1,872 disturbances. The combined proxy has separated locally restoring modes under weak/moderate context coupling; strong coupling, attachment penalties and large disturbances expose failures. Graded layer connections, multiple layers, full structured modules, an autonomous core and learning are absent and receive no credit. Architecture selection remains open; this result is not a failure or success of those omitted features.

**Current note:** [Interaction semantics before OP-04](interaction-semantics.md) treats hard and graded compatibility, joint transformation, and local-to-global agreement as possible aspects of one collective mechanism, including higher-arity interactions. It separates admissibility, perturbation/response realization, and structural learning; no candidate has been selected.

**New obligation:** define collective transformation without requiring an explicitly constructed global configuration space, and evaluate the core as a selective compatibility mediator. Distinguish changing admissibility from changing reachability or transiently available responses. A core that merely conjoins restrictions cannot enable a previously forbidden satellite pair under an unchanged relation; enabling mediation needs an explicit mechanism and matched comparison.

**Semantic evaluation:** [Representation and mediation obligations](collective-interaction-evaluation.md) compares implicit global relations, supported joint transformations, and overlap-based assembly as descriptions of one mechanism. It derives a necessary interface-sufficiency condition and its semantic converse, separates compatibility mediation from realization mediation, and proposes a conditional representation direction. The worked example below evaluates one limited instantiation without selecting a general update algebra.

**Worked example:** [Collective role binding](mediation-example.md) now implements a joint constructor, direct-binding control, retained capacity incompatibility, and a same-evidence context intervention. Its frozen report contains six fixed cases; no full OP-04 model is selected. The core records a constructed binding, and a generic matching solver reproduces its responses. Autonomous core mediation, recovery, grading, topology, and learning remain unproved. The next obligation is a structural contribution beyond this binding certificate.

**Withdrawn proposal:** the [signed-threshold contract](op-04-contract.md) is superseded. Its equations, grid, and gates are not implementation requirements. Input/output modules specify interfaces, and layers specify response modes, not a processing pathway.

**Next obligation:** use the [comparative modeling protocol](model-comparison.md) to test whether the combined proxy's restoration survives richer module/layer structure and an identifiable core contribution. Specify those missing features before testing them, rather than extrapolate from the proxy. Do not infer that tested instances exhaust their families or that native costs are directly comparable. Preserve OP-02 through OP-03B and the unresolved OP-02B target; do not resume the withdrawn threshold search.

### OP-04A — The meaning of overlap [OPEN; formerly OP-04]
The original OP-04 question is retained here; its identifier changed when the OP-04 recovery experiment was specified. No result or scope was discarded.

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

1. **OP-04, informed by OP-01 + OP-04A + OP-06 + OP-09:** Build on the [completed first comparison](comparison-v1-results.md); investigate reconstructible persistent distinctions and a structural core contribution under the [comparative protocol](model-comparison.md). Retain all restricted controls.
2. **OP-11 + OP-13:** Keep changes to configuration spaces and couplings distinct from transient response selection. Acquisition, retention, and held-out reuse need a separate future learning contract; no response semantics currently establishes them.
3. **OP-10:** Derive one certified weak-interaction approximation rule for a restricted class, or find a decisive counterexample. This is a separate computational track, not a prerequisite for the first plasticity experiment.

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

## Evidence entry — OP-03B, bounded response-family investigation

- Claim: under the declared contract, the current three-node OR/XOR family has no distinguishable pair of recoverable responses in either schedule regime; the all-OR subclass obeys `D + R = 5`.
- Status: EXHAUSTIVE (bounded) impossibility; PROVED (scoped) monotone trade-off theorem. This narrow iteration is complete.
- Contract and proof: [OP-03B contract](op-03b-contract.md), [results](op-03b-response-family.md).
- Experiment: `python -m experiments.response_family`; 768 candidate families per regime, no randomness or rollout cap.
- Frozen report: [response-family.json](../reports/response-family.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limits: fixed Boolean local rules, three internal nodes, zero initial state, fixed evidence pairs, declared schedules and one-shot perturbations. No general robustness/expressivity impossibility is claimed.
- Next question: can an explicitly defined graded, layer-resolved model correct disturbances while retaining two contextual response signatures?

## Evidence entry — selective satellite mediation v1 (OP-04, OP-13, OP-20)

- Claim: in the declared two-satellite, four-generator, 256-hypothesis class, exact version-space learning of transformation gates repairs held-out and unseen-context conflicts with zero constraint/path violations and zero false-feasible predictions, while retaining every observed response set.
- Status: PROVED (scoped) BFS exactness, retention, order independence, certain/possible containment, core-free equivalence and an empty all-enabled obstruction class; EXHAUSTIVE (bounded) oracle check over 147,456 cases; SAMPLE-SUPPORTED transfer over 5 frozen splits × 5 orders × 4 teachers. All 8 declared transfer targets pass.
- Contract and results: [contract](selective-satellite-mediation-v1-contract.md), [results](selective-satellite-mediation-v1-results.md).
- Experiment: `.venv/bin/python -m experiments.selective_mediation`; frozen [split manifest](../reports/selective-satellite-mediation-v1-splits.json), no cap.
- Frozen report: [selective-satellite-mediation-v1.json](../reports/selective-satellite-mediation-v1.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limits: the positive-only ablation executes identically (exclusions shrink V but the smallest-id prior does the selecting); no mediation alone meets the accuracy and false-feasible conditions in some cells; T3 is the initial selection; T1's unseen context has no permission exclusions; the core-free control is equivalent, so no core advantage; equality target only; complete clean supervision; centralized search.
- Next question: does learned selectivity survive a non-equality compatibility relation, or realization inside graded satellite dynamics, with a selection/evaluation rule that separates exclusion evidence from a conservative prior?

## Evidence entry — learned reciprocal participation v1, phase A (OP-04)

- Claim: in the declared signed-energy candidate, none of the declared teachers T1–T3 satisfies the mechanism gate. Each realizes input revision, core-dependent participation (no fixed core state reproduces it), core-mediated cross-satellite clamp influence and selective participation, but input revision never changes the response set (81/81 queries equal the input-frozen control).
- Status: REFUTED in the declared teacher family; EXHAUSTIVE (bounded) phase-A diagnostics over 81 queries × 4 teachers; oracle-verified dynamics on 59,049 cases; EXHAUSTIVE (bounded) check that none of the 225 hypotheses with single-satellite input couplings shows a feedback effect (no analytic proof).
- Contract and results: [contract](learned-reciprocal-participation-v1-contract.md), [results](learned-reciprocal-participation-v1-results.md).
- Experiment: `.venv/bin/python -m experiments.reciprocal_participation`; frozen [split manifest](../reports/learned-reciprocal-participation-v1-splits.json), no cap.
- Frozen report: [phase A](../reports/learned-reciprocal-participation-v1-phase-a.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limits: phase B (learning) not run under the stop rule; 312 hypotheses with feedback effects exist in the coefficient family but were not gate-tested; mode-level energy candidate only.
- Next question: under a new prospective contract, do declared teachers with an input coordinate shared by both satellites satisfy feedback necessity together with the other six conditions?

## Evidence entry — shared input interpretation v1 (OP-04)

- Claim: in the signed-energy candidate, satellite feedback into a shared input coordinate (S1) or into internally coupled input coordinates (C2) gives an interventional set-level chain A → input → B. A's clamp changes the input, the recipient-input clamp changes B, and the A→B contrast vanishes under directed feedback removal, under input fixed at evidence, and under removal of each chain link. Both teachers also pass the unchanged v1 gate conditions 1–7.
- Status: EXHAUSTIVE (bounded) over 81 queries × 6 declared teachers, all ordered clamp pairs and recipient values; oracle-verified dynamics (177,147 cases) and every scoring run; v1 regression controls reproduced exactly. S2, S3 (core bypass makes the input route redundant), C1 and C3 (no contrast vanishes under both controls) fail.
- Contract and results: [contract](shared-input-mediation-v1-contract.md), [results](shared-input-mediation-v1-results.md).
- Experiment: `.venv/bin/python -m experiments.shared_input_mediation`; frozen [manifest](../reports/shared-input-mediation-v1-manifest.json).
- Frozen report: [gate](../reports/shared-input-mediation-v1-gate.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limits: the chains make a B response available rather than forcing it; they persist without the core, so they are not core-mediated; a core bypass exists at non-evidence input values; C2's canonical-start feedback effects lie on different queries from its chain; two of six hand-declared teachers; no learning.
- Next question: under a new contract, can learning from outcome sets acquire the S1 or C2 couplings, including the chain links, from withheld-query feedback?

## Evidence entry — learned input mediation v1 (OP-11, OP-14, OP-20)

- Claim: in the 2,187-vector signed-energy space, canonical outputs on all 81 queries determine each declared teacher's battery mechanism up to the core-sign symmetry; from 15 axis-evidence training queries, exact version-space learning certifies all held-out outputs and the mechanism for S1 and C2 in every split. With outputs alone, the negative control S2 leaves 5 survivors per split that match all within-context held-out outputs but carry a false input chain. The teacher and its mirror survive too, so the exact learner reports mechanism uncertainty rather than certifying the false chain; clamp supervision with internal readouts removes the false survivors. A block-local learner recovers S1/C2 behaviour in 25/25 runs with clamp supervision; with outputs alone it recovers C2 in 9/25 and S1 in 0/25.
- Status: EXHAUSTIVE (bounded) ceilings and version-space results over the declared grid; oracle-verified on a declared sample of 54 hypotheses; local-learner results SAMPLE-SUPPORTED over 25 streams per arm.
- Contract and results: [contract](learned-input-mediation-v1-contract.md), [results](learned-input-mediation-v1-results.md).
- Experiment: `.venv/bin/python -m experiments.mechanism_learning`; frozen [manifest](../reports/learned-input-mediation-v1-manifest.json); v1 split manifest reused.
- Frozen reports: [battery summary](../reports/learned-input-mediation-v1-battery-summary.json), [ceilings](../reports/learned-input-mediation-v1-ceiling.json), [learning](../reports/learned-input-mediation-v1-learning.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limits: finite battery equivalence only; realizable noise-free complete-set supervision from four teachers known to the researchers; arm I mixes intervention with internal readouts; the local learner uses global error and is unreliable with outputs alone; C2 arm I misses the 95% retention target at 77.2%, a pooled metric that mixes items not yet acquired with forgetting (19 break events among 598 failed checks).
- Next question: with the local rule fixed, can the mechanism be learned reliably from less privileged feedback (mixed-input coverage, or clamps with external readout only) while preserving earlier behaviour? Addressed by [local learning supervision v1](local-learning-supervision-v1-results.md) below. Whether a rule needs coordinated cross-block edits remains a hypothesis.

## Evidence entry — local learning supervision v1 (OP-11, OP-14, OP-20)

- Claim: with the v1 block-local rule fixed, only clamp supervision with internal input readouts (I) yields reliable recovery (25/25 for S1 and C2) with end-of-stream preservation (100% final retention, 0 forgotten). Clamps with partner-mode readout only (IB: 5/25, 7/25) and 12 additional mixed-evidence queries (O+: 1/25, 13/25) fail. On the same streams, exact version-space learning reaches the teacher-plus-mirror class in every split under IB and O+ and removes S2's false-chain survivors, so those streams are informationally sufficient and the local rule does not use them. v1's 77.2% C2 retention was delayed acquisition and temporary loss, not end-of-stream forgetting.
- Status: EXHAUSTIVE (bounded) ceilings and version-space results over the declared grid; v1 arms O and I reproduced exactly; projection verified against the independent oracle on 54 hypotheses; local-learner results SAMPLE-SUPPORTED over 25 streams per condition.
- Contract and results: [contract](local-learning-supervision-v1-contract.md), [results](local-learning-supervision-v1-results.md).
- Experiment: `.venv/bin/python -m experiments.local_supervision`; frozen [manifest](../reports/local-learning-supervision-v1-manifest.json); v1 battery and split manifest reused.
- Frozen reports: [ceilings](../reports/local-learning-supervision-v1-ceiling.json), [learning](../reports/local-learning-supervision-v1-learning.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limits: one fixed rule; O+ confounds mixed evidence with extra updates; IB and I differ in readout detail and tuple-level loss volume; L3 is end-of-stream, so I's pass coexists with delayed acquisition and temporary losses; four hand-chosen teachers, noise-free complete-set feedback, one 12-query coverage subset, finite battery.
- Next question: with IB supervision held fixed, why do the block learner's updates fail: no improving single-block edit, improving edits that break earlier items, or progress that needs neutral or worsening moves? Then vary one rule property at a time, counting the extra candidate evaluations. Replay is a diagnostic comparator only, not a proposed mechanism (retention should live in learned structure, without an external experience store). Addressed by [local update diagnosis v1](local-update-diagnosis-v1-results.md) below.

## Evidence entry — local update diagnosis v1 (OP-11, OP-14, OP-20)

- Claim: under IB supervision the block learner's failing updates (S1 255, C2 197 of 375) have mixed causes: improving non-breaking commits that leave the example inexact (40%, 32%), preservation conflicts where every improving single-block edit breaks an earlier item (22%, 35%; the source of 76% and 86% of loss events), stuck states a two-block edit would improve (21%, 18%), and ignored preserving alternatives (12%, 11%). Under a frozen example the teacher class usually needs a plateau move (72%, 64% of visited states). No single-property variant meets L1–L3 for both teachers: SCOPE (two-block edits) recovers S1 19/25 at 10× candidates; REPEAT 15/25 for both at about 1.5×; REPLAY (history-aware selection, a comparator with an external store) suffices for C2 (23/25, 96.8% retention) but not S1.
- Status: EXHAUSTIVE (bounded) per-decision classification and exact frozen-example reachability over all visited states; B0 reproduced in lockstep and against the frozen report; variants SAMPLE-SUPPORTED over 25 streams each.
- Contract and results: [contract](local-update-diagnosis-v1-contract.md), [results](local-update-diagnosis-v1-results.md).
- Experiment: `.venv/bin/python -m experiments.update_diagnosis`; frozen [manifest](../reports/local-update-diagnosis-v1-manifest.json).
- Frozen reports: [diagnosis](../reports/local-update-diagnosis-v1-diagnosis.json), [variants](../reports/local-update-diagnosis-v1-variants.json).
- Code revision: the commit introducing this evidence entry, its accompanying source, tests, and frozen report.
- Limits: diagnostics use evaluator-only knowledge (teacher class, earlier-item correctness); graph categories hold one example fixed; variants are not compute-matched and change one property each; REPLAY is outside the target architecture; four hand-chosen teachers, noise-free complete-set IB feedback.
- Next question: can a mechanism inside the learned state, without an experience store, supply the preservation signal that REPLAY gets from stored feedback, and do declared plateau moves help reach the teacher class?

## Evidence entry — core abstraction v1 (OP-13, OP-19, OP-20)

- Claim: no core qualifies for S1 or C2 in any of 10 split/assignment runs. The teacher core transfers on true private blocks but violates isolated-response retention after assembly; preserving cores give no transfer there. S1 also exhibits assembly interference in 9 runs. This is a conflict between the declared preservation and transfer targets, not evidence that the core lacks capacity or that dreaming fails.
- Status: EXHAUSTIVE (bounded), 124 surviving assemblies × 9 cores for four teachers; information classes over all 2,187 vectors and 10 split/assignment conditions. All reported correctness gates pass.
- Contract and results: [contract](core-abstraction-v1-contract.md), [results](core-abstraction-v1-results.md). Reproduce with `.venv/bin/python -m experiments.core_abstraction`.
- Frozen reports: [manifest](../reports/core-abstraction-v1-manifest.json), [ceiling](../reports/core-abstraction-v1-ceiling.json), [information](../reports/core-abstraction-v1-information.json).
- Limits: evaluator-only core search; isolated observations contain no information about core identity; S2's sole qualifying run recovers its existing core. The post-hoc tally on true private blocks finds only teacher-sign-class solutions. Neither acquisition nor dreaming was tested. The contract and manifest were written before scoring but were not committed before it.
- Next question: can context-consistent preservation and a bounded amount of connected waking evidence support donor-dependent transfer from learned constraints? See the unimplemented [v2 draft](core-abstraction-v2-draft.md). Endogenous dreaming and the preservation-signal, plateau-move, and larger-edit investigations remain deferred.
