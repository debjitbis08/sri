# Structural Resonance Intelligence (SRI): Research Program

**Status:** Research hypothesis and program specification, v0.1 (2026-10-23)  
**Scope:** Mathematical foundations, executable finite models, structural learning, and CPU resource analysis.  
**Claims policy:** Nothing in this document establishes intelligence, compositional generalization, a biological mechanism, or better efficiency than existing models.

## 1. Motivation

Can a useful, continuously developing personal intelligence be built as a self-contained computational object that runs on an ordinary consumer CPU, without relying on dense matrix multiplication as its fundamental inference primitive, remote infrastructure, or an external episodic-memory/replay database?

The long-term product ambition is a reasonably capable shared foundation that develops characteristic ways of reasoning through interaction with one individual. Immediate usefulness matters: the user should not have to train an unintelligent seed for months. This ambition motivates the project but is not an experimental finding or a near-term deliverable. The mathematical project comes first.

Instead of assuming intelligence is a sequence of learned dense numerical transformations, we investigate a **coupled system of evolving finite structures**. A comparatively general **core** interacts with specialized **satellites**. Their interactions may be context-dependent and nonadditive. Input interpretation participates in feedback. Common patterns across the *whole responses* of modules might be consolidated into the core, without copying every satellite's specialized knowledge into it.

This program is separate from the native MCS/relational-web physics program, even if techniques and mathematical intuitions overlap. No results from that program are imported as theorems here.

## 2. Architectural picture: hypotheses, not axioms of intelligence

### 2.1 Core and satellites

A candidate system is a finite collection

\[
\mathcal I=(K,S_1,\ldots,S_n;\mathcal C),
\]

where \(K\) is a learned compositional core, \(S_i\) are specialized learned modules, and \(\mathcal C\) is the collection of intra- and inter-module couplings. Some satellites may mediate input and output. Another satellite implements structurally plastic short-term memory (STM). The boundaries between modules need not be permanent; neither attach/detach semantics nor independent executability is required.

The core's *hypothesized* distinct role is to mediate collective responses and consolidate shared abstractions. Specialized content may remain in satellites. The core need not memorize the complete knowledge of each satellite.

**Refined mediation hypothesis (unproved):** the core is a dynamically evolving structural mediator enabling selective collective compatibility among specialized satellite responses, including configurations that cannot coexist without its participation. Abstraction may consist in acquiring reusable compatibility-mediating structure, rather than extracting similarities alone. Joint interaction may transform the participating configurations and thereby change their currently available responses. Transient core evolution remains distinct from persistent structural learning. This hypothesis requires a declared core-free comparison, preservation of meaningful incompatibilities, and held-out reuse; it does not establish a constraint-satisfaction architecture. See the [interaction-semantics note](interaction-semantics.md).

### 2.2 Layers are response modes, not abstraction levels

Each module may consist of nested strata of modes or of a discretized topological structure. The 'concentric shells' metaphor suggests geometric families to investigate; literal spheres are **not required**. Layers are families of response modes with different coupling profiles, **not ranks of abstraction**. Abstraction is assessed over responses of the whole module/coupled system.

Candidate substrates include finite graphs, cell/simplicial complexes, annuli, tori, higher-genus surfaces and non-geometric finite algebras. Having topological holes is a hypothesis to test against matched, simpler controls. Topology alone does not determine a response law or semantics.

### 2.3 Graded coupling and structural resonance

Every layer may influence others, with stronger *permitted* inter-module coupling near the interior and weaker permitted coupling toward the exterior. Actual influence additionally depends on compatibility and current activation. Weak excitations can exist mathematically even if inference drops them under an explicit approximation budget.

'Resonance' is **structural**, not necessarily temporal oscillation. A perturbation excites modes; simultaneous coupled responses can reinforce, suppress or transform one another. Overlap denotes compatible modes or relations; interference denotes their joint effect. A useful response might be a fixed configuration, a recurrent orbit, a constraint-compatible assignment, or another rigorously defined invariant; we do not prescribe fixed points in advance.

Input has two roles: (a) immutable external evidence for the current inference and (b) revisable internal interpretation. The latter remains coupled to core and satellites throughout inference. Output structures participate in the same architecture, but an output or action requires an explicitly defined readout/decision semantics.

### 2.4 Plasticity and memories are intrinsic

Both STM and long-term memory (LTM) are modifications within the model itself. STM is a highly plastic satellite: it changes its structure more frequently, has stronger sensitivity to waking perturbations, and usually less durable modifications. It may have relatively thin separation between exterior and interior response families. Thin geometry alone does **not** prove fast memory decay or rapid learning. LTM is expressed in more persistent satellite configurations and shared core abstractions. Plasticity rate, persistence and resistance to modification are independent quantities to measure.

No separate historical replay store is needed or permitted in the target architecture. During endogenous dreaming, internally generated perturbations are drawn from **current learned structure**, possibly biased by its recent plastic changes. Distinctions not encoded in that structure cannot be reconstructed exactly. Transient inference state and bounded instrumentation used *for experiments* do not constitute a production episodic memory system; all experiments must state this separation clearly.

### 2.5 Wake, dream and varying thinking effort

Wake: external perturbation → joint inference → feedback/outcome → local structural changes where justified.

Dream: internally sampled perturbation → joint inference → measure recurring/robust whole-module patterns → possible changes in learned structure/core. Dreaming must not rely on historical perturbation records or an external generative model. Randomized variations are *not* exhaustive abstraction search. Repetition alone is insufficient evidence of a meaningful abstraction, since the dynamics may reinforce its own biases.

Thinking effort is an approximation budget, potentially specifying per-module layer reach, minimum interaction importance, and iteration/work limits. More expensive inference may revisit weaker, exterior, or previously inactive contributions. A weak signal can be decisive after feedback, so pruning requires validation or error controls; 'deep' does not automatically mean correct.

## 3. Formal questions to resolve before choosing an algebra

**Current method:** compare modeling choices before choosing a substrate. Mode
identity, nesting, joint composition, core representation, and persistence/error
bounds remain open. The [comparative modeling protocol](model-comparison.md)
specifies candidate cards and a common evidence matrix. The binding example and
the proposed structural-transformation direction do not select an architecture.

We deliberately leave the following objects unspecified as alternative candidates:

- **Structural category:** graph, finite cell complex, layered relational system, or other finite object.
- **Local state sets:** boolean, finite integer alphabets, phase-like discrete states, or other finite algebras.
- **Interaction law:** family of operators/relations determining compatibility and joint effects, potentially order-sensitive, noncommutative, or nonlinear.
- **Global response semantics:** fixed points, constraint satisfaction, periodic modes, equilibria, or finite terminated inference with status.
- **Learning transform:** persistent local changes to mode states, link strengths, incidence structure or local laws.
- **Observable equivalence:** what responses and counterfactual perturbations must be preserved by consolidation.

One broad notation is

\[
 z' = T_{\Theta,x}(z), \qquad y=O_\Theta(x,z),
\]

where \(\Theta\) is the *persisted learned model structure*, \(x\) is external evidence, and \(z\) is transient coupled activity (including the revisable interpretation of \(x\)). There may be several admissible outputs/settled states. We will not conflate a capped trajectory with a proved fixed point.

An intrinsic learning step may be written

\[
 \Theta' = L_{\text{wake}}(\Theta,x,z,\text{feedback})
\]

and endogenous dreaming as

\[
 \tilde x\sim Q_\Theta, \quad
 \tilde z=\operatorname{Respond}(\Theta,\tilde x),\quad
 \Theta''=L_{\text{dream}}(\Theta,\tilde z).
\]

These equations are **interfaces for hypotheses**, not executable learning laws or results.

## 4. What counts as evidence?

### 4.1 Interaction rather than mere co-activation

For a candidate output metric with a meaningful additive embedding, a two-satellite interaction can be tested by the cross-difference

\[
 I(x)=Y_{AB}(x)-Y_A(x)-Y_B(x)+Y_0(x).
\]

A nonzero value refutes simple additivity in that embedding; it does not establish reasoning or intelligence. For categorical outputs define a separate interaction test with its assumptions, not an arbitrary numeric encoding that guarantees nonadditivity.

### 4.2 Abstraction rather than memorization

A candidate core change counts as **evidence toward abstraction** only if it (i) captures common response relations across whole satellite behaviors, (ii) preserves designated earlier responses within declared error bounds, and (iii) improves held-out compositional generalization, new-domain accommodation, or other prospectively specified transfer tasks versus controls. No success should be credited solely for compression or recurrent activation.

### 4.3 Locality and efficiency

For active cells \(A\), visited couplings \(E_A\), and \(k\) update passes, a bounded-cost local engine might have work \(O(k(|A|+|E_A|))\) *provided* scheduling, mode selection, and convergence checks fit this budget. This is not a bound independent of whole-model size unless an additional theorem bounds activation, visited edges and passes. Irregular memory traffic must be measured.

Given layer-wise path attenuation \(q\) and a branch-count bound \(b\), a bound of the form \(\sum_d C(bq)^d\) is finite for \(bq<1\) only under assumptions about norms, path multiplication and lack of uncontrolled feedback amplification. We must not export this heuristic to nonlinear systems without proof.

### 4.4 Memory and storage

Measure *persistent* topology, states, coupling metadata and learned structural changes separately from transient active sets, debug traces and run reports. The reference implementation is a monotone reachability control, with neither learned abstraction nor true algebraic interference. Its array-size estimates are not a forecast of intelligence capacity.

## 5. Program design: proof, computation, counterexamples

An admissible research result has a statement, assumptions, proof or explicitly bounded exhaustive verification, implementation commit, tested configurations and counterexamples. We distinguish:

1. **Proved**: mathematical argument applies to specified class.
2. **Exhaustively checked**: all cases within a stated finite bound.
3. **Sample-supported**: randomized or selective computational evidence only.
4. **Conjectural**: credible but unproved and unverified claim.
5. **Refuted**: counterexample found, retained in the repo.

We prioritize minimal exact models and analytical understanding over scale campaigns. Any trial involving topology must match cell/edge budgets and local rules with a no-hole control. Any claim for dreaming must include no-dream, wake-only, random-weight-change and input-history controls (history may be used as a comparator, **not** as part of the target engine). Any claim for abstraction must include unseen domains/combinations and guard against leakage.

## 6. Candidate proof program

**P1 — Well-defined response.** Define the local algebra and update schedule. Determine whether a global response exists, is unique, or can oscillate; define status for truncation/nonconvergence.

**P2 — Interference certificate.** Find a smallest finite witness of a joint response not reducible to a declared independent baseline; prove it by exhaustive cases or algebra.

**P3 — Safe approximation.** Derive conditions under which skipped weak modes/couplings yield a bounded observable error, explicitly accounting for path multiplicity and feedback.

**P4 — Structural preservation.** Bound changes in old response families after a learning update. Find minimal counterexamples to unrestricted Hebbian strengthening.

**P5 — Endogenous dream mechanism.** Define internal perturbation generation solely as a function of current \(\Theta\). Establish sampling/recurrence properties; separate replay resemblance from generalization.

**P6 — Abstraction/transfer.** Define behavioral comparisons of whole-module interactions before and after core consolidation, and test whether transfer actually improves under controlled conditions.

**P7 — Complexity/representation.** Bound work and persistent bits in restricted families; compare against flat graphs, LUT automata, and sparse numeric reference implementations.

## 7. Experimental standards and stopping rules

- Start with tiny finite systems; enumerate feasible bounded spaces exactly and archive witnesses/counterexamples.
- Freeze seeds, code revision, parameter schema, runtime details and machine-readable results.
- Keep original evidence distinct from a changing internal interpretation.
- Include explicit `complete`, `bounded`, `truncated`, and `heuristic` statuses in experiment outputs.
- Avoid changing reported thresholds or regenerating historical reports silently. Append new versioned reports.
- Record *negative* results and revise models rather than retuning until an interesting pattern appears.
- Pre-register an experiment's transfer tests before examining its learning results.
- Do not call an emergent pattern 'abstraction' without a novel-case transfer test.

## 8. What's currently implemented

`src/model.py` contains only finite undirected weighted reachability. `docs/foundations.md` proves convergence and threshold properties for **that restricted model**. These results do not extend automatically to cancellation, nonlinear interference, higher topology, sleep consolidation, or reasoning.

See [`roadmap.md`](roadmap.md) for milestones and acceptance criteria and [`open-problems.md`](open-problems.md) for the continuously maintained problem register.
