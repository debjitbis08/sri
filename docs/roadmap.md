# Research Roadmap

**Version:** 0.1 · **Status:** Proposed, not a promise of eventual intelligence.  
**Operating rule:** Advance by proof-quality and falsifiable results, not by model size or benchmark excitement.

## Milestone 0 — Reproducible finite control (existing starter)

**Aim:** a permanent reproducible baseline, explicitly weaker than the proposed architecture.

- [x] Define finite labeled weighted nodes/links, thresholded reachability and seeded internal excitation.
- [x] Supply elementary finite-convergence/threshold proofs for the control in `foundations.md`.
- [x] Maintain executable unittest checks and a baseline JSON report.
- [ ] Pin interpreter/runtime version in report metadata and verify byte-identical regeneration.
- [ ] Add `complete` vs `truncated` experiment status and tests for capped execution.
- [ ] Establish deterministic report naming and immutable archival policy.

**Exit gate:** re-run documented commands; compare a frozen receipt and independently verify the exact scope of all claims.

## Milestone 1 — From algebraic combination to recurrent contextual interaction

**Question:** Can reciprocal influence produce stable collective responses not reducible to isolated components or simple propagation?

| Level | What is demonstrated | Current status |
| --- | --- | --- |
| OP-02 | Algebraic combination differs from an independent set-union control | Complete within its restricted class; original certificate frozen |
| OP-02B | Recurrent contextual interaction and bounded attractor dynamics | Complete within scoped recurrence-necessity class; stronger full-star stable-control target remains open |
| OP-03 | Schedule, attractor, and structural robustness | Partial: exact Boolean fair-schedule/basin/edit analysis; main witness fails joint robustness |
| Later milestones | Stable structural modes beyond propagation, graded resonance, learning, and abstraction | Open |

- [x] Retain OP-02's two candidate algebras, declared observable, independent control, and scoped minimality certificate.
- [x] Specify a stateful recurrent core–satellite model with persistent evidence and exact schedules.
- [x] Exhaustively classify fixed points, cycles, and basins in the first bounded OR/XOR family.
- [x] Compare matched isolated satellites, uncoupled nodes, feed-forward coupling, and edge ablations.
- [x] Retain a reachability-reducible contextual witness and cycle/schedule counterexamples.
- [x] Specify and test recurrence necessity under matched acyclic edge deletions.
- [x] Prove the minimum edge count within the declared star-ablation family.
- [x] Analyze all node-fair atomic schedules, exact basins, single-bit recovery, and single-edge deletion in the existing Boolean model.
- [ ] Specify a jointly robust, distinguishable response family before expanding the algebra.
- [ ] Find a fully reciprocal response distinction while all acyclic satellite observations are stable, with declared schedule/initialization requirements.
- [ ] Define a separate prospective task before claiming useful inference.

**Completed recurrence gate:** a stable contextual witness requiring recurrence under declared controls, scoped edge minimality, and exact convergence/cycle semantics. **Next robustness gate:** a declared response family satisfying schedule, initialization, and edit-preservation requirements while remaining distinguishable. Interaction alone is not capability. See [OP-03](op-03-robustness.md).

## Milestone 2 — Layering, graded coupling and topology

**Question:** What behavior is due to layers, algebra, or holes?

- [ ] Define layer-resolved mode states and graded inter/intra-module couplings.
- [ ] Compare flat, layered, cyclic, annular and higher-genus finite structures under matched size and rules.
- [ ] Define exterior perturbation families and inner/whole-module response signatures.
- [ ] Test rich exterior responses against stability/distinguishability of inner responses.
- [ ] Find whether nontrivial cycles produce response modes absent in matched controls.
- [ ] Separate exact global propagation from the effects of numerical truncation.

**Exit gate:** controlled comparisons and one mathematical proposition/counterexample explaining when topology changes responses. Do not infer a learning benefit simply from extra modes.

## Milestone 3 — Adaptive inference budgets

**Question:** Can the same structure produce useful progressively refined responses at lower cost?

- [ ] Introduce per-module maximum reach, sensitivity and work-pass budgets.
- [ ] Preserve response status (`exact`, `bounded-error`, `truncated`, `uncertified`).
- [ ] Derive sufficient conditions for safe pruning or give bounded counterexamples.
- [ ] Test weak-but-decisive mode interactions and aggregate effects of many weak edges.
- [ ] Benchmark CPU time, memory traffic, active-set size and output error against full inference.

**Exit gate:** certified approximate behavior for at least one nontrivial restricted class, plus quantified limits on broader claims.

## Milestone 4 — Intrinsic structural STM and satellite learning

**Question:** Can repeated perturbations modify structures while preserving established responses?

- [ ] Define local changes to links, incidence, mode states and plasticity resistance as competing rules.
- [ ] Implement STM as a highly plastic module, **not** an external replay store.
- [ ] Vary plasticity and persistence independently; measure both.
- [ ] Measure forgetting/interference on previously defined perturbation families.
- [ ] Include controls for fixed topology and unconditioned random rewiring.

**Exit gate:** reproducible structural learning and an explicit preservation metric; record cases of catastrophic interference.

## Milestone 5 — Endogenous dreaming

**Question:** Can the current learned system generate useful internal perturbations without retained raw experience?

- [ ] Specify a structural sampler \(Q_\Theta\), requiring no historic input/replay database.
- [ ] Generate slight variations, cross-satellite combinations and counter-patterns from the current structure.
- [ ] Separate raw recurrence, diversity and cross-context robustness.
- [ ] Establish pre-registered held-out tests and matched wake-only/no-dream baselines.
- [ ] Measure self-reinforcing spurious patterns and drift.

**Exit gate:** demonstrate either a reproducible benefit *or* a falsifying negative result. Do not credit replay resemblance as abstraction.

## Milestone 6 — Whole-module abstraction and transfer

**Question:** Can sleep-generated shared responses change the core to improve unforeseen combinations?

- [ ] Define invariant/equivalence families for *whole-module* behavior.
- [ ] Propose consolidation rules that do not copy satellite-specific knowledge wholesale.
- [ ] Test before/after compatibility and previously established response preservation.
- [ ] Evaluate unseen combinations and structurally related new domains.
- [ ] Measure future accommodation effort, computation and storage before/after consolidation.
- [ ] Compare against explicit lookup, memorization and randomized-rewiring controls.

**Exit gate:** prospective transfer improvement, not just compression or stronger activation.

## Milestone 7 — CPU viability and scalability

**Question:** Does any validated candidate genuinely outperform simple baselines at a useful resource point?

- [ ] Implement hot paths using contiguous integer arrays, LUTs, bitsets and sparse event queues as warranted by profiling.
- [ ] Measure memory allocation, cache misses, edge visits, active-set expansion, and wall-time.
- [ ] Report workloads in terms of model size *and* true active structure.
- [ ] Compare against sparse graph, finite-state, binary-vector and compact numeric baselines.
- [ ] Explore bounded memory operation and safe structural consolidation.
- [ ] Define an end-to-end task demonstrating real utility before product claims.

**Exit gate:** a repeatable CPU benchmark with accuracy/resource trade-offs; include failures and hardware specifications.

## Milestone dependencies

`M0 → M1 → M2 → M3` establishes response and approximation. `M1 → M4 → M5 → M6` establishes structural learning and candidate abstraction. `M3 + M6 → M7` evaluates the long-term application claim. Some analytical work may proceed in parallel, but later milestones cannot borrow unproved assumptions from earlier ones.

## Each milestone's artifact contract

1. `docs/`: precise definitions and proof statements with assumptions.
2. `src/`: reference implementation with simple, inspectable semantics.
3. `tests/`: invariants, counterexamples and known edge cases.
4. `experiments/`: deterministic runner, configuration and seeds.
5. `reports/`: frozen machine-readable result plus short interpretation.
6. `docs/open-problems.md`: updated status, counterexamples and next obligations.

### Immediate next iteration

Continue [OP-03](op-03-robustness.md) on the same tiny Boolean model. Define the required combination of schedule robustness, attractor recovery, structural preservation, and distinguishability before expanding local rules or introducing graded coupling. Retain OP-02 and OP-02B as successful scoped baselines; preserve the stronger full-star all-controls-stable target as unresolved. Milestone 0 reporting hardening remains outstanding. Structural learning and dreaming remain deferred.
