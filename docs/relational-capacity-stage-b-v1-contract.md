# Relational capacity Stage B v1: rank-two, parallel and coupled capacity search

**Status:** prospective execution contract, 2026-10-25, frozen by the commit
that adds it, before any implementation. Sources, tests and the manifest are
committed before any recorded test phase, check or fit. Nothing under this
contract had been implemented or run when it was frozen.

This contract integrates the [Stage B decisions](relational-capacity-stage-b-decisions.md)
into the [proposal](relational-capacity-and-composition-study-proposal.md). Its
basis is the [analytical preflight](relational-capacity-and-composition-analysis.md)
and its [results](relational-capacity-and-composition-analysis-results.md), the
[weak-coupling bound](relational-composition-weak-coupling-bound.md) and the
[handoff audit](relational-capacity-handoff-audits.md). It follows the
[computation protocol](computation-reporting.md). Where wording differs, the
decisions document governs and this contract states how it is executed.

All earlier sources, contracts, scientific results and runtimes are preserved.
The 72-scene task and both renderings are known development fixtures from
[masked capacity v1](single-module-masked-capacity-v1-contract.md); their
outcomes for S1 are known. This is privileged all-scene capacity fitting, not
transfer or learning. No fitted parameters may initialize a later learning study.

## 1. Questions

1. Can the rank-two retained module (S2) represent the task under the unchanged
   witness criterion?
2. Can two independent rank-one modules with a fixed mean readout (P0) do so?
3. Can a designated output module coupled to an input module through terminal
   identity ports (D1), at μ ∈ {0.001, 0.01, 0.1}, do so?

Each (configuration, rendering) group is classified as `constructive_witness`,
`unresolved_after_budget` or `excluded_by_bound`. A D1/P0 difference is not
credited to communication, because their interfaces and evidence placement
differ (decisions §1). A failure within budget is not an impossibility result.
S1 and fixed-route compositions are already excluded. R and G are deferred:
no relation-selection mechanism is specified, and no Stage B score may be used
to choose their scaffold.

## 2. Task, renderings and witness criterion (unchanged)

Use `src/masked_capacity.py:task` and `rendering(world)` for world ∈ {0,1}:

- 72 scenes ordered by (r, q_index, t);
- visible parts 0,1 and hidden parts 2,3;
- loss per scene `||prediction_hidden - target_hidden||^2 / 2`;
- the explicit observation mask: anchors only on outer sectors 0,1.

A **witness** is one parameter setting with loss ≤0.10 on all 72 scenes and
both hidden orientations correct on every scene. Orientation is decoded with
`src/abstraction_preflight.py:decode`. The witness must pass independent
verification of its serialized parameters (§8).

The shortcut and comparator audit is the frozen masked-capacity v1 audit
(`reports/single-module-masked-capacity-v1-audit.json`). It is cited by
SHA-256 in the manifest and is not rerun: task, renderings and comparators are
unchanged. The task-audit concerns about the formula shortcut, orbit closure
and second-order violations ([handoff audit §2](relational-capacity-handoff-audits.md))
apply to later transfer studies. They do not affect this representability
question.

## 3. Configurations and energies

Module state `x ∈ R^80`:

- rings of sizes 32, 24, 16, 8 at offsets 0, 32, 56, 72;
- four sectors per ring, using `src/single_module_plasticity.py:sector_indices(0)`;
- radial maps D_l,j with orthonormal columns, of shapes (8,6), (6,4), (4,2);
- unit terminal vectors v_j with `Q_j = I - v_j v_j^T`.

For routes a, the local energy is

```
E_loc(x; a) = 1/2 sum_(l<3, j) a_l,j ||x_l,j - D_l,j x_l+1,j||^2
            + [rank one] 1/2 sum_j a_3,j ||Q_j x_3,j||^2
            + lambda/2 sum_l x_l^T L_l x_l,      lambda = 0.001,
```

where L_l is the unweighted cycle Laplacian of ring l in its existing order.
`M` masks the 16 visible outer coordinates.

| Name | Energy | Output (16 hidden coordinates) | Stored floats |
| --- | --- | --- | --- |
| S2 | `0.1/2 ||M(x_0 - h)||^2 + E_loc(x)` without terminal blocks | `x_0` at parts 2,3 | 320 |
| P0 | Two independent energies `0.05/2 ||M(x^k_0 - h)||^2 + E_loc^k(x^k)`, k=0,1 | `(x^0_0 + x^1_0)/2` at parts 2,3 | 656 |
| D1(μ) | `0.1/2 ||M(x^in_0 - h)||^2 + E_loc^in + E_loc^out + mu/2 ||x^in_3 - x^out_3||^2` | `x^out_0` at parts 2,3 | 656 |

Conventions:

- **D1 modules.** Module 0 is the input module and module 1 the output module.
- **Ports.** The D1 port uses all eight terminal coordinates in sector-major,
  coordinate-major order. Sector numbers are identical on both sides, with no
  permutation and no learned or renderer-based transport. T=I is a coordinate
  convention, not an exact gauge symmetry, because the terminal lateral cycle
  is not rotated with it.
- **Fixed constants.** μ is held fixed across both inference passes. No
  coupling, readout, gain or anchor coefficient is learned.
- **Evidence budget.** Zero-state observation energy is 0.1 for every
  configuration. P0 uses each scene twice locally but counts as one
  independent example and one scene exposure.

**Solvability.** The masked Hessian is SPD for every allowed parameter
setting and route:

- S2: analytical preflight §2;
- P0: the existing uniqueness proof applies to each module;
- D1: the analytical preflight §3 sufficient construction, with positive cycles,
  full-rank radial maps and an injective terminal correspondence.

The checks in §8 still apply to every solve.

## 4. Inference: one collective pilot, local gates, one routed solve

1. **Pilot.** Assemble the collective Hessian with all local routes equal to 1,
   including μ for D1. Solve it once per batch for all 72 right-hand sides. S2
   uses one 80-state system and D1 one 160-state system. P0 uses each module's
   own anchored 80-state system, solved block-diagonally.
2. **Gates.** Each module computes its own gates from its own 80-state slice of
   that pilot, using its own parameters:
   ```
   radial:   r_l,j = ||z_l,j - D_l,j z_l+1,j||^2 / (||z_l,j||^2 + ||D_l,j z_l+1,j||^2 + 1e-12)
   terminal: r_3,j = ||Q_j z_3,j||^2 / (||z_3,j||^2 + 1e-12)        (rank one only)
   g = 0.1 + 0.9 exp(-r/0.25),   a = 0.25 + 0.75 g.
   ```
   S2 has 12 adaptive radial gates. Rank-one modules have 16. Output gates are
   neither copied from the input module nor computed from an isolated pilot.
3. **Routed solve.** Assemble the collective Hessian with each module's routes
   and the unchanged μ. Solve once. There is no rerouting, coupling gate, extra
   pass or router.
4. **Small denominators.** If the denominator of any gate ratio is below 1e-8,
   record it as a diagnostic (ε-dominated gate), not an error. The output
   module is driven only through its ports.

Frozen helpers hard-code anchor 0.1 and a single module. New code must assemble
these Hessians itself, from `v1.block_curvatures` and its own anchor and
coupling terms. It must not edit frozen sources.

## 5. Initialization and seeds

- **Root seed** 2026102506. Each stream uses
  `np.random.default_rng(SeedSequence([root, world, start, module]))` with
  world, start and module each in {0,1}.
- **Renderers** remain exactly those of masked-capacity v1.
- **Draw order** per stream: Gaussian radial blocks in ring-major, then
  sector-major order (shapes (8,6), (6,4), (4,2), four blocks each). Apply
  reduced QR with a positive diagonal, as in
  `src/abstraction_preflight.py:random_params`. Then draw four Gaussian
  2-vectors and normalize them.
- **Use.** S2 uses module 0's maps and discards its terminal vectors. P0 and D1
  use modules 0 and 1, with module 0 as D1's input module.
- **Pairing.** Initial parameters are paired across configurations, μ values
  and step sizes. No seed depends on μ, step size or configuration name.
- Save exact initial arrays and their hashes. No hand-designed or signal-plane
  initialization and no reseeding.

## 6. Optimization and selection

**Objective.**

- Train on the exact maximum of the 72 per-scene losses, differentiated
  through the pilot, gates and routed solve.
- At exact ties, use the uniform average of the active-scene gradients
  (PyTorch `amax` semantics), checked independently.
- No softmax temperature, warm-up or schedule.
- Record mean loss as a secondary metric.
- The full derivative is a capacity optimizer only, not a plasticity rule.

**Optimizer** (as in masked-capacity v1):

- float64 CPU PyTorch, one process, one intra-op and one inter-op thread,
  deterministic algorithms;
- project gradients to the Stiefel and unit-sphere tangents, then clip the
  global norm at 10;
- Adam with betas (0.9, 0.999), ε=1e-8, no weight decay, constant step size;
- after each update, apply the positive-diagonal QR and unit retractions;
- ambient moments are kept without transport;
- S2 omits all terminal-vector operations.

**Menu.** Step sizes {0.003, 0.01} × starts {0,1}. Each candidate gets at most
1,000 full-batch updates on all 72 scenes. Checkpoints are
{0, 100, 250, 500, 1000}.

**Selection.**

- Within a candidate, select the lexicographically smallest
  (worst-scene loss, mean loss, update index) among its checkpoints, using
  actual float64 values and no tie band.
- Across candidates, use the same first two metrics, then enumeration order.

**Enumeration.** S2, P0, D1; world ascending; for D1, μ ascending; start
ascending; step size ascending.

**Stopping.**

- At a scheduled checkpoint that meets the full witness criterion, serialize
  the parameters and verify them immediately (§8).
- If verification succeeds, stop the remaining candidates of that
  (configuration, world) group, including the remaining μ values for D1. Mark
  them `unexecuted_after_witness` and continue with the other groups.
- First-witness stopping establishes existence. It does not identify an
  optimal μ.
- Without a witness, finish the declared menu. There is no early stop on
  stagnation or low sensitivity, and no extension after failure.

**Ceiling.**

- 5 configurations × 2 worlds × 2 starts × 2 rates = 40 candidates.
- 40,000 updates and 2,880,000 scene exposures.
- 200 scheduled checkpoint evaluations.
- Verification and screen evaluations are extra.

## 7. Exclusions, screens and mechanism controls

**Pre-search exclusions.** Only a completed theorem or independently certified
bound that covers every allowed parameter setting, route and the actual
interface can exclude a configuration. The universal lower bound must be
strictly above 0.10. Applied:

- S1: the masked-capacity obstruction (not in the budget);
- fixed-route compositions: the affine floor;
- D1 weak-coupling bound: its threshold is μ < 6.90e-4, so it excludes none of
  the three values. It is recorded with its arithmetic.

The sector-aligned and rank-two bounds remain open and are reported as
unresolved. If a proof completes mid-run, keep the work already incurred,
amend the ledger in a separate document, and do not reallocate budget.

**Route-sensitivity screen.** At every scheduled checkpoint, compute the
fixed-route reference f_fixed with the same parameters, μ, evidence and output
and all local routes equal to 1. This is exactly the pilot output, so it costs
no extra solve. Over the 24 q=1 scenes, report:

```
d   = sqrt(mean ||f_routed - f_fixed||^2)
tau = sqrt(2F) - sqrt(0.2) = 0.254592,   F = (8 - 2 sqrt 2)/21
t_p = max(0, sqrt(mean ||f_fixed - target||^2) - sqrt(0.2))
```

All quantities are in absolute output units. `d < tau` (or `d < t_p`) shows
that this parameter setting cannot be a witness. That excludes the setting
only; the candidate continues. Passing the screen is not sufficient. A verified
witness with `d < t_p - 1e-9` is treated as a correctness error.

**Mechanism controls**, evaluated only on the selected checkpoint of each
executed candidate:

- the fixed-route output (`f_fixed`) and its scores;
- for D1, the disconnected-output analytical fact: with no port, the output
  module has zero right-hand side, so its SPD equilibrium is zero and every
  scene's loss is exactly 1. This is stated, not computed.

These controls describe deployed dependence. They are not a
communication-benefit experiment.

## 8. Numerical gates, verification and provenance

**Per-solve gates.** Each of these is an error that stops execution:

- finite inputs, states, gates, losses and gradients;
- QR rank `min |diag R| > 1e-10`;
- Stiefel and unit constraints within 1e-9;
- Hessian symmetry within 1e-12 relative;
- Cholesky success;
- relative solve residual ≤1e-10.

**Tests required before any check or fit:**

- Energy/Hessian agreement with an independent residual expansion, for each
  configuration, including the α/2 anchors and the μ ports.
- Masking and absence of any hidden-input dependence.
- SPD across random parameters and routes in [0.325, 1] for every configuration.
- Agreement of collective solves with a Schur complement onto output
  coordinates within 1e-9. For D1, eliminate the input module and the output
  module's interior.
- Block-diagonal P0 solving agrees with its dense 160-state equivalent.
- Gates use the collective pilot slice, not an isolated pilot.
- Full gradients match finite differences:
  - away from ties;
  - at a constructed exact tie under `amax` semantics;
  - including off-manifold terminal curvature.
- Tangent projection and retraction.
- Exact seed and initialization reproduction.
- The screen equals the pilot.
- Target-free inference.

**Verification.** Serialize parameters to 12 significant digits. Run
independent NumPy/SciPy inference, sharing no Torch code, with Schur checks.
Require output agreement with Torch within 1e-9, constraints after
serialization, and the target. Verify each serialized selected checkpoint twice
and require byte-identical verification records. The full optimizer trajectory
runs once; this is disclosed.

**Provenance.** Order of work:

1. commit this contract;
2. commit sources, tests and a manifest recording hashes of the sources,
   tests, contract, decisions, frozen dependencies, frozen v1 audit and parent
   manifest;
3. run tests;
4. run the search;
5. verify.

Runners refuse changed frozen inputs, uncommitted manifests and phases already
recorded. Use the frozen helpers by import only
(`experiments/locality_adaptation.py` encoding/hash/git/environment, and the
masked-capacity freeze pattern). An implementation-only correction needs a
disclosure, a refreeze and a rerun; no failed work is silently replaced.

New files:

- `src/relational_capacity.py`
- `experiments/relational_capacity.py`
- `tests/test_relational_capacity.py`
- reports prefix `relational-capacity-stage-b-v1`: manifest, results, runtime
- `docs/relational-capacity-stage-b-v1-results.md`

## 9. Computation and storage

**Dimensions.**

| | D | P | Notes |
| --- | --- | --- | --- |
| S2 | 80 | 320 | |
| P0 | 2×80 | 656 | Block-diagonal |
| D1 | 160 | 656 | Dense, with a 64-entry port block |

N=72, E=1 port block of width d=8 (D1).

**Per forward pass, cost orders.**

| Step | Cost |
| --- | --- |
| Curvature assembly | O(Σ_modules LJ·80²) |
| Pilot factor | one O(D³) |
| Pilot solves | N·O(D²) |
| Routed Hessian assembly | N·O(LJ·80² + d²) |
| Routed factors and solves | N·O(D³) |

Dense backward work is O(N D³), and the retained autograd state is
O(N D² + P). QR retraction costs O(Σ J s_l s_(l+1)^2) per module per update,
and Adam O(P).

Estimated Cholesky flops (D³/3):

- 1.71e5 per factorization at D=80;
- 1.37e6 per factorization at D=160.

These are estimates, not measurements.

**Maximum factorizations** before verification. Each configuration counts
1,005 forward batches per candidate:

- S2: 586,920 at D=80;
- P0: 1,173,840 at D=80;
- D1: 1,760,760 at D=160.

**Bytes** (float64):

| Item | Bytes |
| --- | --- |
| Parameters | 2,560 (S2) / 5,248 (P0, D1) |
| Adam moments | twice the parameter bytes, plus counters |
| Inputs and targets per rendering | 27,648 |
| Curvature cache | 614,400 (S2) / 1,638,400 (P0, D1) |
| D1 batch Hessian | 14,745,600 |

These are named allocations, not a bound on peak native workspace.

**Planning estimate.** Masked v1 measured 311 s for 8,000 updates at D=80.
Assuming cost is cubic-dominated, the full menu needs on the order of 2–3 h on
one thread. This is an extrapolation for planning, not a speed or scaling
claim.

**Runtime JSON** records:

- pinned versions, CPU, BLAS and thread configuration, hashes, revision,
  command and dirty paths;
- wall and process CPU time in a nested phase tree (inclusive and exclusive,
  with nested totals never added to their components), per candidate and per
  checkpoint;
- peak RSS (Linux KiB, process scope, `RUSAGE_SELF` / `RUSAGE_CHILDREN`
  labelled);
- factorization and solve counts by dimension and pass;
- exposures, independent examples and local observation uses;
- updates, retractions, backward calls and screen evaluations;
- every candidate, including rejected and unexecuted ones;
- verification counts.

Native workspace, FLOPs and backward primitive-solve counts are null, each with
a reason. Report quality against cumulative updates, exposures and measured
search cost, labelled as privileged fitting. Scientific JSON contains no times.

**Missing historical cost data.** The S1 reference has no per-scene inference
latency, native workspace, FLOP or backward-solve data. This is disclosed and
not retrofitted.

## 10. Decision

- **A witness for a (configuration, world) group** shows that the
  representation is feasible for that development fixture only.
- **What may follow.** A separately frozen learning and transfer contract with
  fresh worlds may follow. It must address the handoff audit's task issues:
  the formula comparator, orbit closure including reversal∘reflection, and
  second-order-matched violations.
- **D1 witness without a P0 witness.** This does not establish a communication
  benefit; the interfaces differ.
- **Unresolved after budget.** Report the completed budget and the
  route-sensitivity record. Do not extend the grid, add arms or move to R/G
  on that basis.
