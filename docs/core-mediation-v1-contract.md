# Core mediation v1 — bounded contract

**Status at declaration:** specified before any implementation or run. This is an
in-session contract, not an external preregistration. It is a mechanism and
realizability study, not a selected SRI architecture. Preserve all earlier
controls, certificates, and reports unchanged.

**Question.** Can a structured core K mediate multiple stable, distinguishable
joint configurations of two structured satellites A and B, including
configurations that are not realized without K's participation, while permitting
recovery from small perturbations? Under what conditions can the response be
computed from a bounded neighborhood, with omitted interactions certified harmless
or explicitly reported as uncertain?

## Departures from earlier studies

1. **Discrete, exact computation.** No gradient flow or numerical integration.
   Modes are finite labels, interaction tables hold integers, updates are
   asynchronous single-module or single-unit moves, and every stability and
   recovery claim is checked by exhaustive enumeration with integer arithmetic.
2. **Compatibility, not agreement.** Couplings are general compatibility tables
   φ(a,k), not penalties on difference. Agreement-form pairwise coupling is
   retained only as a declared control.
3. **Criteria before energies.** The targets below are conditions on joint
   configurations. Interaction classes are then tested for whether any table
   meets them, rather than designing one energy and reporting what it does.
4. **Dependencies.** numpy and scipy (HiGHS LP/MILP) are permitted. Solver output
   is never trusted directly: witnesses are verified exactly in integers, and
   impossibility claims require an exactly verified rational certificate.

## Phase B — mode-level realizability

**Configurations.** M_A = M_B = {0,1}; M_K = {0,…,n−1} with n ∈ {2,3,4}. A joint
configuration is x = (a,b,k). The observable readout is the satellite pair (a,b);
two configurations are *distinguishable* iff their satellite pairs differ.

**Interaction classes.** Every class has unary tables u_A, u_B, u_K. Couplings:

| Class | Coupling terms | Integer entries stored |
| --- | --- | --- |
| U (control) | none | 4+n |
| D (direct, no core) | φ_AB | 8+n |
| S (pairwise star) | φ_AK, φ_BK | 4+5n |
| F (full pairwise) | φ_AB, φ_AK, φ_BK | 8+5n |
| T (ternary) | ψ_ABK | 4+5n |
| G (agreement control, n=2) | w_A[a≠k] + w_B[b≠k], identity mode map | 8 |

S and T store the same number of entries at equal n, so comparing them at equal n
is already a storage-matched comparison.

Write E = U + C, where U holds the unary and direct (A–B) terms and C holds the
core-coupling terms. Removing K means deleting u_K and C and keeping U unchanged:
this is the matched system without the core.

**Criteria** (margin m = 1; a move changes the mode of exactly one module):

- **Stable:** every single-module move from x raises E by at least m.
- **Unstable:** some single-module move lowers E by at least m. A configuration
  that is neither is *marginal* and counts as neither in any criterion.
- **R1, core-dependent:** at least one target satellite pair is unstable in the
  matched system without K.
- **R2, participation, not bias:** for every clamped core mode k, with K fixed at k
  and only satellite moves allowed, at least one target satellite pair is unstable.
  A fixed context that realizes every target, even marginally, is bias rather than
  mediation.
- **R3, no collapse under strong coupling:** targets remain stable for E_s = U + s·C
  for every s ≥ 1. Since ΔE_s is affine in s, this holds exactly iff each move
  satisfies ΔU+ΔC ≥ m and ΔC ≥ 0. No sampling of s is needed.
- **R4, exclusivity (strict variant only):** every non-target configuration is
  unstable, so none is stable or marginal.

U holds u_A, u_B, u_K, and φ_AB; C holds φ_AK, φ_BK, ψ_ABK, and G's agreement
terms. The no-core system keeps u_A, u_B, and φ_AB. The base variant imposes
R1–R3; the strict variant also imposes R4.

**Target levels.**

- **Level 1:** two configurations with distinct satellite pairs. Enumerate all
  satellite-pair sets and core assignments, up to relabeling each module's
  modes and swapping A and B.
- **Level 2, joint dependence:** three or four configurations with distinct
  satellite pairs, in which k is neither a function of a alone nor of b alone.
  This is the declared formalization of "compatibility that exists only through
  collective participation." Agreement through the core cannot meet it under an
  identity mode map. That is a prediction to verify, not an assumption.

**Analytic note: why Level 1 alone is insufficient.** The agreement control G
already meets Level 1. Take w_A = w_B = 2, u_A(1) = u_B(1) = 1, and all other unary
entries 0. The targets are (0,0,0) and (1,1,1). Both are stable with margin 1.
Without K, (1,1) is unstable (R1). Clamping K at either mode destabilizes the
opposite target (R2). ΔC ≥ 0 holds for every move (R3). So Level 1 cannot tell
consensus apart from compatibility. Credit for mediation beyond consensus is
assigned only at Level 2.

**Decision procedure.** For each (class, n, target set, variant), solve for
integer tables with entries in [−16,16] using MILP. Disjunctions (instability,
R2, R4) use binary indicators. Outcomes:

- *Realizable:* an integer witness, re-verified exactly against all criteria.
- *Impossible:* every disjunctive branch is infeasible as a real LP with an exactly
  verified rational Farkas certificate. Since criteria are invariant under scaling
  apart from the margin, this excludes real-valued tables of any magnitude. scipy's
  HiGHS returns no infeasibility ray, so the certificate y ≥ 0 with Aᵀy = 0 and
  bᵀy < 0 is found by a separate LP, rationalized, and checked in `Fraction`. Branches
  are enumerated explicitly only when there are at most 4096; otherwise only the
  bounded result is reported.
- *Bounded-infeasible:* MILP infeasible within [−16,16] without such a certificate.
- *Unresolved:* solver failure or limits. Never counted as either outcome.

**Primary measurements.** The smallest n at which each class realizes each level,
the storage of the witness at that n, and whether S at some n matches T at
smaller n. The trivial hidden-variable rewriting (any ternary relation becomes
pairwise through an auxiliary variable whose modes list the allowed pairs)
predicts S succeeds once n reaches the number of target pairs. Any result is
interpreted against that baseline, not as evidence for or against higher-order
coupling in general.

## Phase C — structured modules and recovery

Each module carries binary units: 4 for A and B, and 2n+2 for K (bounded so that
the joint space has at most 2^18 states). Its internal energy is λ·(Hamming
distance to the nearest declared prototype), with one prototype per mode and
prototypes at pairwise distance at least 3. The readout r maps a state to the
unique nearest prototype's mode, or to ⊥ on a tie. ⊥ takes a coupling row
equal to the table maximum plus m. Modules interact **only through readouts**:
E = Σ internal_i + (phase B witness evaluated at the readouts).

Updates are asynchronous single-unit flips accepted only when they lower E
strictly. E decreases on a finite space, so every schedule terminates. *Recovery*
from a perturbation means every terminal state reachable under every schedule
reproduces the original readout. Full-state recovery is reported separately.
Perturbations flip 1 or 2 units in a single module, from every isolated target
configuration, enumerated exhaustively. λ ∈ {1,2,4,8} relative to the witness
scale is the only parameter grid; λ is never tuned to rescue a failure.

**Controls:** (i) a phase B witness with no internal structure (mode moves only);
(ii) the core with prototypes at distance 1, i.e. internal structure carrying no
error-correcting distance; (iii) U and D witnesses with the same satellite
internals. If structure contributes no recovery beyond (i), report that plainly.

### Phase C details, declared before implementation

- **Witnesses.** Every phase B witness realizable in the strict variant: classes
  S, F, T, and G at all core sizes. Strict witnesses are used so that every
  mode-level terminal configuration is a target.
- **Prototypes.** A and B: 0000 and 1111. K uses blocks: for n=2, 000000 and 111111;
  for n=3, 00000000, 11110000, and 00001111; for n=4, 0000000000, 1111100000,
  0000011111, and 1111111111. The distance-1 control (ii) uses prototype j =
  first j units set, so neighbouring modes differ in one unit.
- **⊥ entries.** Each table term whose argument includes ⊥ takes the maximum value
  that term attains over modes, plus m. For G, the term w·[a≠k] attains values in
  {0,w}.
- **Perturbations.** All 1- and 2-unit flips within one module, plus a full
  switch of one module to each of its other prototypes. A full switch is the
  unit-level counterpart of control (i)'s mode move, so (i) is compared with
  structure on exactly that perturbation.
- **Amendment of control (iii).** Phase B has no U or D witnesses, so this control
  is vacuous and is recorded as such. It is not replaced by an invented witness.
- **Search.** Reachable terminal sets are computed exhaustively by memoized search
  over the strict-descent graph. A search touching more than 2·10^6 states is
  unresolved.

### Input and output modules

- **Input.** An input pulse toward mode e of A adds h·Hamming(s_A, prototype_e)
  to E. It acts on A's units only, not on B or K. Dynamics run to every terminal
  state under the pulse, then the pulse is released and dynamics run to every
  terminal state again. Start from each target, e ∈ {0,1}, h ∈ {1,2,4,8,16,32}.
  *Input-selected mediation* means: from one start, the two pulses each lead,
  under every schedule, to a single target after release; the two targets differ;
  A's final mode equals e; and K's mode changes for at least one pulse. Only
  configurations that persist after release count. The pulse is never credited
  with installing a configuration it does not itself encode.
- **Output.** The read-only output is the readout (a,b). The feedback variant adds
  an output module O with 4 units and energy μ·Hamming(s_O, (r_A,r_A,r_B,r_B)),
  where ⊥ mismatches both bits. Because its energy depends on readouts, O pushes
  back on A and B. Run μ ∈ {1,4}, with O starting consistent. Feedback earns credit
  only where it changes recovery or input outcomes relative to read-only.
- **Locality data.** For each perturbation, record whether any other module's readout
  ever changes along any descent path. Phase D uses these records.

## Phase D — locality and certified truncation

- **Interface certificate.** A perturbation of a satellite that never changes its
  readout cannot change any other module's local energy differences, so the other
  modules need no evaluation. Measure, over all phase C perturbations, the units
  actually evaluated against the total, and how often this certificate applies.
- **Omission certificate.** When a module's move is decided with some interaction
  tables omitted, the decision is *certified* if its local margin exceeds the
  maximum absolute change those tables can contribute to that move. Otherwise it
  is *uncertain*. Compare every certified decision against the full computation:
  **one certified error refutes the certificate.** Report the uncertain fraction.
- **Near transitions.** Sweep one unary entry of the mediated satellite across its
  switching boundary in integer steps, from −16 to 16, and report where decisions
  become uncertain. The test passes if uncertainty concentrates at the boundary and
  there are no certified errors. Silent mis-answers are the failure mode tested.
- **Sparse storage.** Report stored entries per class. Linear growth in modules
  and active relations cannot be tested with three modules. A family with N
  satellites is declared future work, with zero credit here.

## Refutation conditions, stated in advance

- Phase B: if S never realizes Level 1 for n ≤ 4, the core gives no pairwise
  mediation in this bounded class. If G meets Level 2, the joint-dependence
  formalization fails to exclude consensus and must be revised before Phase C.
- Phase C: if no λ yields recovery from all single-unit perturbations of the
  mediated targets, structured recovery fails in this construction.
- Phase D: one certified decision that disagrees with the full computation.

## Scope discount

| Component | Treatment and permitted credit |
| --- | --- |
| Finite modes, compatibility tables, core mediation criteria | Implemented; scoped mechanism and realizability evidence only. |
| Internal module structure | Hand-declared prototypes; no learning, no acquired modes. |
| Layers, graded inter-layer coupling, topology | Not implemented; zero evidence either way. |
| Many satellites, scaling, CPU efficiency | Not tested beyond stored-entry and evaluation counts. |
| Input/output modules, STM, dreaming, abstraction | Not implemented. Readout is an observation proxy. |

Record solver status, certificate checks, counts of enumerated states and
schedules, evaluations, runtime, and memory. Runtime goes in a separate report from
the deterministic certificate. Run all historical regressions unchanged.
Stop at the declared bounds; no parameter search beyond them.
