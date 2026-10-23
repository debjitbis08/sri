# Graded core interface v1 — feasibility contract

**Status at declaration:** specified before implementation. This is a small,
separate follow-up to [phase C](core-mediation-v1-phase-c-results.md), not a sweep.
Its job is to prove that meaningful unit-level transitions *can* coexist with
recovery before any grid is run. Earlier reports and contracts stay unchanged.

## What changes from phase C

Phase C coupled modules only through decoded labels, so a single bit flip from a
prototype gained nothing from coupling. Here coupling acts on internal units
**before** a decoded label changes. The label remains only an observation.

Each module has 4 binary units and two complementary prototypes, 0000 (mode 0)
and 1111 (mode 1); the core size is n = 2. Let x_M be the number of ones in module
M, with weights w_1(x) = x/4 and w_0(x) = 1 − x/4. Given a phase B mode energy
E_mode (unary, direct, and core terms), define

`E = Σ_M λ_M·min(x_M, 4−x_M) + σ·Σ_{a,b,k} E_mode(a,b,k)·w_a(x_A)·w_b(x_B)·w_k(x_K)`

The second term is the multilinear extension of the mode energy, and it agrees
with σ·E_mode at prototypes. Any pairwise mode table extends to pairwise unit
interactions; a ternary table extends to three-body unit interactions. So the
pairwise-versus-ternary distinction survives. σ is a declared scale on the phase B
witness, which is allowed because the phase B criteria are scale-invariant apart
from the margin. Energies are multiplied by 64 and computed in exact integers.
Readout is the majority label, and ⊥ at a tie, for observation only.

## Proposition, proved before running

Hold every other module at a prototype, and move module M from mode m toward m'
one unit at a time. Let Δ = E_mode(after) − E_mode(before). The multilinear term is
affine in x_M, so after j flips the energy change is λ_M·min(j, 4−j) + σ·Δ·j/4.

- A strictly descending single-flip path between the two modes exists iff
  σ·Δ < −4·λ_M. The first half requires λ_M + σΔ/4 < 0; the second half then
  descends automatically.
- Every mode-stable target (Δ ≥ 1 for all single-module moves) is a strict local
  minimum at unit level for every λ_M ≥ 0, since each first flip costs λ_M + σΔ/4 > 0.

So the internal barrier `4·λ_M` is a threshold: mode moves whose scaled gain
exceeds it become available at unit level, and smaller ones are blocked. Phase C
is the limiting case in which coupling slopes inside a module are zero.

**Worked feasibility example.** The phase B ternary witness with targets
{(0,0,0), (0,1,1), (1,0,1)} has u_A(1) = 1, ψ(0,0,1) = ψ(0,1,0) = ψ(1,1,0) = 1,
ψ(1,0,1) = −1, and other entries 0. Its mode energies are 0 at the targets,
E(1,0,0) = 1, and E(1,1,0) = 2. Start at (0,0,0) and pulse A toward 1.

- At (1,0,0), the core move to (1,0,1) has Δ = −1, the B move has Δ = +1, and A's
  return has Δ = −1.
- With the pulse held, K's slope at x_A = j is σ(1 − j/2)/4, so K descends only after
  A passes the midpoint. B never descends.
- Prediction: with σ = 8 and λ = 1 for all modules, the core switches (8 > 4), and
  after release the system rests at the target (1,0,1).
- Without the core, or with K clamped at 0, the released A returns to 0, because
  σ·1 > 4·λ_A.
- At λ = 2 the core's first flip has gain exactly 0, so it is blocked and the
  mediated response fails. The same holds at λ = 3.

These predictions are stated before running and verified exhaustively below.

## Criteria, required together in one parameter setting

- **P1:** every target, at its prototypes, is a unit-level strict minimum.
- **P2:** every single-unit perturbation of every target recovers its full state
  under every schedule.
- **P3, input-driven core participation:** a pulse on A only (h·Hamming toward
  mode e) is held until every schedule terminates, then released. Under every
  schedule the final readout is one single target y, with y_A = e and y_K ≠ x_K.
- **P4, core removed:** with K and its terms deleted, and the satellite unary and
  direct terms kept, the same pulse ends in a satellite pair other than (y_A, y_B).
- **P5, core clamped:** with K's units frozen at x_K, the same pulse ends in a
  satellite pair other than (y_A, y_B). This rules out a fixed bias.

Also report 2-flip recovery, full-switch recovery, and whether any other module's
readout changes, for phase D. No criterion is scored from these.

## Scope and stopping rule

This contract checks only the worked example: σ = 8, λ ∈ {1,2,3} applied to all
modules, and h = 8. Every reachable state is enumerated exhaustively (2^12 unit
states). If the λ = 1 predictions fail, the proposition or the implementation is
wrong, and that must be resolved first. A broader grid over witnesses, σ, per-module
λ, and n > 2 requires a separate contract after review.
