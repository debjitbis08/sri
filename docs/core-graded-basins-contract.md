# Graded core interface — basin contract

**Status at declaration:** specified before implementation, as a follow-up to
[graded core v1](core-graded-v1-results.md). The question: in the graded unit
interface, how large and how reliable are the useful basins, where do the spurious
terminal states come from, and do the answers depend on pairwise versus ternary
coupling or on a softer core? Earlier contracts and reports stay unchanged.

## Model

As in graded v1: 4 units per module, complementary prototypes, n = 2, and

`E = Σ_M λ_M·min(x_M, 4−x_M) + σ·(multilinear extension of E_mode)`

The one change is per-module barriers: λ_A = λ_B = λ_S for satellites and λ_K for
the core. Energies are scaled by 64 and exact.

## Propositions, proved before running

**P-tie (where spurious terminals come from).** Fix the other modules. Along x_M the
internal term is a tent, linear on [0,2] and on [2,4] with its peak at 2, and the
coupling term is affine. Their sum is concave and piecewise linear. A unit flip
changes x_M by ±1, so a state with x_M ∈ {1,2,3} can be terminal only if some flip
leaves E unchanged. At x_M = 2 even that is impossible, because the concave peak
is strict. Every off-prototype terminal state is therefore a **tie**.

A flip changes E by ±λ_M plus σ·k/64 for an integer k. The multilinear slope is a
multiple of 1/16 in mode units, scaled by σ/4. Hence:

> If 64·λ_M/σ is not an integer for every module, no flip is energy-neutral.
> Every terminal state is then a strict local minimum at a prototype configuration.

The graded v1 setting σ = 8, λ = 1 gives 64/8 = 8, an integer, which is why its 96
spurious terminals exist.

**P-terminal (which prototypes are terminal).** In a tie-free setting, the prototype
configuration x is terminal iff no single-module mode move has σ·Δ < −4·λ_M. The first
flip costs λ_M + σΔ/4, and later flips continue the same affine-plus-tent profile.
So unit-level exclusivity holds iff every non-target mode configuration has a move
with σΔ < −4·λ_M. Phase B's strict witnesses guarantee a move with Δ ≤ −1, so
exclusivity follows whenever σ > 4·max(λ_S, λ_K).

The implementation must agree with both propositions in **every** setting. A single
mismatch is a defect in the theory or the code, and must be resolved before any
basin numbers are interpreted.

## Grid, declared and fixed

- **Witnesses:** all 14 phase B strict-variant witnesses at n = 2: S, F, and T at
  Levels 1 and 2 where realizable, and G at Level 1.
- **σ ∈ {5, 9, 13}**, all tie-free for λ ≤ 3, and **λ_S, λ_K ∈ {1, 2, 3}**: 378 settings.
- **Pulse strength h = 32,** enough to move A in every setting.
- **Tie control:** the graded v1 witness at σ = 8, λ_S = λ_K = 1, which is predicted to
  have 96 off-prototype terminals.

No other values. Nothing is tuned to rescue a failure.

## Measurements

- **Criteria P1–P5,** as in graded v1, evaluated per setting, together with whether
  they hold jointly.
- **Global census:** terminal states, and spurious terminals split into off-prototype
  and prototype, each compared against the propositions' prediction.
- **All-schedule basins:** a state commits to a target if every schedule from it ends
  there. Computed by dynamic programming on the strict-descent DAG.
- **Random-schedule basins:** each step chooses one free unit uniformly and flips it
  iff the flip strictly lowers E. Absorption probabilities are computed exactly in
  rationals; self-loops do not affect them.
- **Radius profile:** for each target and each Hamming radius r = 1…6, report the
  minimum and mean probability of returning to that same target. The *certain
  radius* is the largest r such that every state within r commits to the target
  under every schedule.

## Questions, with no predicted answer

- Do tie-free settings that also satisfy exclusivity give large certain radii, or
  does inter-target ambiguity dominate?
- Do ternary witnesses differ from pairwise ones in joint P1–P5 or in basin size?
- Does a softer core, λ_K < λ_S, increase the number of settings where P1–P5 hold
  jointly, compared with λ_K = λ_S and λ_K > λ_S?

## Stopping rule

Report the grid as declared. Interpret basin results only if the propositions are
confirmed everywhere. Larger n, more units, and coupling restricted to interiors
need their own contracts.
