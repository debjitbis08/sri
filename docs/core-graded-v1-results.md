# Graded core interface v1 — recovery and core participation coexist

**Result:** when coupling acts on internal units through the multilinear extension
of the mode energy, one parameter setting gives both outcomes:

- every single-unit perturbation recovers;
- an input on A alone makes the core switch modes and stabilize A's new mode.

That change persists only with a dynamic core. If the core is removed or clamped,
A reverts. This is a feasibility result for one declared example, not a sweep.

The [contract](core-graded-v1-contract.md) stated a transition proposition and its
predictions before implementation. The criteria in the [report](../reports/core-graded-v1.json)
are exhaustive over the states *reachable* from the declared perturbations and
pulses, under every schedule; that is not every unit state. A separate global
census over all 2^12 states, added after review, is reported below.

## Proposition, verified

With other modules at prototypes, a strictly descending single-flip path between
modes of module M exists iff σ·Δ < −4·λ_M, where Δ is the mode-level energy change.
Mode-stable targets are unit-level strict minima for every λ_M ≥ 0. A randomized
exhaustive test checks the threshold for 60 tables and every module move from every
configuration (`tests/test_core_graded.py`), including λ = 0.

## Worked example: ternary witness, targets (0,0,0), (0,1,1), (1,0,1)

σ = 8, h = 8, and the same λ for every module:

| Criterion | λ = 1 | λ = 2 | λ = 3 |
| --- | --- | --- | --- |
| P1, targets are unit-level strict minima (every flip raises E) | yes | yes | yes |
| P2, all single flips recover full state | yes | yes | yes |
| Pulse A→1 from (0,0,0), core dynamic | **(1,0,1)**, K switched | (1,0,0), stuck | (1,0,0), stuck |
| Same pulse, core removed | (0,0), A reverts | (1,0) | (1,0) |
| Same pulse, core clamped at 0 | (0,0,0), A reverts | (1,0,0) | (1,0,0) |
| P1–P5 jointly in one setting | **yes** | no | no |

Every entry matches the contract's prediction. At λ = 2 the core's first flip has
gain exactly zero, so it is blocked. This is the phase C obstruction reappearing
once the barrier reaches the coupling gain.

The other pulses are informative, but none is credited:

- From (0,1,1), a pulse A→1 reaches (1,0,1): the core transmits A's change to B.
  Clamping K at 1 gives the same outcome, so this is fixed-context bias (P5 fails).
  It is reported, not credited.
- From (1,0,1), a pulse A→0 ends in either (0,0,0) or (0,1,1), depending on the
  schedule. These are two legitimate responses, so no single outcome is credited.

## Global census, added after review

The original P1 implementation checked only that no flip lowered energy, which
equal-energy neighbours would also pass. It now checks that every flip strictly
raises energy, as the contract states, and the result is unchanged. Enumerating
all 4,096 unit states shows the declared criteria do **not** imply global
exclusivity:

| Setting | Terminal states | Spurious (non-target) terminals | States committing to a target under every schedule | Ambiguous states |
| --- | --- | --- | --- | --- |
| λ = 1 | 99 | 96 | 2,013 (671 per target) | 1,923 |
| λ = 2 | 104 | 101 | 681 | 2,628 |
| λ = 3 | 8 | 5 | 375 | 3,096 |

At λ = 1 the spurious terminals fall into six popcount classes of 16 unit
arrangements each. Examples: A, B, K holding 0/1/3 ones, which reads (0,0,1), and
3/3/4, which reads (1,1,1). None of them is a prototype configuration. These are
interior local minima of the multilinear energy on the popcount lattice.

At λ = 3 the only terminals are the 8 prototype configurations. Every mode
combination is stable, which is the phase C inertness again.

*Ambiguous* means different schedules from that state reach different readouts.
This all-schedule notion is the harshest available; under a random schedule, some
of these states would commit with high probability. That was not measured.

None of this changes the declared perturbation and pulse results: no run reached
a spurious terminal. But target exclusivity, which phase B imposed in its strict
variant, does not survive the move to unit level.

## Side observations, not scored

- **2-flip perturbations:** recovered in every trial at λ = 1 and 2, and in none at
  λ = 3. At a tie, the graded coupling slope σΔ/4 picks the correct side only while
  it exceeds λ. Phase C's ⊥ rule had no such slope.
- **Full module switches:** they never restore the original state. At λ = 1 the
  other modules then reorganize into a different target. At λ ≥ 2 nothing moves.
- **Locality:** no 1- or 2-flip perturbation ever changed another module's readout,
  at any λ.

## What this does and does not establish

It establishes that the phase C obstruction is specific to readout-only coupling.
A graded unit interface admits recovery and input-driven core participation in the
same setting, with matched removal and clamp controls. The working window is
σ·|Δ| > 4·λ for the move the core must make. In effect, internal barriers must be
smaller than the scaled coupling gains of desired mediated moves, and larger than
nothing.

Not established:

- basin size and reliability: under every schedule only about half the state space
  commits to a target at λ = 1, and 96 spurious minima exist. This is now the main
  open question, not whether core participation is possible;
- generality across witnesses, σ, per-module λ, n > 2, or pairwise classes;
- whether the core needs a lower barrier than satellites in general;
- the coupling-to-interior restriction, which is not implemented here;
- the scaling and locality claims of phase D.

The witness and its σ were chosen by analysis, not by search. A grid requires its
own contract.

Reproduce with `.venv/bin/python -m experiments.core_graded` (seconds) and
`.venv/bin/python -m unittest discover -s tests -v`.
