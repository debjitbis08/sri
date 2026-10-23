# Core mediation v1, phase C — robust codes make readout coupling inert

**Result:** with structured internals, every mediated target recovers from every
single-unit perturbation, for all 45 phase B witnesses and every λ. That success
is vacuous for mediation. Under readout-only coupling and single-unit descent,
error-correcting prototypes make **every** mode configuration stable, whatever the
tables say. The core can no longer exclude, select, or reorganize anything. No
input pulse ever moved K or B. The control without error-correcting distance lets
coupling act again, and loses recovery.

The declared phase C refutation condition (no λ recovers from all single-unit
perturbations) did not fire. The contract did not anticipate this failure mode,
and it is reported here as the main result.

**Design gaps, acknowledged.** (1) The refutation condition tested recovery
without requiring continued mediation, so an inert system could pass it. (2) The
obstruction below follows from the declared energy and update rules
(`src/core_structured.py`, `System.energy`). It could have been proved before the
sweep was run. Phase B's mediation relied on whole-mode moves; phase C replaced
them with single-bit moves without providing a downhill path between modes. The
implementation matches the construction, and no coding error explains the result. The [contract](core-mediation-v1-contract.md)
fixed the codes, perturbations, λ ∈ {1,2,4,8}, output feedback μ ∈ {1,4}, and pulses
h ∈ {1,…,32}. The [certificate](../reports/core-mediation-v1-phase-c.json) and
[runtime](../reports/core-mediation-v1-phase-c-runtime.json) are separate from phase B.

## Why mediation disappears

**Proposition.** Suppose each module's prototypes are pairwise at Hamming distance
at least 3, modules interact only through readouts, and updates are strict
single-unit descent. Then every configuration with all modules at prototypes is
a strict local minimum, for any tables.

*Proof.* One flip leaves the flipped module at distance 1 from its own prototype
and at least 2 from every other prototype, so its readout is unchanged. Every
coupling, unary, and output term is unchanged, and internal energy rises by λ > 0. ∎

So all 4n mode configurations are stable at unit level, including those phase B
made unstable (R1, R4) or bias-dependent (R2). A module at a prototype cannot be
moved by any other module, because coupling offers no single-flip gradient until
a readout changes, which takes at least two flips against λ. The proposition was
checked exhaustively in 6,624 witness/λ/μ/configuration cases and in randomized
tables (`tests/test_core_structured.py`).

## Measurements

Separated codes, read-only output. Feedback with μ = 1 and μ = 4 gave **identical**
counts, and λ had no effect on any count:

| Perturbation | A | B | K |
| --- | --- | --- | --- |
| 1 flip, readout and full state recovered | 1952/1952 | 1952/1952 | 4056/4056 |
| 2 flips, readout recovered | 0/2928 | 0/2928 | 14172/15468 |
| Full switch to another prototype, recovered | 0/488 | 0/488 | 0/1052 |
| Any other module's readout changed (1–2 flips) | 0 | 0 | 0 |

Two flips in a 4-unit, distance-4 satellite always land on a tie (⊥). Because ⊥
is penalized above every mode, both exits descend, and some schedule reaches the
wrong mode. Core failures occur only at n=3, whose code has distance 4; the n=2
and n=4 codes (distances 6 and 5) correct two flips. These are coding limits,
not coupling effects.

**Control (i), no internals:** a single-module mode move recovers in 119 of 507
cases at mode level, while the matching full-prototype switch recovers in 0 cases
with internals. Structure converts recoverable mode errors into persistent ones.

**Control (ii), distance-one core code:** 1-flip core recovery drops to 3752/4056,
only 76 of 180 witness-λ configurations recover from all single flips, and
satellite perturbations now change other modules' readouts in 1,320 (A) and
2,256 (B) 2-flip trials. Coupling acts again exactly when robustness is lost.

**Control (iii)** is vacuous: phase B has no U or D witnesses.

**Input pulses:** 0 of 8,784 trials met input-selected mediation. With h ≤ λ, A does
not move. With h > λ, A often reaches the pulsed mode (in 1,232 of 1,708 opposing pulses, read-only output), but K and B never move,
so after release the system stays in a non-target configuration (for example XOR
start (0,0,0) → (1,0,0)). The input works only on the module it touches.

**Locality:** the interface certificate of phase D applies to every 1- and 2-flip
satellite perturbation here, but trivially, for the same reason mediation fails.

## Meaning for the question

Phase B showed that a core *can* define mediated stable configurations at the mode
level. Phase C rules out one mechanism: separated prototypes, readout-only coupling,
and strict single-bit descent together. It does **not** show that robustness and
mediation are incompatible in general. Within this construction they exclude each other. Either prototypes are
separated, in which case perturbations are corrected but the core is inert, or they
are not, in which case the core acts but small perturbations propagate and recover
less. The study gives no evidence on which mechanism resolves this; candidates are
listed below and none has been tested.

- **Graded interface:** coupling that depends on distance to prototypes, not only
  on the readout, so the core exerts a per-unit field below the readout threshold.
  This matches the program's graded-coupling hypothesis. It also fits the stated
  intent that the core couples to a satellite's interior: per-unit coupling on
  interior units, with exterior units keeping their error-correcting distance.
- **Block moves:** allow whole-module switches as moves. This returns to phase B
  semantics and only makes sense if a cost model for block moves is declared.
- **Stochastic dynamics:** finite barriers become crossable. This is exactly
  analyzable as a finite Markov chain, but recovery then becomes probabilistic.

Do not tune λ, codes, or penalties to rescue this result. A follow-up needs its own
contract.

## Limits

Hand-declared prototypes; two binary satellites; core sizes 2–4; strict
single-unit descent with all schedules; the ⊥ rule as declared. The output module
cannot move a satellite either, by the same proposition, so its lack of effect is
structural rather than a measured weakness of feedback.

Reproduce with `.venv/bin/python -m experiments.core_structured` (about 6 minutes,
after phase B) and `.venv/bin/python -m unittest discover -s tests -v`.
