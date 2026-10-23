# Graded core interface — basins: ties explain spurious minima; one window governs success

**Result:** both propositions declared in the [contract](core-graded-basins-contract.md)
hold in all 379 settings: the grid's 378 plus the tie control.

- Spurious off-prototype terminals arise **only** from energy ties. Choosing σ so that
  64·λ/σ is never an integer removes them.
- With σ > 4·max(λ_S, λ_K), only the targets are terminal (196 of 196 settings).
  Every start then reaches a target with probability 1 under a random schedule.
- Joint P1–P5 holds in exactly the settings where the witness's target structure
  permits it **and** σ > 4·max(λ_S, λ_K). That is 56 of 378 settings, with no
  exceptions.
- Neither a softer core nor ternary coupling changes this window.
- Basins that hold under every schedule are small: a certain radius of at most 2
  unit flips in any joint setting.

[Report](../reports/core-graded-basins.json) and
[runtime](../reports/core-graded-basins-runtime.json) (about 4 minutes). Every
basin number is exact, from a DP over all 4,096 states, or from rational absorption
probabilities.

## Propositions confirmed

- **P-tie.** Every tie-free setting has zero off-prototype terminals. The declared
  tie control (graded v1, σ = 8, λ = 1) reproduces its 96 off-prototype terminals.
  Moving to σ = 9 at λ = 1 leaves only the 3 targets as terminals. A randomized test
  over 25 additional tie-free tables also passes. The 96 spurious states found in
  review were therefore a tie artifact of choosing σ = 8, not a property of graded
  coupling.
- **P-terminal.** In every setting, the prototype terminals equal the predicted set
  {x : no mode move has σΔ < −4λ_M}. The sufficient condition σ > 4·max λ gave
  exclusivity in 196 of 196 settings. Exclusivity also held in 60 settings that
  fail the condition, where a particular witness has larger gains.

## Joint P1–P5: governed by one window and by target structure

| Witness (class, targets) | Joint settings |
| --- | --- |
| T {(0,0,0),(0,1,1),(1,0,1)} — Level 2 | 14 of 27 |
| S {(0,0,0),(1,1,1)} | 14 of 27 |
| G {(0,0,0),(1,1,1)}, agreement control | 14 of 27 |
| G {(0,0,1),(1,1,0)}, agreement control | 14 of 27 |
| The other 10 witnesses | 0 |

For each of the four, the 14 settings are exactly those with σ > 4·max(λ_S, λ_K):
σ = 5 only at (1,1), σ = 9 for λ ≤ 2, and σ = 13 everywhere. The reasons the other
10 witnesses fail are structural, not parametric:

- **No target with A in the pulsed mode** (for example {(0,0,0),(0,1,1)}): A reverts
  after release. No pulse on A can be credited.
- **Direct A–B coupling replaces the core** (F witnesses): the pulse reaches a target,
  but with the core removed the satellites end the same way, so P4 fails.
- **Two legitimate responses** (the T parity target): from (0,0,0), pulsing A→1 ends
  in (1,0,1) or (1,1,0) depending on the schedule, so P3's requirement of a single
  outcome fails.
- **The core does not follow** (T {(0,0,0),(1,1,1)}): with the pulse held, the
  core's gain from the half-moved state is too small, and A reverts on release.

## Answers to the declared questions

- **Softer core.** No. The joint window depends only on max(λ_S, λ_K): settings
  with λ_K < λ_S, λ_K = λ_S, and λ_K > λ_S gave 16, 24, and 16 joint settings,
  respectively. That difference just counts how many pairs lie under σ/4. A stiffer
  core actually improved exclusivity (112 of 126 settings, against 60 for softer)
  and the mean target probability from a uniform start (0.934 against 0.764),
  across the whole grid, not only inside the window.
- **Ternary versus pairwise.** No difference within the window. The joint-capable
  witnesses include pairwise S, ternary T, and the agreement control G. The only
  Level 2 joint-capable witness is ternary, but that is phase B's pairwise limit at
  n = 2 (S has no Level 2 witness there), not a basin effect. Agreement passes P1–P5
  on its Level 1 targets. As in phase B, those criteria do not by themselves
  separate consensus from compatibility.
- **Basin size and reliability.** Inside the window:
  - Every state reaches a target with probability 1.
  - The certain radius around each target is 2 for the S and T witnesses and 1
    for the G witnesses.
  - For the T witness, random-schedule return to the *same* target averages
    0.89–0.98 at radius 3 and 0.63–0.79 at radius 4.

  So useful basins are reliable for perturbations of up to half a module, and
  probabilistic beyond that.

## Limits

- Two-mode modules of 4 units, n = 2, and phase B witnesses with unit gains.
- Uniform random single-unit schedules.
- σ is chosen tie-free by declaration. The propositions say ties matter, and real
  implementations need a tie rule.
- The certain radius uses the harshest all-schedule semantics.
- Not tested: larger modules, n > 2, coupling restricted to interiors, learning,
  and the locality and scaling claims of phase D.

Reproduce with `.venv/bin/python -m experiments.core_graded_basins` and
`.venv/bin/python -m unittest discover -s tests -v`.
