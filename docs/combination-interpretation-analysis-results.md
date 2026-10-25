# Combination and interpretation: first analytical findings

Completed 2026-10-25. This is a finite mathematical illustration, not a new
capacity or learning result. No SRI operator has been selected.

[Derivations and declared check](combination-interpretation-analysis.md) ·
[checker](../experiments/combination_interpretation_analysis.py) ·
[deterministic results](../reports/combination-interpretation-analysis.json) ·
[computation receipt](../reports/combination-interpretation-analysis-runtime.json)

## Findings

1. **Decomposition is selection among compatible explanations.** Once different
   outer patterns share an inner state, a context-free inverse cannot recover
   both. Stored preference selects an explanation when evidence is incomplete;
   available evidence can select a different member of the same inner class.
2. **Defaults do not determine the operator.** C(0,s)=s is consistent with many
   algebras. Absolute state descriptors and deviation messages must be typed
   separately to avoid applying a preference twice.
3. **A concrete nonlinear interpretation is possible.** In the supplied binary
   construction, the inner state represents same/different. Default outer state
   01 and default inner state different give 01 with no evidence and 10 when the
   first observed bit is 1. Both use the same inner state. Fully observed 00 or
   11 instead changes the inner interpretation to same. All nine observation
   patterns have unique exact minima. The full-data inner response is XOR and
   is not affine in the declared real input coordinates; the single-missing-bit
   completion alone is affine.
4. **Stopping is weaker than correct interpretation.** Starting from defaults
   after observing the first bit as 1, inner-first strict local descent stops at
   energy 3; outer-first stops at the global minimum 2. The former is a strict
   local trap. No oscillation occurs in either schedule.
5. **Relational simplification and low spatial frequency are distinct.** The
   example identifies a pattern with its simultaneous complement. That is not
   the equivalence induced by ordinary averaging. A geometry relating these
   notions still needs to be defined, rather than assumed.

## Verification and cost

Executed `python3 -m experiments.combination_interpretation_analysis`.
All exhaustive assertions passed: 9 evidence patterns, 72 state visits,
32 feasible states scored, 2 local schedules with 8 proposals and 2 accepted
updates, 4 neutral-combination table entries, and 4 message round trips.
There were 43 energy evaluations in total. No fits, tuning, empirical examples,
or optimizer steps were used.

Successful checking and scientific serialization took about 0.00049 seconds
wall and CPU on the recorded Ryzen 5 3600 / Python 3.12.3 environment. This
excludes interpreter startup, provenance collection, and runtime-receipt writing;
it is a one-off cost receipt, not an inference benchmark. Peak process RSS was
16,448 KiB, including interpreter/imports. The supplied preference/configuration
occupies 653 recursively counted Python object bytes, or 120 serialized JSON
bytes; there is no learned state or replay. Complexity and the independent
multi-coordinate extension are derived in the analysis, sections 7–8.

Development disclosure: `python` was unavailable, so the actual invocation uses
`python3`. One initial assertion exposed an arithmetic transcription error in
the local-trap neighbor energy (6 instead of 8). The assertion and prose were
corrected without changing the model, coefficients, or check menu. That failed
attempt did not write artifacts or serialize timing; its wall/CPU costs are
explicitly unavailable in the receipt. Manual development cost is not measured.

## Next mathematical obligation

Find a family that combines declared mode geometry, contextual interpretation,
and a tractable realization procedure. The finite example supplies interpretation
and an exact tiny reference; it does not supply learned modes, learned relations,
a scalable local solver, or a witness on the existing relational task. Higher
dimensions are cheap only under the explicitly independent-coordinate assumption.

Preserve the distinction between a constructed relation and one acquired from
evidence. No extension of the previous Stage B search is justified by this check.
