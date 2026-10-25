# Outer-state readout before settling: v2 results

Completed 2026-10-25. Outputs are outer activations only. Inner coordinates are
diagnostic or supplied class context; inner classification is not task success.
The initial inner-output draft was corrected before implementation/execution.

[Contract](outer-anytime-v2-contract.md) ·
[runner](../experiments/outer_anytime.py) ·
[deterministic results](../reports/outer-anytime-v2-results.json) ·
[runtime receipt](../reports/outer-anytime-v2-runtime.json)

## 1. What changed

The learned eight-unit decoder and four output scores from v1 are gone. Four
outer units interact through four local disagreement states and two inner
coordinates: disagreement fraction and parity. A single four-entry bilinear
operator is shared across outer and inner interactions. With XOR entries
(0,1,1,0), the inner coordinates distinguish the four domain relationships
exactly; this is a supplied sufficient representation, not discovered geometry.

The readout is x itself, optionally thresholded at 0.5. It has no learned weights
or hidden layers and can distinguish all 16 binary outer patterns. Inner code
checks only diagnose whether the operator has retained the expected structure.
During completion the class is supplied as context, as in v1; hidden outer bits
are never supplied. This is conditional reconstruction, not unlabeled prediction
of a class from an ambiguous partial input.

Every query emits its actual outer state at steps 0,1,2,4,8,16,32,64. Neither
confidence nor equilibrium gates these outputs. Intermediate mixed activations
are saved. Correctness concerns observed-bit preservation and semantic outer
completion, not whether the state has stopped changing.

## 2. Correct answers can precede equilibrium

Using supplied XOR and frozen weights, the initial default-with-observations
matches 18/48 semantic completions. At 64 updates, 30/48 match. All 12 newly
correct answers still have activity residual above 1e-5: they are usable task
answers before settling by the earlier criterion. No observed bits were changed.

| Update horizon | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XOR, frozen: correct outer completions /48 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 30 |
| XOR-initialized, adapting: correct /48 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 21 |
| Frozen: mean hidden continuous error | .3958 | .3958 | .3958 | .3938 | .3827 | .3519 | .2971 | .2284 |

There is useful continuous movement before any additional thresholded answer
becomes correct. This supports reporting graded outer activity rather than treating
all nonsettled responses as failures. It does not establish that all earlier
discrete decisions are reliable: the thresholded improvement first appears at
the declared 64-step observation. Times between those samples were not scored.

Adaptive probes reach only 21/48, with three newly correct unsettled answers.
Their mean operator drift at step 64 is 0.02684. Thus ongoing parameter changes
can interfere with an already useful operator, even though every numerical step
decreases the current energy. Each probe starts from a fresh copy and never
contaminates another probe or the main stream. No correct-to-incorrect reversals
were observed at the sampled horizons; missed between-sample reversals are not
ruled out.

## 3. The representation is learnable, but joint adaptation still fails

Direct gradient training of the SAME four local parameters recovered XOR to
floating-point precision in all three replicas after block A and retained it
after block B. No learned decoder is available to absorb the work. All 16
full-input inner diagnostics pass, and the actual outer completion curves match
the supplied-XOR curves above for every replica/checkpoint. In particular,
frozen completion reaches 30/48 and adaptive completion 21/48 at step 64.

The simultaneous joint-energy learner did not recover that operator. Its tables
after A were approximately:

| Replica | theta00 | theta01 | theta10 | theta11 |
| --- | --- | --- | --- | --- |
| 0 | .3306 | .4013 | .3969 | .3665 |
| 1 | .3132 | .3868 | .3863 | .3725 |
| 2 | .3114 | .3907 | .3854 | .3769 |

After B, all entries lay between .469 and .501. These operators respond weakly
to their local input values. Their actual outer completion remained 18/48 at
every horizon, with or without probe plasticity. Their 4/16 inner diagnostic
accuracy is supplementary evidence, not the reported task output.

This is a clearer separation than v1: a simple representation and readout now
have a constructive solution, and the local operator can be trained by the direct
comparator under the same stream. The remaining failure is specific to the
declared joint-energy adaptation procedure/objective and budget, rather than an
established impossibility of representing the relation. It does not identify a
unique cause within that procedure or exclude all continuous local learning rules.

Direct gradient training is online parameter adaptation with an analytically
differentiated direct function; it is not the desired local reciprocal activity/
parameter dynamics. Do not describe it as having solved that remaining problem.

## 4. Readout simplicity does not make inference free

The supplied-XOR benchmark used all 48 queries, one warmup and three measured
batches. It includes state initialization, updates, gradient/line-search work,
outer readout, allocation, and a semantic oracle call for scoring each query.
It is not the cost of merely reading an already available outer value.

| Method/budget | Correct /48 | Median amortized milliseconds/query |
| --- | --- | --- |
| Direct 16-pattern semantic oracle | 48 | 0.139 |
| Continuous circuit, 1 update | 18 | 0.431 |
| Continuous circuit, 8 updates | 18 | 1.646 |
| Continuous circuit, 64 updates | 30 | 11.548 |

The oracle uses the supplied discrete class definition, not the learned parameters
or continuous energy, and is the simpler task reference. These are distinct
quality–cost points, not equal-accuracy speed comparisons. The continuous circuit
does not achieve all semantic completions in this budget and does not establish
an efficiency advantage. Benchmark min/max and CPU samples are in the receipt.

## 5. Verification and computation

Executed `.venv/bin/python -m experiments.outer_anytime` once, after the corrected
contract and code were fixed. All correctness assertions passed on the first run.
Maximum finite-difference discrepancies were 2.51e-11 for joint derivatives and
3.11e-11 for direct derivatives. Exact XOR codes and positive diagnostic margins
were checked on all 16 patterns. No nonfinite values, clamp violations, rejected
line-search trials, or exhausted searches occurred.

All six learned trajectories were reported: 204,000 training updates over 10,200
presentations of ten distinct labeled patterns. Completion diagnostics used
1,248 queries, 79,872 updates, and 9,984 outer readouts. Including correctness
and benchmark work, there were 195,889 joint steps, 102,000 direct training steps,
401,793 energy calls, 205,874 joint gradient calls, and 206,153 direct calls.
No best-seed selection, budget extension, or post-result tuning occurred.

Whole-run wall/CPU time was 49.105/49.095 seconds. Peak process RSS was 88,272 KiB,
including all curves and JSON serialization. Each deployed model has 32 learned
parameter bytes, 80 activity bytes, and 64 supplied prototype bytes (float64),
plus query/implementation overhead. No optimizer state or replay. Allocator-level
temporary workspace was not separately measured. Source/contract hashes,
revision/dirty state, environment, phase costs, and all timing samples are archived
separately from deterministic science.

## 6. Decision

Keep outer-state readout and the full time/quality curve as the task interface.
Inner diagnostics must remain explanatory rather than substituted outputs.
Complete settling is not required for a valid answer; this run gives explicit
correct outer answers while the state is still moving.

The next learning question is how to adapt the local operator without treating
temporary inference mismatch as a reason to erase its useful distinction. The
supplied/direct-learned XOR models are now available as positive mechanism
controls. Any proposed slower, gated, or trace-based plasticity needs a new
prospective test; this run does not justify quietly changing the rates or
freezing weights and calling the result continuous adaptive inference.
