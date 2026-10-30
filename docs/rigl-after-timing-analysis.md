# After the timing result: selective resets and predictor attrition

2026-10-30. Hypothesis development and read-only checkpoint audit, not a new
training contract. Sources: [timing results](fan-in-pilot-v1-timing-results.md),
`rigl_step` and `move_rows` in `experiments/fan_in_pilot.py`, and final RigL
checkpoints for all ten timing seeds 230–239. No new external literature claim
or novel-method claim is made; earlier RigL reading is in [prior work](prior-work.md).

## What the result rules out

The chosen coordinate plus the tested one-way concentration schedule are not
sufficient for RigL's transfer. English comparison is .75 for RigL versus .59
for gradual concentration on RigL's final coordinate; Devanagari is .81 versus
.70, taps to withheld Chinese .66 versus .26. Coordinate choice contributes no
detected benefit. The actual training trajectory is not matched: concentration
was not logged, and the gradual schedule was obtained by integrating cosine
alpha, whereas k=1 always reconsiders its single connection. Therefore the
result does not exclude every timing-dependent mechanism.

My earlier emphasis on finding a privileged coordinate is weakened by this
result. A coordinate's numerical identity is not the interesting missing cause.
The surviving hypothesis is adaptive training dynamics: gradient-conditioned
connection survival, repeated erasure of weights/moments, and loss of predictor
units. Final concentration could be an effect of that process as well as a cause.

## A mechanism specific to this k=1 implementation

At a rewiring event, each row's sole connection in each operand is dropped and
the largest absolute dense-gradient coordinate is selected. The same coordinate
can be selected again, preserving its weight; changing coordinate sets its new
weight and Adam moments to zero. At k=1 there is **no magnitude-based choice
among active edges within an operand**: there is only one. Nor does the code
select only weak rows to reconsider. The report's generic description of
"moving units whose connections matter least" does not describe this special case.

Thus every 100 updates until 4,500 the reader repeatedly tests whether its
current feature remains preferred, and erases the connection if it does not.
The gradual control instead moves each row at most once and carries its old
weights to a different feature. Those are substantially different operations.

**Hypothesis:** features consistently useful across batches survive, while
unstable script- or context-specific solutions repeatedly lose learned weights
and optimizer state. This may select reusable quantity computations. It is not
guaranteed: rare but useful features can also be erased, and raw gradients are
scale-sensitive. The fall in taught matching/retention is compatible with either
useful regularization or destructive forgetting.

An illustrative independent-reset model makes the relation to memory decay
explicit. If a connection resets with probability p at each 100-update event,
the surviving component of an old contribution has expectation (1−p)^k after
k events. Its effective half-life is

    H_eff = 100 ln(2) / [−ln(1−p)].

This is a survival calculation, not an estimate of RigL's reset probabilities.
Actual resets depend on gradients and are correlated with learning. It suggests
that RigL may already implement **selective forgetting**, unlike the revised
proposal's uniform decay of a preassigned dense subset.

## Why P is the first place to look

P's first-layer units have only the two code operands. A k=1 unit is

    h=ReLU(w_a c_i + w_b c_j + b).

If both selected coordinates change at an event, both new active weights are
zero. If b≤0, h is then zero for every possible code pair. Dense growth
gradients, incoming loss gradients and bias loss gradients are zero at that
state (PyTorch uses zero ReLU derivative at zero). If bias remains nonpositive,
future zero-gradient tie-breaking cannot revive it. Adam's retained bias
moments can transiently move b, so immediate permanent absorption is not proved
without its optimizer state. The final checkpoints do not save that state.

The relation reader g additionally has unmasked relation-token inputs. These
can keep its units active even when both code weights reset. This gives a
mechanistic explanation for the *asymmetry* in dead units, without yet explaining
transfer: P loses roughly half its units, g roughly one eighth.

The audit checks a stronger condition than inactivity on evaluation samples:
all effective code weights are exactly zero, and maximum preactivation is
nonpositive over the three relation tokens (or simply bias≤0 for P). Such rows
cannot respond to any real code input at this checkpoint. It does not infer
when or why those rows became inactive. Results are in
`reports/rigl-reset-diagnostic.json`.

**Measured:** across all ten seeds, P has 28–37 globally inactive units out of
64 (mean 31.5); g has 3–12 (mean 7.1). Every P row with exactly zero effective
input weights has a nonpositive bias. P has 31.9 nonpositive-bias rows on average,
so nearly all of that final subset is structurally inactive. This describes
final biases, not their initial values or proof that resets caused each death.
The original evaluation-only dead-unit diagnostic is broader and can also count
nonzero-weight units inactive on the sampled codes.

**Leading combined hypothesis:** repeated resets make complex reader solutions
fragile; P loses many units and must predict from a simpler representation;
the encoders adapt to make quantity easier to read. This could yield both the
near-rank-one code and broad transfer. But 32 live units still suffice for
nonlinear computations: neither reduced width nor inactivity mathematically
forces a one-dimensional code. P's squared-distance prediction loss also
touches all target-code coordinates, making its role different from g's.

The facts to explain jointly are strong transfer, improved form probes,
collection participation ratio 1.02, severe P attrition and worse taught fit.
A story about final masks alone does not explain that combination.

## Discriminating checks, in order

1. **Localize the intervention:** apply original RigL to P alone versus g alone,
   with both readers starting from the same random k=1 masks as the original
   arm. The other reader retains those masks. Compare both with the original
   both-RigL and random arms on fresh, paired seeds. This tests locus before
   introducing an arbitrary smaller architecture.
2. **Separate selection from erasure:** retain RigL's gradient-based selection
   but carry old weights to new coordinates, with the same moment-reset rule.
   Unlike gradual's predetermined destination, this keeps adaptive selection.
   Conversely, a reset intervention without coordinate changes tests erasure,
   but its event rate and per-row selection must be specified before running.
   Adaptive trajectories will diverge; these are mechanism interventions, not
   identical-mask counterfactuals.
3. **Separate history from final architecture:** retrain with a frozen final
   mask and the same permanently disabled rows, against the original dynamics.
   This is an oracle topology from a completed run; account for its source
   cost. Merely reducing P to 32 random units does not reproduce which units
   died or when, so a negative result there is weak evidence against attrition.

Instrument whichever training comparison is chosen: before/after rewiring,
measure per-loss jumps, code spread, activation loss, concentration and reset
counts per reader. Evaluate at language boundaries and rewiring cessation.
Prediction of the reset hypothesis: resets cause transient damage, followed by
recovery that increasingly relies on common quantity structure. Causal support
would require the interventions, not just this trajectory correlation.

Do not launch all controls at once or tune them to the final scores. A new
contract must specify priors, compute and comparison rules. The cheapest current
step is the final-state audit; it tests whether the proposed failure mode is
actually present without a new training run.

## Computation and scope

`.venv/bin/python -m experiments.rigl_reset_diagnostic` loads ten existing
checkpoints, computes exact zero-row and preactivation tests, and hashes inputs.
With S seeds, P stored parameters, m reader units, d code width, token width t
and R tokens, time is O(S(P+md+mRt)); memory O(P+md+mR) while processing one
checkpoint at a time, plus O(S) scalar outputs. Checkpoint parsing/hashing is
included. Here S=10, m=64, d=32, t=8, R=3. No data rendering, fits, optimizer
steps, candidate selection or random sampling occur. Scientific counts and
nondeterministic timing/RSS/environment are separate JSON files. Final states
can verify structural inactivity, not its onset, cause, or causal role in transfer.
