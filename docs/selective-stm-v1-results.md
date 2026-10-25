# Selective STM with frozen long-term weights

Selective short-term memory retains the previously inferred pattern after the
input is removed, but makes replacement slower in this protocol. No long-term
weights changed. Major learning through sleep/dream remains an intended mechanism,
not an implemented feature of this study. The existing least-squares pretraining
must not be described as sleep learning.

Only the intermediate module has an explicit memory trace, covering its 15
activities. During input-present intervals it tracks activity at write rate 0.2;
during absence it is held fixed. Its readback strength lambda is 0, 0.1 or 1.
The endpoints have no such trace. Cue presence controls writing explicitly; this
gate is supplied, not learned. There is no trace decay. All three conditions
perform the same trace writes, with lambda=0 disabling memory influence.

The eight existing held-out signals are presented for 1024 updates, removed for
1024, replaced by their opposite-scalar partners for 256, then removed for another
256. States and trace carry through all four phases. Output is always the output
module's outer activity. No inner target or desired output enters the dynamics.

| STM readback strength | End encoding | After first absence | End replacement | After second absence |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.17683 | 0.25577 | 0.34454 | 0.25842 |
| 0.1 | 0.17690 | 0.19522 | 0.37594 | 0.28713 |
| 1 | 0.17769 | 0.17793 | 0.42738 | 0.34646 |

Values are outer-output RMSE against the currently desired pattern. Strong STM
retains roughly the encoding accuracy through the first absence. All eight strong
STM queries are below activity-gradient norm 1e-5 at its end, although outputs
are emitted at every declared horizon regardless of residual.

The replacement interval shows the retention/update tradeoff: stronger STM gives
larger error after equal new-input exposure. Conditions begin replacement from
different states, because the old pattern was retained to different degrees.
This does not isolate a pure write-speed effect from the distance to the new
pattern. It does demonstrate the operational cost of retention in this sequence.
The shorter replacement interval leaves a partially updated trace, which can then
hold a mixture during the second absence. This is not evidence that replacement
is impossible with more time or a different gate.

A directly stored scalar input mean with the same write rate/gate, followed by
the frozen output reconstruction, has near-machine-precision error by the end
of each phase. Its memory costs 8 bytes versus 120 for the middle trace. Therefore
this simple task still offers no evidence of a computational benefit from the
recurrent architecture. Holding an undegraded trace during absence also builds
persistence into the protocol; the experiment measures its influence on output,
not spontaneous discovery of memory or an autonomous write policy.

A useful next distinction is separate control of writing, retention/decay and
readback strength. This run varied readback strength only. No strength was selected
or tuned, and no follow-up runs were added after seeing the results.

## Verification and computation

Every long-term parameter remained byte-identical. Finite-difference gradient
error was 3.96e-10, and one lambda-zero update reproduced the existing v3 update
exactly. Input clamps, activity descent for fixed memory and trace-write descent
were checked. No backtracking halvings were needed. A changing trace changes the
objective between steps; this is not a global convergence proof.

Execution took 11.52 seconds including checks, input loading, inference, trace
writes, direct comparator, scoring and scientific serialization. There were
61,440 main activity updates plus one equivalence-check update, 30,720 middle
trace writes, 30,720 comparator trace writes and 528 outer readouts. Eight unique
held-out pairs were reused in 24 episodes with 48 cue presentations. New fits,
weight-optimizer steps and selection/tuning costs were zero. Historical v3
pretraining used nine pairs and is separately recorded in its runtime artifact;
the v3 full study took 25.56 seconds, not all of which was fitting. Manual
preparation time is unmeasured.

Persistent array storage is 1,272 bytes: the existing 1,152-byte chain plus
120-byte STM, excluding Python overhead and transient workspace. The trace is
retained history and is counted. No optimizer state, raw-input replay or newly
fitted statistics are retained. Peak single-process RSS was 35,792 KiB. For K
modules of width N with p local coefficients, an activity update is O(KNp) plus
O(N) STM work; T steps with J energy trials cost O(TJKNp). Initialization loads
O(KNp) parameters. This is a whole-run mechanism measurement, not an inference
latency benchmark; no speed claim is made.

Artifacts: [contract](selective-stm-v1-contract.md),
[runner](../experiments/selective_stm.py),
[scientific results](../reports/selective-stm-v1-results.json),
[runtime, counts and hashes](../reports/selective-stm-v1-runtime.json).
