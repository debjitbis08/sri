# Local consolidation: reviewer test, first bounded result

The proposed module-local rule produces changes in network-level reconstruction,
but this first test does **not** validate stability-gated consolidation. Ungated
local energy learning has the best aggregate reconstruction among the learning
arms. No learning arm improves the second structure's cross-module reconstruction
during the second sleep phase. The first structure is retained, partly in a
protocol that explicitly replays its traces. Perfect discrimination is already
present before learning, making that part of the fixture too easy.

The tested candidate remains a hypothesis, not a successful learning mechanism.
A separate exact-inference audit shows that the second-structure failure is not
merely insufficient settling in the 32-step readout protocol.

## Candidate rule and locality boundary

Every module has the same 8→4→2 architecture and the same update rule. None is
designated an input or output during learning. For each local pair of layers,
residuals propose weight changes using only presynaptic activities, postsynaptic
activities and that module's stored relationships:

`ΔW = η g/(1+s/8) * (mean[(x-Wh)hᵀ] - 0.001 W)`

The adjacent inner relationship V uses the analogous local residual. A local
stability gate g may reduce the proposal when inner activities are changing.
During consolidation, the candidate is checked on different stored episodes:
hide one actually observed outer coordinate, infer it from remaining evidence,
and retain the candidate only if that module predicts the hidden observation
better across at least three distinct contexts, without excessive individual
regression. The hidden value is available only to the local check, not inference.

Support s counts distinct accepted supporting episode IDs, once per episode per
module. This reduces subsequent update sizes as evidence accumulates. Stability
permits change; it does not authorize retention by itself. No pattern identity,
external task reward or global error gradient enters this rule. Its local
prediction error is still an explicit self-supervised learning criterion. The
rule is module-local, not strictly single-synapse-local: acceptance aggregates
observations from within that module and uses a local trace buffer.

Inference propagates a local parameter change through the network, so it can
have global effects without differentiating a global loss. That does not guarantee
those effects are beneficial. Local coordinate completion and reconstruction of
a wholly unobserved module are different prediction problems.

The current implementation learns local W/V matrices only. Fixed chain coupling
0.2 compares inner coordinates; unlike the earlier frozen studies, local encoders
start random and no correct pattern mapping is supplied. Learned cross-module
alignment and feedback strength remain unimplemented. Unknown alignment is a
possible contributor to failure, not a cause isolated by this experiment.

## Reviewer protocol

Three seeds, two hidden structures, each represented by 16 waking variations.
The learner receives three observed 8-dimensional surfaces and episode IDs, never
structure IDs or the private generator. Variations include shared amplitude and
independent local nuisance. Train structure A, consolidate; then train structure B,
consolidate again with retained A traces. Half of each stage's episodes provide
proposals; half provide adaptive local checks. All arms retain the same bounded
32-trace pool. Waking and evaluation leave long-term weights unchanged.

Each sleep phase has six cycles and 18 module proposals. Activities use 32 implicit
Euler steps per call. Parameters and timing choices were fixed prospectively;
there was no tuning or extension after results. Held-out evaluation uses 12 new
variations per structure and rotates which module's entire outer surface is hidden.
The other two surfaces provide evidence. Separate evaluator-only reference examples
support centroid classification. Evaluation labels never enter consolidation.

Eight arms separate energy proposals, stability gating, predictive-support checks,
accumulated inertia and slower inner activity. A repeated-single-trace arm matches
replay counts but cannot pass the predeclared diversity requirement. Its rejection
is a designed bookkeeping control, not discovered evidence for replay diversity.

## Results

Mean held-out missing-module RMSE across three seeds, after both sleep phases:

| Learning condition | Combined | Structure A | Structure B |
| --- | ---: | ---: | ---: |
| Frozen | 0.37367 | 0.35968 | 0.38699 |
| Local energy updates | **0.33915** | 0.27621 | 0.39036 |
| Stability-gated energy updates | 0.33955 | 0.27726 | 0.39031 |
| Stability gate with slower inner activity | 0.36542 | 0.35377 | 0.37663 |
| Predictive support, no stability gate | 0.36426 | 0.33818 | 0.38790 |
| Predictive support plus stability gate | 0.36443 | 0.33856 | 0.38790 |
| Same, without accumulated inertia | 0.34838 | 0.29993 | 0.38935 |
| Repeated-single-trace support | 0.37367 | 0.35968 | 0.38699 |
| Direct linear predictor using the same waking data | 0.09554 | — | — |

The slow-inner arm starts with different inference behavior: its lower B error
relative to ordinary frozen inference is not a learning benefit. B error increases
during its second sleep phase too. Across every seed and every updating arm,
B reconstruction becomes worse after the B phase than immediately before it.
For example, energy learning goes from mean B RMSE 0.38335 to 0.39036; support
plus stability goes from 0.38529 to 0.38790. Thus improvements in combined error
come from A and do not establish acquisition of both structures.

A is retained within the declared 10% margin in all seeds/arms; updating arms
actually improve its reconstruction after B. This is not evidence of replay-free
resistance to forgetting: old observations are still in the sleep buffer.

Mean stability gates are 0.99077 for energy/stability and 0.98999 for support/
stability. The gate intervention is weak in this regime. The comparison finds no
additional benefit from it; this does not disprove more informative stability
signals, longer learning, or other operating regimes. None was selected or tried
post hoc in this run.

Support counts reach 16 distinct checking episodes per module. Inertia therefore
reduces the proposal scale from 1 initially to 1/3 for a fully supported module.
Mean parameter drift is 0.52655 with support/stability/inertia versus 0.97486
without inertia. That is the intended resistance to change, but its functional
benefit is unestablished here. Counts are module-wide and based on prediction
improvement, not a calibrated confidence for each individual weight. Replay of
the same ID cannot increase support repeatedly.

Across three seeds, energy/stability/slow-stability each accept all 108 proposals;
support and support/stability accept 92; the no-inertia support arm accepts 90.
Frozen and repeated-trace arms accept none. Differences in update opportunities
and accepted magnitudes remain visible rather than being mistaken for a uniquely
beneficial stability mechanism.

## Abstraction and the inertia control

All ordinary initial models already achieve 100% evaluator centroid accuracy on
the two structures; every arm stays at 100%. Random inner features suffice for
this discrimination task. Consequently this result cannot demonstrate learned
class discrimination or discovery of hidden identities.

The mean within-structure/between-structure squared-distance ratio falls from
0.05708 (frozen) to 0.01419 (energy), 0.02795 (support/stability), and 0.01014
(slow-inner/stability). But the slow-inner arm reconstructs worse than energy
learning. Absolute within-spread, centroid separation, code norm and activity
residuals are also retained. The ratio is not a stand-alone abstraction score:
shrinkage, increased separation, or inertia can change it without useful prediction.
An explicit zero-code diagnostic has zero spread, zero separation and 50% balanced
tie-broken centroid accuracy. Its collapse would not satisfy the criteria.

The full reviewer-positive condition fails: there is no second-structure
acquisition, no demonstrated gate benefit, and discrimination was already easy.
Three seeds support a bounded descriptive comparison, not a significance claim.

## Post-run diagnosis: local checks do not ensure global prediction

A separately costed audit uses the frozen final weights, with no further learning.
It evaluates single-coordinate completion on unseen data and solves the same
network's constrained activity equations exactly when an entire module is hidden.

Energy learning improves unseen single-coordinate completion for both structures
in all three seeds. Yet it worsens B reconstruction when that entire module is
unobserved, even under exact inference. Support/stability improves local B
completion in two seeds and worsens it in one, while also failing global B
completion. Accepted changes on adaptively reused local checking traces therefore
provide neither guaranteed held-out local generalization nor guaranteed global
reconstruction.

Exact B reconstruction RMSE for energy learning is 0.39588, 0.38525 and 0.41082
versus frozen 0.38231, 0.38314 and 0.40828. Support/stability also remains worse
than frozen in all three seeds. Waiting for the exact equilibrium does not fix
this failure. These are diagnoses of the existing models, not additional trained
candidates or extensions of the registered trajectory budget.

A useful next design question is whether local relationship checks must explicitly
test transmission across neighboring modules or more diverse missing-evidence
conditions. Simply increasing support for successful one-coordinate completion
has not established the broader usefulness we intended. Such changes need a new
prospective contract, not a retrospective success claim for this study.

## Verification, computation and retained state

Finite-difference errors: activity 2.10e-10, local parameters 4.75e-10. Checks also
verify clamping, implicit reciprocal descent, byte-identical waking/evaluation
weights, hidden-coordinate leakage exclusion, chain-reversal consistency of
inference, and repeated-ID support bookkeeping. Sequential module proposal order
is fixed and may affect the learned result; arbitrary scheduling invariance has
not been established. Frozen and repeated-support parameters remain exactly at
their shared initialization.

The main run takes 0.8122 wall seconds / 0.8068 process CPU seconds before runtime
serialization: consolidation 0.6150 s, all evaluator/probe calls 0.1347 s, waking
0.0158 s, direct comparator 0.00134 s, plus construction/checks/serialization.
The post-run diagnosis adds 0.0772 wall seconds in a separate runtime artifact.
These are whole-run/phase measurements of a very small mechanism study, not warmed
inference latency benchmarks or hardware scaling results.

There are 96 independent waking episodes, 72 independent held-out episodes and
48 evaluator reference episodes across seeds. Reusing these across arms does not
multiply independent data. Main execution includes 2,140 inference calls,
29,576 inferred examples and 946,432 activity-update/example operations; 10,368
proposal and 10,368 checking-trace exposures; 432 local candidate checks; 598
accepted and 158 rejected learning proposals. The proposal counter is 757 because
it also includes one finite-difference verification proposal; 756 are learning
candidates. Frozen replay still incurs its declared activity work.

The direct comparator performs 18 ridge fits, with 432 cumulative fitting-example
exposures and 432 evaluation-example exposures. It uses all currently available
waking traces, including the local-check pool, but no hidden labels. There is no
hyperparameter search, rejected tuning grid or chosen deployment model. All local
candidate evaluation and rejected proposals are charged to consolidation, not
omitted as free selection. Manual design cost is unavailable.

For state dimension S=42, retained batch B<=32 and 32 iterations: dense Hessian
construction O(L S²), implicit factor solve O(S³), updates O(T B S²); proposal
formation O(B P), P=120 weights. Local checking repeats inference for both
incumbent and candidate. The direct predictor solves three ridge systems with
F=17 inputs including intercept, O(B F²+F³) each. All costs and numerical-library
metadata are recorded. BLAS thread environment is set to one before importing
NumPy; actual active backend thread count is unmeasured. No child workers or
accelerators are used.

Per learner, weights occupy 960 bytes, retained observations 6,144 bytes, episode
IDs 256 bytes and maximum support IDs 384 bytes: 7,744 persistent numeric bytes.
Additional candidate copy 960 bytes, batch state up to 10,752 bytes, dense Hessian
14,112 bytes, and an implicit matrix upper bound 14,112 bytes are workspace.
There is no optimizer history. Python containers, solve workspace and evaluator
artifacts are additional; main process lifetime peak RSS is 41,380 KiB. The direct
predictor retains 3,264 coefficient bytes after fitting, plus any retained training
pool if kept. Do not compare parameter bytes alone while ignoring replay.

Historical frozen studies are different tasks with supplied relationships and
are not learning comparators. Preserve their artifacts unchanged.

Artifacts: [prospective contract](local-consolidation-v1-contract.md),
[runner](../experiments/local_consolidation.py),
[scientific results](../reports/local-consolidation-v1-results.json),
[runtime](../reports/local-consolidation-v1-runtime.json),
[diagnostic audit](../reports/local-consolidation-v1-audit.json),
[audit runner](../experiments/local_consolidation_audit.py),
[audit runtime](../reports/local-consolidation-v1-audit-runtime.json).
