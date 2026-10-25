# Outer-state anytime readout and continuous local learning v2

Status: frozen before execution, 2026-10-25. The initial inner-readout draft was
corrected at the user's instruction BEFORE implementation or execution: outputs
must be outer activations; inner readings are diagnostics only. No results exist
for that superseded draft. A domain-specific revision following
the user's request for a simpler decoder and readout before equilibrium. It
changes the representation as well as the decoder; it is not an isolated decoder
ablation of v1. Preserve all earlier artifacts and their negative results.

## 1. Inner geometry is diagnostic; the outer layer is the output

Four binary outer units x_i form a ring. Let d_i be local neighbor disagreement.
With XOR at each local interaction,

```
d_i = x_i XOR x_(i+1)
q = mean_i d_i
p = d_0 XOR d_2 = XOR_i x_i.
```

The inner coordinates are r=(q,p), giving these exact class prototypes:

| Class | q | p |
| --- | --- | --- |
| Uniform | 0 | 0 |
| One different | 0.5 | 1 |
| Adjacent pairs | 0.5 | 0 |
| Alternating | 1 | 0 |

All 16 outer patterns map to the correct prototype. Minimum inter-prototype
distance is 0.5. Thus an inner Euclidean error strictly below 0.25 guarantees
nearest-prototype classification. These coordinates are supplied from domain
analysis, not discovered features or a claim of generic representational learning.
They place the nonlinearity inside the interacting units, not in the readout.

For internal diagnostics only, compare the code to classes using FIXED scores

```
s_c(r) = 2*prototype_c dot r - ||prototype_c||^2.
```

This equals nearest-prototype selection after dropping the common -||r||^2.
It is not the system's output and its accuracy cannot count as task success.
There are no learned readout parameters or extra hidden decoder layer.

The actual readout is the identity map on outer activities x, with optional
coordinate-wise thresholding at 0.5. It distinguishes all 16 binary patterns.
Their minimum Euclidean separation is 1; a coordinate-wise error strictly below
0.5 from a desired binary pattern guarantees correct thresholding. No inner
label or learned decoder substitutes for these outer values. Observed coordinates
remain hard-clamped; success is assessed on the missing outer coordinates.
Emit outputs at every declared horizon irrespective of residual or ambiguity.

## 2. Local graded operator and shared energy

Learn four stored truth-table values theta_ab in [0,1], in order 00,01,10,11.
Their bilinear interpolation is

```
g_theta(a,b) = theta00*(1-a)*(1-b) + theta01*(1-a)*b
             + theta10*a*(1-b) + theta11*a*b.
```

The exact XOR operator is theta=(0,1,1,0), so g=a+b-2ab. The SAME theta is used
at outer-neighbor and inner-parity interactions. The continuous circuit state
is x (4), d (4), r=(q,p) (2), all in [0,1]. Define

```
E = 0.5 sum_i (d_i-g_theta(x_i,x_(i+1)))^2
  + 0.5 (q-mean d)^2 + 0.5 (p-g_theta(d0,d2))^2
  + kappa/2 ||r-target_code||^2.
```

kappa=1 only when an external class label is supplied; otherwise kappa=0 and no
target enters the objective. r is not clamped. Training labels provide the known
prototype code. At a free readout, only observed x is supplied, not the correct
class. Completion probes explicitly condition on a supplied class; their inner
classification must not be counted as recognition.

Projected simultaneous gradient steps use state rate 0.1 and parameter rate
0.005, both from the old state. No weight decay. Backtrack a shared multiplier
by halves up to 12 times if energy increases by more than 1e-12. Count all trials.
The line search coordinates the numerical update globally; derivatives themselves
are local-factor derivatives, with shared-parameter contributions summed. Thus
this is not an entirely decentralized physical implementation.

Stored parameters are four float64 values (32 bytes), activity ten values
(80 bytes), and four fixed two-coordinate prototypes eight values (64 bytes).
No optimizer state or replay. A single fixed scalar output could not represent
all this state for free. Full-input frozen-theta direct inference is
d=g(x), r=(mean d,g(d0,d2)), an exact zero-residual state with kappa=0.

## 3. Arms, training, and data access

Three configurations:

1. Supplied XOR: no training, provides a constructive capacity/readout control.
2. Joint continuous: learn theta while d,r adapt on every step.
3. Direct gradient: same theta/function, projected gradient on squared error
   between direct r and the supplied prototype, with rate 0.005 and backtracking.

For the learned arms use root seed 2026102512, replicas 0,1,2. Initialize theta
from Uniform(0.35,0.65) using SeedSequence([root,replica,0]); initial d and r are
0.5. Pair the initial arrays across arms. Use SeedSequence([root,replica,1]) for
the same 200 permutations of the original eight training patterns as in v1.
Each presentation lasts 20 updates: block A has 32,000 updates. Block B presents
0010 and 0110 alternately for 100 presentations, 20 updates each: 2,000 updates.
States carry across presentations; parameters adapt throughout both blocks.
There are no settling requirements or separate update phases in the main stream.

Six learned trajectories, 204,000 update attempts, 10,200 presentations of ten
distinct labeled patterns in total. No search over rates/initializations, no
best-seed selection, no accuracy stopping, and no extension after a negative.
Initial/post-A/post-B direct INNER DIAGNOSTICS are recorded on all 16 patterns, with
original training, A-held-out and still-unseen-after-B subsets separated. The
fixtures and class definitions are known, not fresh confirmation data.
For arbitrary learned tables, the parity composition need not be rotation- or
complement-invariant; those properties are exact at the supplied XOR solution,
not automatically guaranteed throughout learning.

## 4. Anytime outer completion: the actual task

At 13 configurations (supplied once, 6 learned x 2 checkpoints), use the original
48-query battery (8 labeled
patterns x 6 pairs of observed coordinates), frozen/adapting probe copies.
Initialize unknown x from default 0011 and overwrite observations. Initialize d,r
by direct prediction at that x. Supply the true class as explicit context with
kappa=1; do not treat inner recognition of this supplied class as a result.
Add 0.05/2*||x-default||^2 + 0.05*sum x_i^2(1-x_i)^2 to the energy.

Read outer x at steps {0,1,2,4,8,16,32,64}, threshold at 0.5, and score class-valid
completion and membership in the nearest-default semantic completion set. Record
continuous hidden error, reversals, residual and parameter drift. Record first
correct readout and correct readouts whose residual is still above 1e-5. Residual
does not gate acting or define task success. Soft/mixed outer activations are
retained in the report rather than treated as invalid outputs. The experiment
reports choices but does not connect them to an external actuator. Semantic oracle
enumerates the 16 binary patterns; it is not a global optimizer of the continuous
energy. Default-only behavior is included at time 0. Observed x is hard-clamped.

13*2*48*64=79,872 completion updates, all charged. Correct class context is
explicitly supplied, as in v1; this is conditional reconstruction, not inference
of the class from two visible bits. Inner diagnostics during these queries are
context-driven and must not count as recognition. Queries/epochs are not
independent empirical examples. Full-input inner diagnostics use direct evaluation
only and do not replace this outer-output task.

## 5. Checks, cost and reporting

Before training, verify exact XOR codes and positive correct score margin for
all 16 patterns. Finite-difference every state/parameter derivative of joint
energy with label/default terms and every direct theta derivative, using central
step 1e-6 and absolute tolerance 2e-7. Check clamping and one descent step.
Scientific failures such as low accuracy do not abort the run; numerical
nonfinite values or exhausted line searches are correctness stops.

Each local factor/gradient or direct prediction costs O(n) for the n-edge circuit
and fixed four-entry theta; activity storage is O(n), parameter/readout storage
O(1). This grouping/prototype definition is proved only for n=4; the complexity
order is not an assertion of task generalization. T updates with J line-search
trials cost O(T*(J+1)*n), not one forward pass. No inference-iteration tolerance
controls acting: wall/CPU and actual work are reported at the declared budgets.

Time correctness, each training block, inner diagnostics, completion curves,
and serialization separately. Additionally benchmark supplied-XOR outer completion
at fixed budgets 1,8,64 on the 48 queries with frozen theta, and the direct
16-pattern semantic completion oracle:
one warmup batch, three measured batches; report batch median/min/max and
amortized per-query time with accuracy at those budgets. Include allocation,
gradients and line search; no accelerator/compilation. The oracle uses the supplied
class definition, not learned parameters or continuous energy, and is the simpler
task comparator. These are distinct quality–cost points, not equal-accuracy speed
comparisons. The cost of simply reading an already available x is O(n); report
the preceding inference work, not merely that final read operation.
Each timed numerical query also computes its semantic oracle for scoring; this
validation overhead is included in every numerical budget and disclosed.

Use float64 NumPy, one process, one BLAS/OMP thread, no workers/accelerator.
Record source/contract hashes, revision/dirty state, seed policy, command,
environment, phase wall/CPU, actual forward/gradient/update/trial/readout counts,
peak RSS with units, retained arrays and unavailable allocator workspace. Keep
deterministic results and runtime JSON separate. Raw fixtures are generated by
the driver, not retained as inference replay. Manual development is unmeasured.

Implement `experiments/outer_anytime.py` and run once with
`.venv/bin/python -m experiments.outer_anytime`. Do not overwrite reports or
modify the contract after seeing scores. This study tests a supplied sufficient
inner geometry with an identity/threshold outer readout, continuous local operator learning, and
early decisions. It does not demonstrate general mode discovery or autonomous
real-world action.
