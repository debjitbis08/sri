# Compositional interactions and efficient edge operators

2026-10-27. Exploratory reading requested by the user; evidence tags follow
[prior work](prior-work.md). Connects to the candidate stateful edge algebra in
[continuous-learning notes](continuous-learning-notes.md).

## Sources and reading scope

- Atri Rudra, [Arithmetic Circuits, Structured Matrices and (not so) Deep
  Learning](https://arxiv.org/pdf/2206.12490v2), 2022, v2. Read the framing,
  butterfly/kaleidoscope definitions, Theorem 5.5 and its proof outline,
  and open questions (§§5–6).
- Jason Morton, [An Algebraic Perspective on Deep
  Learning](https://helper.ipam.ucla.edu/publications/gss2012/gss2012_10605.pdf),
  IPAM lectures, July 2012. Read selected slides on tensors, contraction,
  compositional diagrams and identifiability; not a full audit of the
  graphical-model and singular-learning material.
- Vincent Abbott and Gioele Zardini, [Weaves, Wires, and Morphisms:
  Formalizing and Implementing the Algebra of Deep
  Learning](https://arxiv.org/pdf/2604.07242), read the version resolving to
  v3, August 2026. Read framing, constructed terms, array-category/weave
  construction, compilation and future-work sections; implementations not run.

## Findings from the sources

**Rudra [read].** Butterfly factors implement mixing through fixed sparse
patterns. Kaleidoscope matrices compose butterfly matrices and transposes,
with optional expansion. For width w and expansion e, parameter storage and
matvec cost are O(wne log(ne)). Theorem 5.5 represents an n-by-n linear map
with circuit size s and depth d using O(ds log s) parameters and arithmetic
operations. The fixed patterns permit differentiable parameterization;
the depth overhead remains an explicit open question in this survey.

**Morton [read].** Tensor networks give mathematical meaning to diagrams:
wires carry spaces, boxes represent tensors, sequential composition connects
interfaces, and tensor products combine parallel components. Multilinear
operators support multiple interacting inputs. The later identifiability
discussion asks which parameter differences change observable behavior and
which reflect internal symmetries.

**Abbott–Zardini [read].** Their framework represents composition,
nonlinear operations, array axes and broadcasting together. Constructed terms
connect mathematical meaning to diagrams and code; weaves integrate axis
targeting and reindexing. The paper supplies Python/TypeScript tooling and
compositional PyTorch compilation. Automatic optimized low-level kernel
derivation is future work. Its architecture/execution distinction is useful
independently of the GPU-oriented motivation.

## Synthesis for SRI

**Design direction [derived].** Use compositional operators as the language
of the architecture, and structured circuits as candidate implementations of
its expensive translations. These address different questions: what an
interaction means, how interactions combine, and what they cost to execute.

An edge operator can be inserted as a box on a typed wire. Module aggregation,
compartments and multi-input interactions also need boxes. This accommodates
the edge-algebra idea without forcing every computation to be pairwise.

A useful explicit-state interface is:

    F_e : X_source × C_e × S_e → M_target × S_e
    S_e = (persistent parameters, activity gate, plasticity gate, trace)

Here C_e carries control/error information, and M_target is the outgoing
message. Sequential composition feeds compatible outputs to inputs;
parallel composition operates on disjoint state components. Shared state
requires explicit wiring or an update schedule. A closed loop carries state
from one step to the next, making the delay and update order visible.

This extends the previous edge sketch: a plain linear translation is one
primitive, gating another, plasticity another. Their composition need not
remain linear. A static tensor contraction loop and a temporal feedback loop
have different meanings; the latter gets an explicit state/delay interface.

**CPU direction [derived].** Consider a translation implemented by small
learned mixing stages rather than a dense matrix. A product of invertible
sparse factors remains full rank, so cheap structure need not mean a
low-dimensional bottleneck. Identity is a simple full-rank butterfly example.
Which structured family captures useful cross-module relations is the
application question. The circuit-size/depth hypothesis in Rudra's result
matters: an arbitrary dense map is not thereby a near-linear operation.

For tiny module states, dense matrices may still be cheaper in practice.
Compare dense, diagonal, low-rank and butterfly translations at the dimensions
the program actually uses. Include factor-gradient computation, inference
passes, state/trace storage and memory traffic, not only forward arithmetic.
If retaining a constant number of trace values per structured parameter,
trace storage scales with that parameterization; that assumption does not
cover a full recurrent sensitivity tensor.

**Learning direction [derived].** Let ongoing activity update selected local
parameters and transient state; dedicated phases can revisit more pathways
or optimize a larger subset of factors using internally generated activity.
The same typed operator interface can accommodate either schedule. Credit
signals and consolidation objectives come from the learning literature
already under review, while this algebra specifies their routes and effects.

**Next reading/design step.** Work through one complete module interaction:
translation, compartment aggregation, output inhibition, plasticity control,
and delayed feedback. Express it using established composition rules, then
compare candidate implementations and their costs. This keeps the mathematics
attached to the program's need: coordinated, continuously adaptable modules
on a CPU. No experiment or implementation was performed for these notes.

## Conditional gates with an error-reduction rule

2026-10-27. User direction: combine inference, fast adaptation and slow
learning through conditionally activated local gates. Error correction should
travel as ordinary activity through suitable pathways.

**Related mechanisms.** Predictive coding supplies local residual-based
inference and parameter updates ([Millidge et al., 2022,
§2](https://www.ijcai.org/proceedings/2022/0774.pdf), [read]). Continual
equilibrium propagation updates states and weights together during its nudged
phase ([Ernoult et al., 2020](https://arxiv.org/html/2005.04168v1), [abstract]).
The latter's use of “continual” refers to updates within that phase, distinct
from lifelong retention. These are relevant learning mechanisms to study
alongside the compositional descriptions above.

**A sufficient descent construction [derived].** Let q=(x,a,theta) collect
inference states, fast adaptive parameters and slow parameters. For fixed
external evidence and one differentiable objective E(q), define

    dq/dt = -D(q,t) grad E(q),
    D = diag(g_x, eta_a g_a, eta_theta g_theta), all entries >= 0.

Along this flow, dE/dt = -grad(E)^T D grad(E) <= 0. Conditional gates may
depend on state; no derivative of D is needed for this identity. Gates select
and scale descent components. They do not change the definition of E.
Zero gates permit stalls; strict decrease requires an enabled nonzero gradient.
Discrete execution requires suitable steps or an acceptance check.

Locality follows if E=sum_f E_f(q_N(f)): each variable's derivative sums all
incident factor contributions. A simple edge residual is r=x_j-W x_i,
with E_e=||r||^2/2. Its negative-gradient messages are -r at the target,
W^T r at the source, and r x_i^T for W. Thus correction can travel as network
activity, but it must carry the relevant dependence on the transformation.
An arbitrary replacement feedback matrix does not inherit this descent proof.

For an isolated block proposal, recompute all incident factors and accept only
if their sum does not increase; unchanged factors cancel. Disjoint proposals
can run together when they share no affected factor. Overlapping proposals
need coordination. Factor degree and recomputation cost determine whether
this local acceptance mechanism is affordable.

**Objective choice [derived].** Internal consistency alone admits unhelpful
agreement. Keep observed evidence fixed, include externally grounded
prediction/task terms, and distinguish decreasing the current energy from
improving a future unclamped prediction. Also keep update gates separate from
loss masks: turning off a residual's measurement must not count as fixing it.
Transmission gates that change predictions are model variables and need
their own update treatment. A task-aligned slow-learning rule can instead
use equilibrium-propagation credit or an explicitly derived predictive-coding
objective; the flow above only guarantees descent of its specified E.

**Program implication [derived].** Start with one local operator family whose
three update channels have different persistence and schedules. Cheap local
residual events can trigger inference and fast adaptation; repeated evidence
can propose slow updates. The gate policy controls when to spend computation,
while the descent rule controls update direction. A dedicated phase can
activate more channels and use internally generated activity. Count event
detection, correction propagation, iterations and acceptance checks in CPU
costs. This is a mathematical candidate, with no new experiment run.

## Local approximations to task gradients

2026-10-27. Targeted source lookup for local gradient approximations.

- **Feedback alignment [abstract].** Fixed random feedback weights convey
  error signals; forward weights adapt during learning. This relaxes matched
  forward/backward weights. Source: [Lillicrap et al.,
  2016](https://www.nature.com/articles/ncomms13276.pdf).
- **Eligibility propagation (e-prop) [read: mathematical basis and plasticity
  equations].** Combines a synapse-specific eligibility trace with a neuron
  learning signal. Online implementation approximates the ideal signal that
  includes future recurrent effects. Source: [Bellec et al.,
  2020](https://www.nature.com/articles/s41467-020-17236-y).
- **Difference target propagation [abstract].** Learned approximate inverse
  mappings send targets to layers, which train against their local targets;
  a difference correction handles imperfect reconstruction. Source:
  [Lee et al.](https://arxiv.org/abs/1412.7525). This is alternative credit
  assignment, not generally the identical parameter-gradient direction.
- **Forward gradients [abstract].** Forward-mode directional derivatives
  give unbiased gradient estimators without reverse-mode differentiation.
  [Baydin et al.](https://arxiv.org/abs/2202.08587).
  [Belouze](https://arxiv.org/abs/2209.06302) studies high-dimensional
  optimization difficulties. The directional derivative of the task loss
  still depends on the whole relevant computation [derived]; this is not
  strictly neighborhood-only learning.
- **Forward-Forward [abstract].** Positive and negative forward passes train
  layer-specific goodness objectives. [Hinton](https://arxiv.org/abs/2212.13345).
  These local surrogate objectives should be distinguished from approximating
  the gradient of one end-to-end task loss [derived].

Predictive coding and equilibrium propagation, reviewed above, are especially
relevant where iterative local dynamics can supply the corrective signal.
“Local” can mean synapse-local application with a transported learning
signal, layer-local optimization, or neighborhood-only information; record
which meaning is used. Eligibility traces require maintenance; iterative
inference requires repeated passes; inverse models require their own training.
These costs matter to CPU comparisons [derived].

**Gating approximate gradients [derived].** For exact task gradient g and
approximation h, a sufficiently small update -eta h decreases a smooth loss
when g^T h>0. With a nonnegative diagonal gate D the condition is g^T D h>0.
Ungated alignment does not guarantee gated alignment: g=(1,1), h=(2,-1)
has positive dot product, but D=diag(0,1) gives a negative one. Consequently,
arbitrary selective gates preserve the exact-gradient flow property above,
but need additional care when applied to approximate credit signals.
Evaluate the gated direction itself when studying this combination.

**Reading priority [derived].** E-prop for temporal credit and persistent local
traces; predictive coding for activity correction; feedback alignment as a
simple alternative feedback baseline. Treat their combination as a candidate
to derive, rather than assuming their guarantees compose. No implementation
or computation benchmark was performed for this lookup.
