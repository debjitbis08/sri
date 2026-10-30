# Revised limited memory: decaying reader parameters

2026-10-30. Review of proposal revision `217701a`. This supersedes the applicability
of the external-store sections of [the earlier review](fan-in-memory-analysis.md)
and [theory](fan-in-memory-theory.md). The earlier results remain valid under
their stated assumptions. The new proposal changes the intervention, not those
identities. No training run or contract is frozen here.

## What changed, and what must be corrected

The proposal now explicitly decays learned reader parameters and their outgoing
connections. FIFO horizons, retrieval-label leakage, fresh-write provenance and
uniform-FIFO controls are no longer criticisms of this design. External rehearsal
is held fixed during acquisition; the form-free delay is now unambiguous. Weight
decay supplies an actual force that was absent from the original memory model.

Two claims about the baseline are incorrect, as verified in
`experiments/fan_in_pilot.py:reader_mask`, `fix_rows` and `make_model`:

1. **Fixed75 did not have 16 dense units.** All 64 units had one code connection
   per operand. Forty-eight were moved onto a common coordinate; sixteen kept
   their random sparse masks. The proposed 16 fully dense units constitute a
   new, larger bypass. Call the new no-decay arm a new mixed baseline, not a
   reproduction of fixed75. Broad gradient access could help as well as hinder
   learning, so its behavior should not be assumed from the old result.
2. **H→0 does not give the 64-unit sparse arm.** With finite bounded optimizer
   updates and evaluation after decay, it silences the memory columns, leaving
   48 active first-layer structure units. A matched `sparse48` control is the
   right limit. Earlier training trajectories can still differ for finite H.

The observations also support "dense fits taught questions better; constrained
readers transfer on some measures", not a categorical separation of dense
memorization from sparse concepts. A scalar reader can still store arbitrary
finite identities. Fixed wiring did not reproduce RigL's English transfer.

## 1. Exact forgetting filter, conditional on optimizer updates

Write a decaying parameter as w=w0+u, let Δ_t be its optimizer increment, and
r=2^(−1/H). Applying decay after the update gives

    u_(t+1)=r(u_t+Δ_t),
    u_T=r^T u_0 + Σ_(t=0)^(T−1) r^(T−t) Δ_t.                 (1)

Outgoing columns obey the same equation with w0=0. This unrolling is exact even
for Adam if Δ_t denotes its actual increment. It does not assert that increments
are independent of the state. It establishes exponential weighting of parameter
increments, not an exact half-life for an association or an output.

If Δ_t=d is constant, u approaches rd/(1−r). Thus ongoing context gradients and
Adam momentum can renew the pathway even without form examples. Unreset moments
are additional learned state. A no-form delay is not a zero-update memory test.

For a frozen-input, no-optimizer-update control, r^2000 is .5, .0625 and
1/65536 at H=2000, 500 and 125 respectively. These are coefficient attenuation
factors. They are not predicted matching accuracies.

### Relation to regularization

For ordinary SGD, u' = r(u−η grad L) equals an SGD step of size rη on

    L + [(1−r)/(2rη)] ||u||².

So "a forgetting rate rather than a cost" is not an absolute distinction. For
Adam this simple scalar L2 equivalence does not hold; decoupled parameter decay
and a penalty inside the adaptive gradient are different updates. Here incoming
weights shrink toward random initialization, whereas L2-SP commonly uses a
pretrained reference. These differences matter for interpretation.

## 2. A bound on functional forgetting—and its missing assumptions

Let memory features be h_m(x)=ReLU((W0+U)x+b0+u_b), and let V_m be their
outgoing columns. Include the relation-token coordinates in x for g when those
weights also decay. On a fixed bounded input set, write z0=W0x+b0 and
Δz=Ux+u_b. Under pure decay for t steps,

    memory preactivation contribution = r^t V_m ReLU(z0+r^t Δz).

Since ReLU is 1-Lipschitz,

    ||contribution|| ≤ r^t ||V_m|| (||ReLU(z0)|| + r^t||Δz||).  (2)

If the downstream map has a fixed Lipschitz bound L, multiply (2) by L to
bound the output difference from silencing memory. Hence outgoing decay does
ensure vanishing functional influence under bounded features and downstream gain,
even when incoming features return to a nonzero random baseline. Learned incoming
deviations can contribute an r^(2t) term, but the random-feature term decays as r^t.

**The actual proposal trains the encoders and later layers during the delay.**
Neither bounded features nor a fixed downstream gain follows from the protocol.
This is a conditional result, not an unconditional output-forgetting guarantee.

### Exact rescaling loophole

For the actual shared second layer, write

    f(x)=W3 ReLU(V_s h_s(x)+V_m h_m(x)+b2)+b3.

For any c>0, replace (V_s,V_m,b2) by c(V_s,V_m,b2) and W3 by W3/c. The output
is unchanged by ReLU's positive homogeneity, while ||V_m|| shrinks by c.
All compensating parameters here are non-decaying except V_m. In particular,
with U=0, random memory features can carry a nonzero function while the norm of
the decaying columns is made arbitrarily small. This is an expressivity and
regularization loophole, not proof that Adam will exploit it.

**Use:** log downstream norms and actual score changes under memory ablation;
small memory-column norms alone do not demonstrate forgetting. A cheap additional
diagnostic is a decay-only copy of a checkpoint with the rest frozen. If a stricter
intervention is later needed, bound downstream gain or penalize functional memory
contribution; freezing all later layers changes learning and is not a neutral fix.

## 3. When decay can cause a handoff to structure

An exactly solvable local model provides a positive result. Let a slow scalar s
and fast scalar m jointly predict y through fixed sensitivities a and b:

    prediction = a s + b m,
    L=1/2(a s+b m−y)²,
    ds/dt=−η a(a s+b m−y),
    dm/dt=−η b(a s+b m−y)−κm,   κ=ln(2)/H.                   (3)

Assume a≠0, η>0, κ>0 and continued presentation of this target. The unique
equilibrium is s=y/a, m=0: **decay shifts the entire representable target into
the slow pathway without explicit distillation.** The residual created as the
fast contribution shrinks trains s. This is a conditional mechanism for the
researchers' idea, not merely an analogy.

Proof: stationarity of s implies zero residual; stationarity of m then implies
m=0. The error dynamics have symmetric matrix

    M = [[ηa², ηab], [ηab, ηb²+κ]].

Its trace is T=η(a²+b²)+κ and determinant ηa²κ>0; both eigenvalues are positive:

    λ_± = (T ± sqrt(T²−4ηa²κ))/2.                            (4)

For κ much smaller than η(a²+b²),

    λ_slow ≈ κ a²/(a²+b²),
    handoff half-life ≈ H (1+b²/a²).                        (5)

The half-life of *consolidation* can therefore greatly exceed the parameter
half-life. For b=4a, it is approximately 17H in this regime. H=500 can correspond
to roughly 8,500 updates, longer than acquisition. These sensitivities are toy
parameters, not fitted neural measurements.

For κ=0, starting from zero, ordinary gradient flow instead converges to
(s,m)=y(a,b)/(a²+b²); the fast pathway can permanently carry most of the target.
If a=0, the slow pathway cannot learn the target at all, and decay creates error
rather than consolidation. No choice of H cures that representational failure.

### Exact discrete update and scope

For SGD then decay, let v=(a,b), D=diag(1,r):

    z_(t+1)=D[(I−ηvv^T)z_t+ηvy].                            (6)

The same equilibrium (y/a,0) holds for r<1, a≠0. Convergence additionally needs
spectral radius of D(I−ηvv^T) below one. Equation (3) is a small-step continuous
model, not an exact rewriting of Adam or of (6).

Most importantly, (3) requires the original target to remain in training.
During the no-form delay, old English residuals are absent. Decay need not cause
old English mappings to migrate into structure; it can simply remove them.
The relevant consolidation window is acquisition plus permitted rehearsal,
before old evidence disappears. Context can maintain genuinely shared components.

**Use:** measure structure-only performance before forms disappear, not just
after memory has been silenced. Interpret a negative short run against the
handoff timescale rather than concluding no consolidation is possible.

## 4. Readout splitting and relearning are not guaranteed

The ablations measure causal dependence under intervention, not disjoint stores
of knowledge. Shared nonlinear layers generally give

    f(s,m) ≠ f(s,0)+f(0,m)−f(0,0).

For example f(s,m)=ReLU(s+m−1), s=m=.75, produces .5 jointly and zero from
either part alone. Add the baseline f(0,0) and report the interaction residual
alongside accuracy and score margins. This prevents synergy from being read as
"neither pathway learned". Later-layer biases can answer even with both silenced.

Returning incoming weights to random initial values does not guarantee revival.
If a unit is inactive on all relevant inputs, both its ReLU derivative and its
activation are zero. If V_m=0, incoming gradients are zero initially even for
active units, though the outgoing columns can begin learning if upstream error
and activation are nonzero. Random initialization offers opportunities, not a
proof of recoverability. Track activity, outgoing-gradient magnitude and time
to reactivation, and retain optimizer-state accounting.

Relearning speed also needs a learning curve relative to its own starting
accuracy or a common target. Comparing only to end-of-block-1 accuracy mixes
retention with reacquisition. A structured quantity coordinate cannot recreate
an arbitrary lost glyph binding without new evidence.

## 5. Stage 2 has a precise representability limit

Suppose a frozen teacher T(x) is distilled on synthetic input distribution μ
into a structure-only student F(q(x)) using squared error. The optimal unlimited
student is

    F*(q)=E_μ[T(x)|q],
    minimum error=E_μ[Var(T(x)|q)].                           (7)

Proof: condition on q and decompose squared error into conditional variance
plus squared error from the conditional mean. For two-item readers q includes
both scalars and the relation token when applicable; vector targets use summed
coordinate variances.

Any teacher dependence on information discarded by the sparse student is
irreducible under that replay distribution. Random codes can train off-manifold
behavior and need not exercise learned glyph representations at all. Distilling
the reader alone does not directly train the image encoder's old mappings.
Because the shared layers are nonlinear, "the memory contribution" needs an
explicit definition, such as a frozen teacher's full-minus-ablated output;
that difference is contextual and not a separately generative memory system.

This is synthetic-input distillation/pseudorehearsal unless a generator models
the learned data distribution; calling it generative replay alone obscures the
coverage assumption. Stage 2 remains deferred.

## Specific changes to make before freezing the proposal

- Correct fixed75's wiring and distinguish the new dense mixed baseline.
- Include or explicitly justify omitting the matched 48-unit sparse control.
- State whether g's eight relation-token input weights decay; its first layer
  has 40 inputs, not 32. Structure-only RigL updates must exclude dense rows:
  the current helper assumes the same fan-in in every row.
- Retain the original rewiring stop at update 4,500 if intended; extending the
  curriculum must not silently rescale it into the delay/relearning period.
- Log effective output dependence and later-layer gain, in addition to weight
  displacement. Evaluate consolidation before the delay, and distinguish a
  decay-only checkpoint diagnostic from trained delay behavior.
- Treat the half-life grid as three declared candidates. It is not adaptive
  tuning, but any claim based on selecting a successful H should account for
  all three and be confirmed independently. Inconclusive is not equivalent.
- Specify initialization and scale of the new dense rows, and identical initial
  weights across finite-H and no-decay mixed arms.

With all g inputs included, decay touches 1,680 parameters in g and 1,552 in P,
3,232 total. Only the 1,184 incoming weights/biases need saved initialization
if outgoing targets are zero: 4,736 extra bytes at float32, excluding optimizer
state. The shared outgoing columns must be classified explicitly despite the
sentence saying later layers do not decay. Runtime is O(3,232) per update for
decay, but evaluation and all six/seven arms need measured accounting.

## Updated verdict

The revision now implements a genuine selective forgetting mechanism. Equations
(3)–(5) show a concrete way it can drive consolidation while teaching continues.
However, consolidation is conditional on slow-path capacity, usable gradients,
sufficient exposure time, and bounded compensation through shared layers.
Neither monotonic transfer improvement with shorter H nor a middle H achieving
both outcomes follows in general. The study should distinguish handoff, erasure
and gain compensation rather than treating any weight shrinkage as handoff.

## Prior work and reproducibility

Read/search date 2026-10-30. Queries: "Hinton Plaut 1987 using fast weights to
deblur old memories"; "L2 SP explicit inductive bias transfer learning initial
weights Li 2018"; "decoupled weight decay regularization Adam Loshchilov Hutter
2017". Targeted component review, no novelty claim.

- [Hinton and Plaut 1987](https://www.cnbc.cmu.edu/~plaut/papers/abstracts/HintonPlaut87CogSciConf.fastWeights.html)
  **[abstract]**: slow weights plus fast decaying overlays, with rehearsal
  recovering blurred associations. Their result does not predict this particular
  sparse/dense split. The two-path handoff model above is **[derived]**.
- [Li et al. 2018](https://proceedings.mlr.press/v80/li18a.html)
  **[abstract]**: anchoring fine-tuning to pretrained parameters. Here the
  reference is random initialization and decay is decoupled from Adam.
- [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101)
  **[abstract]**: distinction between L2 regularization and decoupled decay
  under adaptive optimization. The shifted-SGD equivalence above is **[derived]**.
- Equation (7) is the conditional-expectation least-squares projection
  **[derived here]**, not a claim that random-code distillation has been tested.

Run `.venv/bin/python -m experiments.decaying_readers_theory`. It checks the
increment filter, SGD equivalence, ReLU rescaling, two-mode eigenvalue identities,
exact discrete recurrences, ablation interaction and conditional-projection
counterexample on fixed arrays. Deterministic output and separate timing/RSS/
environment/source hashes are in `reports/decaying-readers-theory*.json`.

Declared calculations: H∈{2000,500,125}, a=1, b=4, η=.01, y=1, 6,000 recurrence
steps per H. These illustrate the timescale, not selected neural hyperparameters.
No learned data, neural fits, random draws or tuning. With A half-lives, T steps
and two scalar modes, cost is O(AT), working storage O(1) per arm; 2×2 solves and
eigendecompositions are constant-size. General d-mode dense recurrences cost
O(ATd²), eigendecomposition/solves O(d³), storage O(d²). Checks of nonlinear
identities use fixed small matrices. No neural speed claim is made.

Verification: all 21 checks passed. In the declared fixture, H=500 gives an
exact continuous-flow handoff half-life of 8,565.27 updates. After 6,000 discrete
SGD-then-decay steps, slow and fast contributions are .4205 and .5748 to a target
of 1. Thus nearly perfect combined fit can coexist with incomplete handoff even
under the favorable assumption of uninterrupted target presentation. This is
an illustrative counterexample to identifying parameter half-life with
consolidation time, not a forecast for English or the selected half-life grid.
