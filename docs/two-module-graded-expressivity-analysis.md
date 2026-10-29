# Two graded modules: analytical expressivity comparison

Date: 2026-10-29. Status: analytical note, not an experiment contract.
All mathematical claims below are **[derived]** unless otherwise tagged.

## Question and scope

Compare a simple two-module system with established linear networks, recurrent
networks, and equilibrium networks. Interpret *graded connections* as fixed
nonnegative strengths on corresponding response layers, optionally stronger
inward, as in [graded three-module v1](graded-three-module-v1-contract.md).
Layer labels need not mean successive feed-forward computation.

Expressivity means the set of input/output functions realizable by varying
allowed parameters. Distinguish it from parameter efficiency, robustness,
learnability, settling time, and the ability to maintain a state after input
withdrawal. Fix the observation mask, initialization rule and output readout.
An input-dependent mask, gate, stopping rule or nonlinear decoder can itself
add expressivity and must be counted.

## 1. A minimal reciprocal model

Let module states be a in R^n and b in R^m. Each includes an outer and an inner
layer. Local quadratic energies summarize internal linear reconstruction
relations and activity priors. Drive only A with input x in R^p, and read
y=Rb from B, with y in R^q:

```
E(a,b;x) = 1/2 a^T A0 a + 1/2 b^T B0 b - a^T Ux
         + 1/2 sum_l g_l ||P_l a - Q_l b||^2,
g_l >= 0; for inward grading, g_inner >= g_outer.
```

P_l and Q_l select and align layer coordinates into an r_l-dimensional
common interface. Assume A0 and B0 positive definite. Internal layer
interactions may be off-diagonal within these matrices. Define

```
A = A0 + sum_l g_l P_l^T P_l
B = B0 + sum_l g_l Q_l^T Q_l
C = sum_l g_l P_l^T Q_l
H = [[A, -C], [-C^T, B]].
```

H is positive definite: its quadratic form is the sum of positive local
forms and nonnegative compatibility penalties. Dynamics are dz/dt=-grad E,
where z=(a,b), or stable discrete gradient steps. This is a concrete
quadratic energy network, not a definition of every possible SRI module.
Hard evidence clamps produce the same analysis on the remaining free
variables whenever their Hessian is positive definite.

## 2. Exact equivalence to a linear feed-forward predictor

Stationarity gives Aa-Cb=Ux and Bb=C^T a. Eliminating a yields

```
S = B - C^T A^{-1} C > 0
b* = S^{-1} C^T A^{-1} Ux
y = Wx,    W = R S^{-1} C^T A^{-1} U.
```

Therefore every such two-module predictor is exactly one linear layer.
Constant local drives give an affine layer. There is no approximation or
large-width assumption here. Compiling W is the mandatory direct reference.
The converse, that this constrained energy family realizes *every* W, does
not follow: fixed alignments and positivity can make it a strict subset.

Finite-time recurrence does not evade the result. From fixed zero state,
T explicit gradient steps give

```
z_T = eta sum_{k=0}^{T-1} (I-eta H)^k [U;0] x.
```

This too is linear. Fixed nonzero initialization adds an offset. A singular
positive-semidefinite H requires separate solvability and initialization
conditions; it is not a source of automatic nonlinearity. Input-dependent
branch or stopping selection would be an additional mechanism.

An immediate strict separation from nonlinear feed-forward networks is
f(x)=|x| on [-1,1]. No affine function represents it. Any affine approximation
has uniform error at least 1/2: its values at -1 and 1 force its value at 0
to be at least 1-epsilon, whereas the target at 0 requires it at most epsilon.
The constant 1/2 attains that bound. A two-unit ReLU hidden layer represents
it exactly as ReLU(x)+ReLU(-x).

Thus even unlimited linear settling cannot reproduce this tiny nonlinear
network. A nonlinear readout could, but then that readout supplies the gain.

## 3. Communication rank: the real expressive restriction

With no direct x input to B or to the readout,

```
rank(W) <= rank(C) <= min(n,m, sum_{l:g_l>0} r_l).
```

The first inequality follows from the factorization above; the second from
rank subadditivity. Feedback changes the gain and mixes communicated
directions but cannot create new independent cross-module channels at
equilibrium. The same rank bound holds for transfer from a local drive to B
through any invertible linear local blocks having this sole cross-interface.

For an inner-only interface of width r, the correct established comparator
is a two-layer linear bottleneck network y=V Kx of width r. Every rank-at-most-r
linear map admits such a factorization (e.g. by SVD), so that comparator
contains the entire energy family; equality needs additional parameter
freedom and is not asserted for fixed P_l,Q_l.

An outer channel can enlarge the rank. For example, choose n=m=2,
A0=B0=I, outer selectors (1,0), inner selectors (0,1), and U=R=I.
Each coordinate independently has transfer gain

```
w_l = g_l/(1+2g_l).
```

Inner-only coupling has rank one. Any g_outer>0 and g_inner>0 gives rank two,
including strictly inward-graded choices. However increasing already positive
gains does not increase rank in this example. A very weak channel has nonzero
mathematical rank while remaining difficult to use in noise or finite precision.
Multiple interfaces need independent directions to attain the sum-of-widths
bound; their contributions can overlap or cancel with general alignments.

The bound concerns the equilibrium map. Across finite-time trajectories, a
low-rank channel can transmit different temporal combinations of the input;
one must not extend the equilibrium rank bound to arbitrary time readouts.

## 4. Grading is a constraint, not a new primitive

For an unrestricted neural message matrix W_l, replacing it by g_l W_l with
fixed g_l>0 does not change its possible values: absorb g_l into W_l. This
argument applies to feed-forward or recurrent message models with independently
free matrices. It fails when weights have norm bounds, are tied, or grading
also fixes self-penalties, as it does in the energy model above.

In the energy family, allowing arbitrary nonnegative profiles contains the
inward-graded family. Imposing an ordering cannot increase expressive power
relative to that unrestricted family. Comparing a fixed uniform profile to a
fixed graded profile need not yield an inclusion: each changes attenuation
and admissible transfer maps under the remaining constraints.

Grading can still matter substantially for conditioning, noise transmission,
regularization and learning. Those are different claims. For a single scalar
per module with local stiffnesses alpha,beta>0,

```
E = alpha*a^2/2 + beta*b^2/2 - x*a + g*(a-b)^2/2
b* = [g/(alpha*beta + g*(alpha+beta))] x.
```

Stronger coupling changes a scalar gain, approaching 1/(alpha+beta). It does
not create a nonlinear feature or arbitrarily large amplification.

## 5. Nonlinear modules: comparison with an ordinary RNN

A separate extension replaces the quadratic modules by block recurrent units:

```
a_{t+1} = sigma(Aa_t + C_g b_t + Ux + c)
b_{t+1} = sigma(Db_t + E_g a_t + Vx + d)
y_T = R b_T.
```

Here C_g,E_g contain the graded layer connections; this is not claimed to be
the gradient update of section 1. Concatenating states makes it exactly an
ordinary RNN with block matrix [[A,C_g],[E_g,D]]. Restrictions on grading,
reciprocity or sparsity restrict that RNN family. If the block matrices are
otherwise unrestricted and every gate is positive, scalar grading can again
be absorbed into weights.

For fixed T, unrolling produces an exactly equivalent T-stage feed-forward
computation graph with tied parameters and repeated input injection. An untied
network with the same graph can reproduce it by choosing equal parameters
across stages, and may realize additional functions. Recurrence offers reuse
of parameters and state; two module names do not establish extra power over
a matched RNN. Comparing one settling step with a T-step network would instead
confound computational depth.

If F_x(z) is globally contractive with constant rho<1, the unique fixed point
defines an equilibrium network. For a readout with Lipschitz constant L_R,

```
||y_T-y*|| <= L_R rho^T ||z_0-z*||.
```

For initial error bounded by M on the input domain, error epsilon is achieved
with T >= log(L_R*M/epsilon)/(-log rho), rounded up and bounded below by zero.
This gives a concrete finite feed-forward approximation, not an exact
finite-depth representation of every nonlinear equilibrium. Near rho=1 the
required depth can be large. Without contraction, existence, convergence and
branch selection need separate analysis; multiple attractors encode history
only when an initialization/continuation rule is specified.

## 6. Established architectures and prior work

Search date: 2026-10-29. Queries: `site.arxiv.org deep equilibrium models Bai
Kolter Koltun 2019`; `site.pnas.org Hopfield 1984 neurons graded response
collective computational properties`. A products-of-experts search was also
screened but is not used to support the expressivity conclusions.

- **Linear networks / Gaussian energy models [derived].** The exact affine
  equivalence and rank bounds above are direct algebra. For positive H,
  exp(-E) normalizes to a Gaussian over state, whose mean equals its energy
  minimizer. This is equivalence of inference; it says nothing about the
  project's learning rule being maximum likelihood.
- **Hopfield (1984) [read: abstract and model introduction].**
  [Primary paper](https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield84.pdf)
  studies graded neuron responses and associative memory. Its use of “graded”
  means continuous neuron activation, not stronger inward layer coupling.
  Energy-based recurrence is a useful architectural analogy; a strictly convex
  quadratic system has one equilibrium per drive and does not inherit
  multi-attractor memory merely by resembling a Hopfield network [derived].
- **Bai, Kolter and Koltun (2019) [abstract].**
  [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) defines inference
  by finding a nonlinear fixed point, with implicit differentiation for
  learning. This is the relevant established comparator for a convergent
  nonlinear two-module system. The block compilation and contraction bound
  here are our derivations, not claims that the paper proves this graded
  architecture stable. Its task results do not transfer to these modules.

Cheaper alternatives are the compiled affine map in the quadratic case,
a rank-r factorized linear predictor for an r-dimensional channel, and a
small nonlinear feed-forward network when only a static function is needed.
No novel architecture or superiority is established by this note.

## 7. Computation and limits

Let N=n+m, input/output widths p,q, and r=sum_l r_l. Dense assembly of the
coupling Gram matrices costs O(r N^2); a dense positive-definite factorization
costs O(N^3). Solving for p drive columns costs O(N^2 p); forming a dense
readout map costs O(q N p). Retaining H and U requires O(N^2+Np) scalars,
plus O(qN) for a general readout, with comparable factorization workspace.
After compilation, retaining only W costs qp scalars and each query costs
O(qp), plus any bias. A rank-k factorization instead stores k(p+q) scalars
and evaluates in O(k(p+q)), excluding the offline factorization cost.

Dense recurrence takes O(T(N^2+Np)+qN) operations per query as written;
precomputing the fixed input drive reduces this to O(Np+TN^2+qN).
Working activation storage is O(N). Sparsity can lower the matrix-vector
cost; generic dense bounds do not imply the sparse implementation is fast.
For quadratic gradient descent, 0<eta<2/lambda_max(H) is sufficient and
rho=max_i |1-eta*lambda_i(H)| controls convergence. Stability alone does
not imply fast settling. Nonlinear solver work must be counted to a common
output tolerance, including failures.

This is a proof-only analysis: no fits, candidate sweeps, numerical validation
scripts or inference benchmarks were executed. Measured compute and manual
selection time are unavailable, not zero; no empirical efficiency claim is
made. No historical artifacts or historical timing data are used to rank
architectures. Any subsequent numerical study must report measured selection,
construction, fitting and inference costs separately under
[computation reporting](computation-reporting.md).

## Conclusion

For fixed quadratic modules, graded coupling selects a constrained affine
predictor, with cross-module expressive capacity limited by interface rank.
Nonlinearity can remove the affine restriction; recurrence can reuse that
nonlinear computation over time. Neither grading nor the two-module partition
by itself exceeds the expressive family of an appropriately matched standard
network. A defensible advantage would need a resource or learning claim, or
an explicit restriction on the comparator, rather than module count alone.
