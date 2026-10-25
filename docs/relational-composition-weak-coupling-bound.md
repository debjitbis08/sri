# A weak-coupling obstruction for a designated output module

Status: supplementary analytical derivation, 2026-10-25, after the finite
checks in [the main analysis](relational-capacity-and-composition-analysis.md).
No fitting or numerical experiment is needed for this extension. The finite
receipt for the main analysis does not purport to verify this additional bound.

## Assumptions

Use two circular modules with the existing rank-one terminal nullspace and
orthonormal radial maps in the output module. All its radial/terminal routes
are at least a_min=0.325. Only the input module receives evidence, with total
zero-state observation energy alpha=0.1 for the task's visible norm squared 2.
Both modules have the existing ring cycles with coefficient lambda=0.001.
Add a reciprocal complete-terminal identity coupling

```
mu/2 ||P x_input - P x_output||^2,
```

where P selects the eight terminal coordinates. Output is the actual outer
state of the output module. Score its parts 2 and 3 against the original
72-scene task. Inference minimizes the quadratic energy with the realized
routes fixed, as in the existing second pass. No auxiliary drive, output gain,
extra readout, or input-dependent port transport is included.

## Derivation

Write all nonradial interactions as C:

```
C = lambda * blockdiag(L,L) + mu * [-P,P]'[-P,P].
```

Both terms are PSD. The original cycle Laplacians have norm at most 4;
the complete-terminal difference operator has squared norm 2. Therefore

```
||C|| <= 4 lambda + 2 mu.
```

The minimizing state satisfies E(x)<=E(0)=alpha. Consequently

```
x' C x <= 2 alpha
||C x||^2 <= ||C|| x' C x <= 2 alpha (4 lambda + 2 mu).
```

Every output-sector stationarity equation is `K_j x_j = -(C x)_j`, since
the output module has no observation anchor. Project the two scored outer
sectors perpendicular to their fixed lifted rank-one directions. The original
anchored-chain argument bounds the combined force-to-projected-output operator
by sqrt(30)/a_min. Thus

```
||prediction_perpendicular|| <= C_mu
C_mu = sqrt(60 alpha (4 lambda + 2 mu)) / a_min.
```

For every relation/ratio group, averaging over its eight orientations gives
mean squared perpendicular target norm at least one. Some scene in every group
therefore has normalized hidden error at least

```
0.5 * max(0, 1-C_mu)^2.
```

The positive-part operation is essential when C_mu exceeds one. The proof
permits arbitrary input-dependent routes above the specified floor: the
lifted rank-one directions remain fixed, and the energy comparison uses the
same realized routes. No fixed-route linearity assumption is required.

In particular, the <=0.10-on-every-scene target is impossible whenever

```
mu < 0.5 * [a_min^2 (1-sqrt(0.2))^2 / (60 alpha) - 4 lambda].
```

The right side is positive for the declared alpha, lambda and a_min. At
mu=0 this recovers the previous conservative bound, though a disconnected
output can have an even stronger information limitation. At equality this
bound alone does not exclude the non-strict <=0.10 criterion. Above the
threshold it becomes inconclusive, not a capacity guarantee.

## Interpretation and extensions

The relevant quantity is the available nonradial force at the output, not the
number of modules in isolation. Stronger communication can invalidate the old
weak-force obstruction even without widening local nullspaces. Conversely,
adding weakly coupled modules is not automatically a solution.

For a graph of modules with identical orthonormal terminal ports, maximum
weighted module degree d_w, and the same total zero-state observation energy,
replace 2 mu by 2 d_w using the graph-Laplacian norm bound. This is a sufficient
conservative bound. If evidence is duplicated, total E(0) changes and must be
accounted for; that is not a free benefit from composition.

For general nonorthogonal transports, bound the spectral norm of the actual
coupling residual Hessian instead. New nonlinear interactions, relation-dependent
nullspaces, widened output retention, or a different output composition require
a fresh argument. This theorem is not a limitation of the general SRI program.
