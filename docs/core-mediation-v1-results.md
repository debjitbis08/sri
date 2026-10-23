# Core mediation v1, phase B — pairwise mediation needs an enumerating core

**Result:** within the declared bounds, pairwise compatibility through the core
realizes a joint-dependent target exactly when the core assigns a distinct mode
to every target configuration. The core then works as a lookup table of joint
configurations. Ternary coupling realizes every target that any energy can,
including targets that reuse core modes, such as a core in the parity (XOR) mode
of the two satellites. The agreement control meets Level 1 but no Level 2 target.

This is phase B only: mode-level realizability. Structured module internals,
recovery (phase C), input/output, and locality certificates (phase D) are not yet
run. The [contract](core-mediation-v1-contract.md) fixed classes, core sizes,
target levels, margin 1, the integer bound 16, and a branch limit of 4096.
The [certificate](../reports/core-mediation-v1.json) contains 312 orbit solves.
[Runtime data](../reports/core-mediation-v1-runtime.json) are separate.

## Outcomes

Counts are symmetry orbits of target sets, base variant (R1–R3). *Imp.* means
an exact certificate excludes real tables of any magnitude. *Bnd.* means no
integer witness exists in [−16,16], but the branch count exceeded the certificate
limit, so larger tables are not excluded.

| Class | Level 1, n=2 / 3 / 4 | Level 2, n=2 | Level 2, n=3 | Level 2, n=4 |
| --- | --- | --- | --- | --- |
| U, no coupling | 0 of 4 realizable; all imp. | 0/3, imp. | 0/6, imp. | 0/7, bnd. |
| D, direct A–B only | 0 of 4; all imp. | 0/3, imp. | 0/6, imp. | 0/7, bnd. |
| S, pairwise star | 2 of 4; others imp. | 0/3, imp. | 1/6; 5 imp. | 2/7; 5 bnd. |
| F, star plus direct | 2 of 4; others imp. | 1/3; 2 imp. | 3/6; 3 imp. | 4/7; 3 bnd. |
| T, ternary | 2 of 4; others imp. | 2/3; 1 imp. | 4/6; 2 imp. | 5/7; 2 bnd. |
| G, agreement | 5 of 9; others imp. | 0/7, all imp. | — | — |

The strict variant (R4, every non-target configuration unstable) realizes exactly
the same orbits. Its negative results are bounded only, because exclusivity creates
too many disjunctive branches to certify. Every realizable witness was re-verified
against all criteria by an evaluator independent of the solver encoding, and no
solve was unresolved. In every witness the stable set at coupling scale 64 equals
the stable set at scale 1, which equals the target set.

## What the pattern means

**Two targets that differ in one module are never both stable.** If x and y differ
by one move, then E(y)−E(x) ≥ 1 and E(x)−E(y) ≥ 1 cannot both hold. T can encode
any energy over joint configurations, so T's failures are intrinsic: they are
exactly the target sets with such adjacent pairs, plus the Level 1 sets whose two
targets share a core mode. Those fail R2, since clamping K at the shared mode
reproduces both targets.

**Level 1 does not discriminate.** S, F, T, and G all realize precisely the Level 1
targets with distinct core modes. As stated in the contract, consensus through the
core passes Level 1.

**Level 2 separates the classes.** For S, at every n, the realizable orbits are
exactly those with pairwise-distinct core modes. Negative results are certified for
n ≤ 3 and bounded for n = 4. So the pairwise star succeeds by the hidden-variable
construction anticipated in the contract and in no other way here. The core
needs at least as many modes as there are mediated joint configurations. The
parity target {(0,0,0), (0,1,1), (1,0,1), (1,1,0)} uses 2 core modes:

| Class | n=2 | n=3 | n=4 |
| --- | --- | --- | --- |
| S | imp. | imp. | bnd. |
| F | imp. | imp. | bnd. |
| T | realizable (14 entries, L1 norm 5) | realizable | realizable |

S and T store the same number of entries at equal n. Ternary coupling therefore
represents this relation with a 2-mode core, where pairwise coupling would need a
core that names all four configurations. Adding direct A–B coupling (F) lies
between the two: it realizes the 3-target set {(0,0,0),(0,1,1),(1,0,1)} at n=2,
which S cannot, but still not the parity target at n ≤ 3.

**Agreement is excluded at Level 2.** All 7 Level 2 orbits are certified impossible
for G. Joint dependence therefore separates consensus from compatibility in this
bounded class, as predicted. The contract's revision trigger did not fire.

## Answer to the question, scoped

- A core can mediate several stable, distinguishable joint configurations that are
  unstable without it (R1). It does so by participating, not as a fixed bias (R2),
  and without collapse under arbitrarily strong coupling (R3). This holds for S, F,
  and T.
- Pairwise interactions through a core are sufficient only when the core
  enumerates the mediated configurations. That is a storage and scaling cost,
  not a proof that higher-order coupling is required. With more satellites, the
  number of joint configurations can grow combinatorially, which bears directly on
  the sparse-representation property. Not tested here.
- Negative S results transfer to any pairwise coupling between a satellite and the
  core's internal units, because such coupling is still a function of (a,k).
  Positive results need not transfer: phase C must realize them internally.
- Recovery, structured internals, input/output, and locality are untested.
  Phase B gives them no credit.

## Limits

Two satellites with binary modes, core sizes 2–4, a margin of 1, and integer
entries bounded by 16. Bounded results at n=4 and in the strict variant do not
exclude larger tables. Symmetry reduction uses the group of mode relabelings and
the A–B swap; for G, only the swap and the simultaneous flip, since G fixes an
identity mode map. Certificates are verified in exact rationals; MILP is used only
to find candidates. Runtime was about 18 minutes, dominated by enumeration of
certificate branches.

Reproduce with `.venv/bin/python -m experiments.core_mediation` and
`.venv/bin/python -m unittest discover -s tests -v`.
