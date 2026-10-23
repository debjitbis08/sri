# Worked example: collective role binding

**Status:** implemented illustrative mechanism, not an OP-04 architecture or
evidence of intelligence. This is the worked example requested after the
[semantic evaluation](collective-interaction-evaluation.md). It does not adopt
constraint satisfaction as the foundation of SRI.

## Structures and joint interaction

A satellite's persistent specification is `(name, local_role, allowed_sites,
direct_site)`. Its transient configuration retains that scoped role and binds
it to a site, or is unbound. Equal local role labels in different satellites
do not identify their roles. Sites have fixed shared identities; the output
records actual bindings, not renamed labels.

The input module contains immutable evidence requesting A and B. A separate
output module observes the binding relation. An internal context intervention
can additionally activate S; this is not another external input channel or a
learned activation rule. All satellite structures remain present in both cases.

The core configuration is a finite relation of `(satellite, role, site)` tuples.
A valid mediated response jointly requires: one allowed site per participating
role, distinct sites for distinct participants, exact agreement between core,
satellites and output, and unbound inactive satellites. External evidence stays
unchanged. The constructor commits one immutable joint configuration. Intermediate
backtracking assignments are solver workspace, not module states or processing
stages. This example does not introduce layers.

## Matched control and fixed results

Without mediation, satellites must use their declared direct sites, exclusivity
still applies, and the core is empty. With mediation, satellites may bind to any
site in their unchanged allowed domains, with the core certifying agreement.
This changes the applicable binding condition; it does not merely add constraints
to an unchanged direct-binding relation. No site or satellite capacity is added.

This permission is hand-designed. A generic joint matching solver can reproduce
the bindings without a distinct core; the independent test oracle does so for
small cases. Here the core is a consistency certificate, not an autonomous
dynamical mediator whose necessity has been established.

For A and B, give each role sites `{a,b}` and direct site `a`. Direct binding
collides, while mediation yields `A:a, B:b`. Adding S with the same domain remains
impossible: three roles cannot occupy two sites injectively. Every pair fits,
but the triple does not. This is a global consequence of shared-site constraints,
not proof of an irreducible ternary primitive.

For context, A and B allow `{a,b,c}` and S allows only `{a}`. Activating S from
the established `A:a, B:b` configuration produces `A:b, B:c, S:a`, without changing
evidence or structures. Both existing participants change in the joint response.

| Case | Outcome | Candidate extensions / collision rejections |
| --- | --- | --- |
| Direct A+B | Infeasible | 2 / 1 |
| Mediated A+B | `A:a, B:b` | 3 / 1 |
| Three roles, two sites | Infeasible | 10 / 6 |
| S inactive, three sites | `A:a, B:b` | 3 / 1 |
| Activate S from that response | `A:b, B:c, S:a` | 12 / 5 |
| Zero construction budget | Unknown; no response committed | 0 / 0 |

Selection is lexicographic by participant and site label, not an intrinsic
preference or graded coupling law. Renaming preserves feasibility but can change
the selected representative. There is no schedule-independent selection claim.

## Scoped guarantees and resource account

The constructor searches allowed site choices depth-first, keeping only a partial
binding and occupied-site set. It checks whole-configuration agreement before
returning a result. It does not construct a global configuration table.

**Soundness:** each committed role has an allowed site, all sites are distinct,
and core/output/satellite projections agree. The direct mode additionally fixes
each binding to its direct site.

**Completeness without a cap:** every site choice at each depth is visited unless
already occupied. Such a collision cannot be repaired by extending that partial
binding. Finite search therefore finds the first lexicographic valid binding or
exhausts the candidates. Empty participation admits an empty binding.

**Selectivity:** three distinct roles cannot occupy two sites injectively. The
retained incompatibility follows from a common invariant, not a blacklist.

**Budget semantics:** a candidate-extension cap yields `truncated` and feasibility
`null` if exhausted before a decision. Exhaustive failure yields `complete` and
feasibility `false`. Neither case commits a partial response or mutates the old
configuration. The old response must not be presented as satisfying a new context
after a failed or truncated reconstruction.

With `n` participants and at most `q` sites per domain, candidate extensions are
bounded by `sum(q**k for k in 1..n)` before pruning. Workspace includes up to `n`
chosen bindings and per-depth sorted domains, up to `O(n*q)` references beyond
input/output storage. Sorting, validation, membership, strings, and snapshot
construction add work: extension counters are not total CPU instructions. Python
recursion limits and exponential worst-case search restrict this implementation
to tiny illustrations; no scalability or bounded-latency claim follows.

A role/direct-site-only interface is insufficient: identical such descriptors
can hide whether a second site is allowed, changing joint feasibility. Tests
retain this counterexample. Providing the domain suffices for this example's
binding semantics, not for general structured-module interaction.

## Reproduction and limits

```
python -m experiments.mediation
python -m unittest discover -s tests -v
```

The runner writes only [mediation-example.json](../reports/mediation-example.json),
with six cases, structures, before/after configurations, work counters, and Python
metadata. Generation is byte-identical on the same interpreter. Cross-version
tests compare semantic contents while preserving frozen runtime provenance.
Top-level `complete` means the six declared cases were evaluated; the deliberately
capped case retains its own `truncated` status.

Tests compare the selector with a product oracle for 49 small domain pairs in
both modes. That enumeration exists only in tests, not in the constructor.
They also check context changes, retained incompatibility, overlap disagreement,
lossy descriptors, input preservation, caps, and frozen-report regeneration.

The example establishes an executable joint transformation relative to an explicit
control. It does not establish post-input resonance, disturbance recovery, graded
layer coupling, topology benefits, learning, abstraction, or an essential core.
The next modeling obligation is to identify what the core contributes beyond a
recorded joint solution, and how that contribution arises from reusable structure.
Do not scale this allocator and call it SRI. OP-02 through OP-03B remain unchanged.
