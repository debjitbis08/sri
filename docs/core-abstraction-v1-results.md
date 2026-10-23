# Core abstraction v1 — no qualifying core for S1 or C2: preserving isolated waking behaviour conflicts with teacher transfer

**Completion:** `complete`. Every correctness gate passed. Stage A evaluated all
4 teachers × 5 splits × 2 assignments (40 runs, 124 surviving assemblies, 9
cores each). Stage B computed information classes over all 2,187 vectors for
all 10 split/assignment runs and scored every member on every assembly of its
class.

**Headline:**

- **Existence fails for both primary teachers.** No surviving assembly has a
  qualifying core in any of the 10 S1 runs or the 10 C2 runs. The declared
  existence result (at least one qualifying assembly per primary teacher) is
  not met, and there is no partial result either.
- **The obstruction is a conflict between the targets, not missing core
  capacity.** The teacher's own core is in the family and, on the true
  assembly, it fixes every mixed-evidence query: C2 gains 16 per satellite,
  S1 gains 8. But the connected teacher itself disagrees with the satellites'
  isolated waking marginals on 3–6 (S1) and 5–8 (C2) of the 15 training items
  per run. Any core that reproduces the teacher therefore fails *retention on
  assembly*. Every core that preserves retention transfers nothing. The
  contract anticipated this conflict (section 5) and required it to be
  reported rather than the target changed.
- **S1 usually fails earlier.** In 9 of 10 runs, joining S1's learned
  satellites at `c0=(0,0)` already loses 1–3 waking items (assembly
  interference).
- **The only qualification is recovery of a known core.** The specificity
  control S2 qualifies in 1 of 10 runs (split 4, swapped). This is the one run
  in which S2's coupled behaviour agrees with all 15 waking items. Its
  qualifying cores are `±(3,3)`, S2's own core, on its own private blocks.
- **On correct private blocks, nothing beyond teacher-core recovery qualifies
  anywhere in the space** (post-hoc tally, below). Across all 2,187 vectors and
  10 runs, every qualifying core on a member's true assembly is that member's
  own core sign class (984 of 984 entries).
- **Stage B is moot for S1/C2, but its structure is as predicted.** Every
  information class is exactly `V_A × V_B × (nine cores) × {r}`. The waking
  state carries no information about the core. Where some member qualifies, a
  single compatible core exists on 240 of 243 class-assemblies. All 3
  exceptions are pure core-relation ambiguity: same private blocks, one member
  needing `±(3,3)` and the other `±(3,-3)`.

[Contract](core-abstraction-v1-contract.md) (sha256 `fe8cf4ef…`) ·
[manifest](../reports/core-abstraction-v1-manifest.json) (written before
computation) · [ceiling](../reports/core-abstraction-v1-ceiling.json) (Stage A)
· [information](../reports/core-abstraction-v1-information.json) (Stage B) ·
[runtime](../reports/core-abstraction-v1-runtime.json).

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Prior-artifact hashes (splits, shared-input substrate and oracle, earlier manifests) and contract hash equal the manifest | yes |
| Every response set used by Stage A (12,088: 11,492 full, 404 input-frozen, 48 each NF and `k` clamped to −1, 0, 1) against the independent oracle | 0 mismatches |
| Core-sign symmetry `Y(c) = Y(−c)`, all 2,187 vectors × 81 queries | 0 mismatches |
| Analytic core potential `Φ_c(a,b)` equals enumeration over `k` (81 cases) | yes |
| Waking filters: identical final masks over the five frozen orders, teacher blocks survive, original-context retention for every surviving block, idempotent under one repeated pass (40 runs) | yes |
| Stage-B class structure `V_A × V_B × cores × {r}`, all 1,229 classes | yes |
| Byte-identical regeneration (manifest, ceiling, information) in a temporary directory, 4 workers instead of 10 | yes |
| Tests | 16 new; full suite 247, all passing |

**Provenance.** The contract and manifest were written and the manifest frozen
before any Stage-A or Stage-B quantity was computed. Neither was committed to
git before the scored run, so the order is attested by the manifest hash and
file times only. The code was written in the same session and is recorded by
hash in each report's `provenance`.

## Stage A — representability ceiling

### A.1 Acquisition

Filtering is exact and order-independent. Private masks are small:

| Teacher | Mask sizes `|V_A|×|V_B|` over 10 runs | Surviving assemblies |
| --- | --- | --- |
| S1 | 1×1 (9), 2×1 (1) | 11 |
| C2 | 3×3 (10) | 90 |
| S2 | 1×1 (9), 2×1 (1) | 11 |
| T1 | 1×1 (8), 2×1 (2) | 12 |

S1 and S2 have identical masks in every run, as the contract predicted, since
they share private blocks and `r`.

### A.2 Outcomes per assembly

| Teacher | Interference | No transfer core | No core-dependent witness | Route-unresolved only | Qualifying | Possible / robust runs |
| --- | --- | --- | --- | --- | --- | --- |
| **S1** | 9 | 2 | 0 | 0 | 0 | 0 / 0 |
| **C2** | 0 | 60 | 0 | 30 | 0 | 0 / 0 |
| S2 | 9 | 1 | 0 | 0 | 1 | 1 / 1 |
| T1 | 1 | 11 | 0 | 0 | 0 | 0 / 0 |

`no_preserving_core` (only `c0` preserves) applies to 9 S1, 9 S2 and 1 T1
assemblies. These are exactly the interfering ones, where `P(c0)` fails as well.
No fixed core qualifies in any S1, C2 or T1 run. For S2, `±(3,3)` qualifies in
the one run.

### A.3 The teacher's own core on the true assembly

| Teacher | `c0` waking items lost (interference) | Teacher core: mixed net gain (A, B) | Teacher core: waking items lost | Preserving cores give transfer? |
| --- | --- | --- | --- | --- |
| S1 | 1–3 in 9 runs; 0 in split 4 swapped | (8, 8) | 3–6 | — / no (split 4 swapped) |
| C2 | 0 | (16, 16) | 5–8 | no |
| S2 | as S1 | (8, 0) | 0–3; 0 in split 4 swapped | yes, in that run |
| T1 | 0 | (8, 8) | 1–5 | no |

The teacher core makes every mixed marginal exact (C2: 36/36 each; the
assembled `c0` gets 20/36). It fails only because the connected teacher's
training marginals differ from its satellites' isolated ones. On the true
assembly, cores that keep the waking marginals (seven of nine for C2: all
except `±(3,−3)`) leave every mixed query as at `c0`.

### A.4 C2's route-unresolved cores (wrong private blocks)

On 3 of C2's 9 assemblies per run — `(0,−3)×(3,−3)`, `(0,−3)×(3,0)` and
`(3,−3)×(3,0)`, none of them the true `(3,0)×(0,−3)` — the cores `±(3,3)`
preserve every training item and gain 2 mixed queries for one or both
recipients, with no mixed or within-query regressions. Both single-link cuts
remove each gain, so the gains are core-dependent. None survives the route
test: with inputs clamped to evidence, the learned and initial cores give the
same recipient marginal. Diagnostics on these witnesses:

- the contrast persists under NF;
- clamping `k=0` removes it; one sign of `k` restores it;
- ablating the *partner's* input block removes the gain, and ablating the
  recipient's own block does not.

The improvement therefore runs from the partner's private structure, through
the transient core, into input revision. This is core-dependent but
route-unresolved transfer, and it occurs only on assemblies the waking data
cannot exclude but that are not C2's blocks. Their core `±(3,3)` is not C2's
`±(3,−3)`.

### A.5 Controls

No consolidation and core removal equal `c0`. Ordinary repetition left every
mask unchanged (verified in all 40 runs). Memorization stores 15 waking entries
per run and answers none of the assembled queries. Fixed-activity, NF, input-
frozen and ablation diagnostics are archived per witness in the ceiling report.

## Stage B — information ceiling

| Quantity (summed over 10 runs) | Value |
| --- | --- |
| Information classes | 1,229 (104–131 per run) |
| Classes with any member having a qualifying core | 161 |
| (class, assembly) pairs with a qualifying member | 243 |
| …with a core qualifying for every such member | 240 |
| Incompatibility certificates | 3, all with identical private blocks |

The three certificates (split 0 swapped, 3 base, 4 swapped) each have `r=0`,
`V_B={(0,0)}` and a single A block. Two members differ only in core, one
qualifying with `±(3,3)` and the other with `±(3,−3)`. A structure-only selector
cannot serve both.

The classes containing the named teachers give the 9×9 member-core ×
candidate-core matrices. On the **true** assembly of C2's class, only members
with core `±(3,3)` qualify, and only with `±(3,3)`. C2's own row (`3,−3`) is
empty. The same holds in S1's class in split 4 swapped, where the qualifying
member is S2. A fixed core succeeds on the true assembly only under the prior
"the teacher's core is that core".

**Post-hoc tally (not declared in the contract; computed after the scored run
from the same code).** Every qualifying or route-unresolved core entry of every
member on every assembly, in all 10 runs (mirror pairs counted separately):

| Assembly | Core in member's own sign class | Other core |
| --- | --- | --- |
| Member's true blocks — qualifying | 984 | 0 |
| Member's true blocks — route-unresolved | 1,600 | 0 |
| Other surviving blocks — qualifying | 3,864 | 9,340 |
| Other surviving blocks — route-unresolved | 11,520 | 35,520 |

On correct private blocks the family never offers a reusable relation other
than the teacher's own coupling. Non-teacher cores qualify only on wrong-block
assemblies, where they compensate for the wrong private structure.

## Readings

| Declared category (section 5) | Observed |
| --- | --- |
| Acquisition failure | none (no empty masks) |
| Assembly interference | S1 9/10 runs, S2 9/10, T1 1 assembly; C2 none |
| No preserving core | only on interfering assemblies |
| No transfer core | every remaining S1, C2 and T1 true assembly: transfer cores exist but do not preserve |
| Core-dependent but route-unresolved | C2, 30 wrong-block assemblies |
| Qualifying | S2 only, by recovering its own core |

The representability question for the declared criterion is answered
negatively for S1 and C2, with exhaustive certificates over all nine cores. The
cause is a property of the criterion on this substrate: retention on assembly
asks the coupled system to keep isolated input–output behaviour that the
coupled teacher itself does not have.

## What this does not establish

- That no core-only abstraction exists under a different preservation target.
  Judging retention against the teacher's coupled training behaviour, or
  splitting retention from transfer, is a different contract.
- Anything about acquisition or endogenous discovery. Stage A was evaluator-
  only, and nothing qualified to be acquired.
- That larger core capacity would help. The contract forbids adding capacity
  after a failure, and the failure here is not a capacity shortfall.
- That the C2 route-unresolved witnesses are abstractions. They occur only with
  incorrect private blocks.

## Accounting

| Item | Value |
| --- | --- |
| Acquisition: filter evaluations (40 runs × 5 orders + repetition pass) | 15,475 |
| Ceiling search: assemblies / core records | 124 / 1,116 |
| Stage-A evaluation: simulation requests / unique | 496,965 / 12,088 |
| Stage-B evaluation per run: assess calls / unique simulations | ≈8,700 / 121k–141k |
| Private candidate-mask storage | 18 bits per run |
| Executable coefficients per assembly / learned core coefficients | 7 / 2 |
| External comparator memory (memorization) | 15 entries per run |
| Wall time (10 workers) | ≈50 s |

None of this work is attributable to a hypothetical dream learner.

## Next obligations (not run)

1. Decide the preservation target before any v2. Options: (a) retention judged
   against the teacher's coupled marginals on training queries; (b) keep
   isolated retention but report transfer separately as a trade-off frontier.
   Either needs a new prospective contract.
2. If v2 keeps this substrate, report whether any non-teacher core qualifies on
   true private blocks. The post-hoc tally suggests not for this nine-core
   family.
3. The deferred items (preservation signal, plateau moves, larger edits,
   endogenous dreaming) remain deferred.
