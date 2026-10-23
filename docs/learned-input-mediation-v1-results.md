# Learned input mediation v1 — outputs identify the mechanism here; for one negative control they leave it uncertain

**Completion:** `complete`. Every correctness gate passed and the full declared grid
ran: 4 teachers × 2 arms × 5 splits × 5 orders × 4 checkpoints.

**Headline:**

- **Exact learners:** for S1 and C2, the exact version-space learners recover both
  unseen response sets and the tested mechanism from 15 axis-evidence queries, in
  both arms and in every split.
- **Local learner:** clamp supervision (arm I) makes recovery consistent: 25/25
  runs for both S1 and C2. With canonical outputs alone (arm O) recovery is
  possible but unreliable: C2 in 9/25 runs, S1 in 0/25. Arm O misses target 4
  for both.
- **Mechanism uncertainty:** the outcome this design was built to detect occurs
  for the negative control S2. In arm O, five survivors in every split match all
  30 within-context held-out outputs, yet carry a forward input chain the teacher
  does not have. The teacher and its mirror survive as well, so the version space
  is mechanism-heterogeneous and the learner certifies no mechanism. It reports
  uncertainty; it does not falsely certify a chain. The contract counts any such
  survivor as false attribution, so target 5 fails as declared. Arm I removes
  the false survivors.

[Contract](learned-input-mediation-v1-contract.md) (sha256 `48b17786…`, frozen in
the manifest before computation; not edited afterwards) ·
[manifest](../reports/learned-input-mediation-v1-manifest.json) ·
[battery summary](../reports/learned-input-mediation-v1-battery-summary.json) ·
[ceilings](../reports/learned-input-mediation-v1-ceiling.json) ·
[learning](../reports/learned-input-mediation-v1-learning.json) ·
[runtime](../reports/learned-input-mediation-v1-runtime.json). The run takes
95 s on 8 workers: 47 s battery, 9 s oracle checks and reproduction, 25 s learning
and evaluation, 8 s local-model gates. Peak resident memory is 259 MiB.

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Independent-oracle check: every one of 1,647 atoms, plus exhaustive qualifying-tuple sets, for the 4 teachers and the 50 indices `44·j` | 54 / 54 hypotheses equal |
| Frozen reproduction: S1, C2, S2 conditions 1–7, 3′ and forward/reverse tuples against the shared-input archive; T1 conditions 1–7 against v1 | all equal |
| Teacher eliminated / survivor retention violations (full, positive) | 0 / 0 |
| `certain ⊆ teacher ⊆ possible` (set atoms), teacher value among survivor values (Boolean atoms) | 0 violations |
| Final V depends on order | 0 |
| Arm-I V not a subset of arm-O V, same prefix and rule (800 checks) | 0 |
| `inconsistent_feedback` | 0 |
| Byte-identical regeneration of manifest, battery summary, ceilings and learning, in a temporary directory with a different worker count | yes |
| Tests | 15 new; full suite 189, all passing |

**Defect found before any scored run.** The first oracle test failed. Production
`pair_mask` *summed* bits over terminal states, so two terminals with the same
`(a,b)` corrupted the mask. It was fixed to bitwise OR before the experiment ever
ran, and a regression test was added.

## Identification ceilings (computed before learning)

For **every** teacher and **both** arms, the complete-information class over all
81 queries has exactly **two** members: the teacher and its core-sign mirror,
`(w_KA, w_KB) → (−w_KA, −w_KB)`. Each class is **mechanism-homogeneous**. The
mirror reproduces every battery atom, because the battery does not observe k.

So in this hypothesis space and battery, canonical outputs on all 81 queries
already determine the tested mechanism. Arm I adds nothing at full information.
The ambiguity the study was designed to detect cannot arise at the ceiling. It can
arise only from limited training coverage, and there it does (S2 below).

No ceiling obstructs any target. No class member has a forward witness for S2 or
T1. For S1 and C2, both class members carry the teacher's forward witnesses.

## Targets (S1 and C2; checkpoint 15)

| Target | S1 | C2 |
| --- | --- | --- |
| 1. Output identification, arm O: certified coverage within / mixed; incorrect certified | **pass**: 150/150, 180/180; 0 | **pass**: 150/150, 180/180; 0 |
| 2. Mechanism identification, arm O: splits where all survivors are mechanism-equivalent | **pass**: 5/5 (V = teacher + mirror) | **pass**: 5/5 |
| 3. Mechanism identification, arm I | **pass**: 5/5 | **pass**: 5/5 |
| Arm I strictly improves on arm O | no (both already at the ceiling) | no |
| 4. Local learner, arm O: accuracy within / mixed (zero baseline); retention; behavioural recovery | **fail**: 0.417 / 0.313 (0.367 / 0.222); 49.3%; 0/25 | **fail**: 0.624 / 0.544 (0.280 / 0.111); 56.2%; 9/25 |
| 4. Local learner, arm I | **pass**: 1.000 / 1.000; 97.4%; 25/25 | **fail on retention only**: 1.000 / 1.000; **77.2%** (B1-only 87.5%; see below); 25/25 |

Details behind the table:

- **Positive-only ablation.** For S1 and C2 it ends with the same 2 survivors as
  full feedback, so exclusions are not needed for these two teachers.
- **Speed of identification.** In split 0, order 0, the full arm-O version space
  already equals teacher plus mirror at checkpoint 5 for both teachers.
- **Local learner, arm I.** It converges to the core-sign mirror (S1 25/25; C2
  24/25, plus 1 run on the teacher itself). That is representational symmetry,
  not an error: the mirror passes the teacher-specific gate and reproduces every
  positive witness atom.
- **C2 retention in arm I is not a forgetting rate.** The metric is the fraction
  of earlier feedback maps reproduced after each update, pooled over all steps.
  A failed check can be an item not yet learned or an item learned and later
  broken. Of 2,625 checks, 598 fail. Only 19 of them are updates that break a
  map correct just before (`lost_previously_correct`). The other 579 were already
  wrong before that update: never acquired yet, or broken earlier and not yet
  repaired. The per-step counter does not separate those two cases. The final
  models are exact. For S1 in arm I the corresponding count is 12.
- **Local learner, arm O.** It drifts among many vectors: 15 distinct final vectors
  for S1 and 12 for C2. It recovers C2's behaviour in 9 runs (the mirror) and S1's
  in none. So observational feedback alone can support recovery with this rule;
  clamp supervision makes it reliable. Its worst-split accuracy is 0.0 on S1's mixed partition. Every per-run
  vector is in `runs`.

## Negative controls (target 5)

| Teacher | Arm O: full-reference survivors with a forward witness (per split) | Arm I | Local runs with a forward witness, O / I |
| --- | --- | --- | --- |
| S2 | **fail**: 5 of 7 in splits 0–3, and 5 of 15 in split 4 (all SF) | **pass**: 0 of 2 | 9/25 / 0/25 |
| T1 | **pass**: 0 of 2 | **pass**: 0 of 2 | 9/25 / 0/25 |

Neither negative teacher has a forward witness of its own, so the assumption that
they are negative holds.

**The S2 failure is the case this study was designed to find.** In every split, 5
arm-O survivors agree with S2 on **all 30 within-context held-out outputs**. They
differ on the mechanism battery, and each carries a false SF chain. Within that
partition the observations certify the outputs (30/30 in splits 0–3) but not the
mechanism.

This is preserved uncertainty, not a false certificate. The teacher is never
eliminated, so the correct candidates (teacher and mirror) survive beside the
false ones. The surviving set disagrees on the SF chain, so the exact learner
cannot and does not certify it. That is the version space working as intended.
The target fails because the contract demands zero false-chain survivors, a
stronger requirement than sound certification.

These survivors are separable. Only 2 of the 7 match S2 on the mixed-evidence
queries, so more observational coverage would expose them, as the ceiling says it
must. Clamp observations (arm I) remove them with the same 15 training queries.

Arm I gives richer observation as well as intervention: it also reads the internal
input coordinates. Its advantage therefore cannot be attributed to intervention
alone.

## Value of negative information (reported)

| Teacher, arm O | Full-feedback survivors | Positive-only survivors | Certified outputs, positive-only, within / mixed |
| --- | --- | --- | --- |
| S1, C2 | 2 | 2 | full coverage |
| S2 | 7–15 | 24–40 | 36/150, 30/180 (full feedback: 146, 126) |
| T1 | 2 | 11–15 | 38/150, 88/180 (full feedback: 150, 180) |

For S2 and T1, exclusions decide both certification and mechanism equivalence.
That differs from selective-satellite-mediation v1, where a selection prior had
masked them. In arm I the two rules coincide (2 survivors everywhere).

S2 and T1 also have output and local-learner numbers in the report. They are
labeled `target_applies: false`, because targets 1–4 are declared for S1 and C2
only. S2's arm-O certified coverage on the mixed partition is 70%.

## Proofs (scoped)

1. **Battery items are functions of h.** So version-space retention, order
   independence (V is an intersection of per-experience filters) and soundness of
   certification (`certain ⊆ teacher ⊆ possible` whenever the teacher is in V)
   carry over to every atom.
2. **Ceiling.** Every member of an arm's complete-information class survives any
   clean subset of that arm's observations. A mechanism-heterogeneous class would
   therefore block sound certification. None was heterogeneous here.
3. **Arm I is no weaker than arm O.** Arm-I feedback includes arm O's B1 entry
   under the same filter, so `V_I ⊆ V_O` for each prefix and rule (800 checks).
   Strict improvement is not guaranteed, and it occurred only for S2.
4. **Core-sign symmetry.** Under `k → −k`, `w_KA → −w_KA`, `w_KB → −w_KB`, the
   energy is unchanged and the start state (k = 0) is fixed. Every intervention
   maps to its mirror, and k is never observed, so every atom agrees (tested on
   three vectors, and visible in every ceiling).
5. **A zero chain link makes a template impossible.** If a template's link
   coefficient is already zero, ablating it does nothing. The full contrast that
   (c) requires then survives (d), so the template can never qualify. This is why
   S2 and T1 cannot have a CF witness, and T1 cannot have an SF witness (tested).

## Cost

| Item | Count |
| --- | --- |
| Battery: inference requests (unique; cache hits) | 11,803,703 (3,341,129; 8,462,574) |
| Atoms per hypothesis (of which fixed-witness) | 1,647 (54) |
| Learner simulation service, arm O / arm I: requests (unique) | 810,925 (67,770) / 1,010,872 (123,398) |
| Version-space evaluations, arm O: full / positive | 361,729 / 408,696 |
| Version-space evaluations, arm I: full / positive | 381,331 / 399,501 |
| Local learner, arm O: evaluations / commits / coefficients changed | 40,500 / 552 / 677 |
| Local learner, arm I: evaluations / commits / coefficients changed | 230,040 / 466 / 678 |

Arm I has more supervised loss terms per experience, so it does not have equal
supervision cost. Persistent learned state is 2,187 bits for V and 7 ternary
coefficients for the local model.

## What this does not establish

- **Equivalence is finite.** "Mechanism-equivalent" means agreement on this finite
  battery (NF, freezing, core removal, single-link removals, satellite clamps and
  chain atoms). It is not equivalence under every possible intervention.
- **Easy supervision.** Supervision is realizable, noise-free and complete-set,
  from four hand-chosen teachers that the researchers fully characterized
  beforehand. "Held out" means withheld from the learner only.
- **No plasticity law.** The local learner uses global set error, and its arm-O
  failure says nothing about other local rules.
- **Retention is not forgetting.** The retention metric mixes items not yet
  acquired with items acquired and later broken (see the C2 arm-I note).
- **No cause for the arm-O local failure.** The run does not show why arm O
  fails. That the rule needs coordinated cross-block edits is a hypothesis.
- **Arm I's advantage has two sources.** It combines intervention with access to
  internal input readouts.
- **No broader claims.** Nothing here shows autonomous discovery, robustness,
  semantics, or that energy descent is SRI's foundation.

## Next obligations (not run)

The open question is whether the mechanism can be learned reliably from less
privileged feedback while preserving earlier behaviour. The next study should
isolate what makes local learning succeed. Keep the local update rule fixed at
first and compare three supervision conditions:

1. Ordinary observations with additional mixed-input coverage.
2. Satellite clamps with only external satellite-output (B1) readout.
3. The existing clamps plus internal input readouts (arm I as run here).

For each condition, report supervision cost, final mechanism recovery, and
acquisition and forgetting as separate quantities. Track per-item state so an
item never acquired is distinguished from one acquired and later lost. The same
conditions also answer the S2 coverage question: which feedback removes false
chain survivors without internal readouts. A change to the update rule, for
example one that coordinates edits across blocks, should come only after this
comparison. It is a hypothesis, not a finding.

This needs a new contract; a draft is
[local learning supervision v1](local-learning-supervision-v1-contract.md).

**Correction for future runs.** In this run, the `completion` field does not
depend on `tests['passed']`. A failed test suite could therefore still report
`complete`. The tests passed here (15/15), so this run is unaffected. The fix
(`completion_status`, which reports `stopped_failed_tests`) is in
`experiments/mechanism_learning.py`, with a regression test in
`tests/test_mechanism_learning.py`. Both files' current hashes therefore differ
from those recorded in the report's provenance.

Reproduce:

```bash
.venv/bin/python -m unittest tests.test_mechanism_learning -v
.venv/bin/python -m experiments.mechanism_learning
.venv/bin/python -m experiments.mechanism_learning --out-dir /tmp/lim-regen --workers 4
for f in manifest battery-summary ceiling learning; do cmp reports/learned-input-mediation-v1-$f.json /tmp/lim-regen/learned-input-mediation-v1-$f.json; done
.venv/bin/python -m unittest discover -s tests -v
```
