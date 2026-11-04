# Phase structure module: one neural mechanism for relational queries

Status: **rejected by the user, 2026-11-04: not universal.** Its one operator covers relations that act as displacements along a dimension; other relation types need added gates and generators, and non-transformation relations fall outside it. Kept as a record. Nothing was run.

Original status: design draft, 2026-11-04; not a frozen contract. It answers the user's request for a new neural component that answers relational queries, where several steps are not computed as several literal steps. It replaces the symbolic evaluator of the [relational expression note](relational-expression-core-design.md) as a system component. That evaluator stays a diagnostic only, by the user's ruling that the system should be purely neural.

## The job

One component should answer, from core codes of any sense:
- **matching:** which candidate is the same quantity;
- **succession and k-step relations:** what is k after (or before) this, for any k, in one pass;
- **continuation with inferred direction:** given previous and current, what comes k later;
- **comparison:** which is more;
- **relation identification:** how two items are related.

It should do this so that **a new name taught only by sameness inherits all of these** (the naming-only goal). It should protect old abilities during relearning, and work with memory present.

## What earlier results require

**[read: local results]**
- **The heads gave transfer.** The affine successor, distance judge and order vector placed never-paired forms (0.42) and transferred comparison to forms (0.88). Their fixed forms arranged the shared code geometrically ([generic relations v1](generic-relations-v1-results.md)).
- **But they were ruled a mistake** (user, 2026-10-29), because they trained relational knowledge directly into the shared code space. That goes against hypothesis H: reusable knowledge should emerge from interactions among modules.
- **The generic comparator `g` keeps what is taught and loses transfer:** placement 0.13, form comparison 0.51.
- **P (sequence prediction) partly restores transfer,** but its rolled-out states are not readable by `g` and degrade with steps ([sequence pilot](sequence-pilot-v1-results.md)).
- **In relearning, P is rewritten by forgotten forms' errors.** That is the composition loss after update 10 ([predictor stop](relearning-predictor-stop-v1.md)).
- **One-checkpoint diagnostic** (seed 300, `delay`; the stage 2 smoke run, not a result): chaining `g`'s `next` scores read two-step composition at 0.91 on collections, against P's rollout at 0.66. The relational knowledge in `g` is better than P's use of it.

So the component needs:
- a **structural prior that makes transfer follow from sameness;**
- one that lives **in its own module, not as relation-specific heads or losses on the core code;**
- **no free predictor** that relearning can rewrite;
- **no literal iteration.**

## Design

### Placement

A new module **M** sits between the core and the readers. It reads the core-inner code c (16 dimensions) of any item: a collection, taps, a form, or a recalled memory trace through the current core. It keeps its own structural state. The core code is not required to be a line. M's map into its state is learned.

### State: a phase code

```
θ(c) = E(c)                          E: small MLP, 16 → 64 → m angles  (outputs 2m numbers, angle = atan2 of each pair)
z(c) = exp(i θ(c)) ∈ C^m, |z_l| = 1  m unit complex units
```

The unit modulus is the constraint, and so the pressure: positions are phases, and nothing else can be stored there.

### One operator: displacement

M has one learned generator ω ∈ R^m, the phase advance per unit step. **Every relational query is a displacement of the source state:**

```
target(c, k) = z(c) ⊙ exp(i k ω)     elementwise; any k ∈ R, computed in one operation
```

| Query | Displacement | Note |
|---|---|---|
| Same quantity | k = 0 | matching is the zero displacement, not a separate relation |
| Successor, k steps up | k | k = 2 or 3 costs the same as k = 1 |
| Predecessor, k steps down | −k | inverse is the negative exponent |
| Continuation from (prev, cur), k ahead | u = z(cur) ⊙ conj(z(prev)); target = z(cur) ⊙ u^k | the observed step u carries the direction; u^k = exp(i k·arg u), one operation; nothing supplies the direction |
| Comparison, more(a, b) | d̂(a, b) > 0 | d̂ = least-squares displacement estimate along ω from arg(z(a) ⊙ conj(z(b))) |
| Relation identification | d̂(a, b) itself | 0 same, +1 successor, −1 predecessor, sign is order |

**"Several steps are not several steps":** k enters as a multiplier on the phase advance, as in path integration with a velocity and a duration. There is no loop, no rollout and no intermediate state, so no compounding of intermediate errors (the failure of P's rollout and of recursive embedding traversal, Guu et al.). The cost is O(m) for any k.

Relations among relations become arithmetic on displacements: inverse is −k, composition is k₁ + k₂. The architecture provides these. They are not learned and not supplied rules at query time.

### Readout: attention over candidates or memory

```
s_j = (1/m) Σ_l cos(arg target_l − arg z_l(c_j))     phase agreement with candidate j
α = softmax(β s), answer = argmax α                     β one learned scalar
```

- **Candidates** can be the fixture's 9 items or, at scale, items proposed by memory.
- **With memory:** memory stores a structural key z next to each sensory trace, the conjunctive address already approved, and retrieves traces by target. The answer is then a recalled trace in any sense, read through the current core.

This is TEM's split: structure in one module, sensory binding in memory.

### Training: one loss family, no relation-specific losses

Every training signal is **"predict the observed item as the answer to a displacement query",** read through the same attention, with cross-entropy over the candidates:
- **pairings** (form with collection, taps with collection): k = 0, the answer is the paired item;
- **sequences** (prev, cur → next): u inferred from the pair, k = 1, the answer is the next item.

There are no succession, comparison or composition losses. Comparison, k ≥ 2, reverse completion and relation identification are **never trained** and are read from the same state. Gradients flow through E into the core: shaping the representation is the point.

### Naming-only transfer, by construction of the pathway

A new script taught only by sameness (k = 0 against collections) is pulled to the collections' phases. Every other query is a function of z only, so it then applies to the new forms unchanged.

**Prediction [derived]:** transfer to untaught relations is bounded by, and should track, the new forms' matching accuracy. In the generic `g`, matching and relational transfer separate, because `g` can learn kind-specific scoring.

### Why it may resist the relearning damage seen so far

- **[derived]** The only shared transition parameters are ω (m numbers) and β. There is no predictor network for forgotten forms' errors to rewrite, which was the route of the composition loss after update 10.
- The other route remains: sequence errors pulling target collections through the candidates' states and E. It was the early interpreter collapse.
- Under naming-only training, form sequences are absent, so that route is not exercised by new names. This must be measured, not assumed.

## What is supplied and what is learned

| Supplied by the architecture | Learned |
|---|---|
| Unit-phase state; displacement as phase multiplication; inverse as negation; composition as addition; cosine phase readout | E: where each item of each sense is placed |
| One generator per relational dimension (G = 1 here) | ω: the step size per unit |
| k as the query's magnitude | β; which items share a place (grounding, cross-sense alignment) |

Inverse and composition consistency hold by construction for any placement. That is architectural pressure, not evidence of learned reverse reasoning. What is learned, and what the experiments test, is whether training places the world so that these operations give correct answers on untaught queries and new names.

## Is this a relation head?

The decision is the user's. The distinction from the rejected heads:
1. **One mechanism, not one head per relation.** Same, next, more, k-step and continuation are all the same displacement, read the same way.
2. **It lives in its own module.** The core code is not trained to be a line. M's map E is what is shaped, and the core only through it.
3. **No relation-specific losses.** Only sameness and observed sequences are trained.
4. **Answers are read through attention over candidates or memory,** not by a distance on the core code.

It does share the heads' key property, a fixed geometric form that arranges a code, because that form is what gave transfer. If H requires that no fixed relational geometry exist anywhere, this design is excluded and transfer has to come from elsewhere.

## Prior work

Review 2026-11-04. Abstracts read where marked **[abstract]**; **[familiar]** means from memory, not re-read today.

| Work | What it establishes | Relation to this design |
|---|---|---|
| Tolman-Eichenbaum Machine, Whittington et al. 2019/2020 (bioRxiv 10.1101/770495) **[abstract]** | Entorhinal cells as a basis for structural knowledge, hippocampal cells binding it to sensory input. It produces grid- and place-like cells, and structure is preserved across environments under remapping. | The closest model: M is the structural code and memory is the binding. TEM learns transitions per action and generalizes structure to new sensory environments, the analogue of new names. |
| Whittington, Warren and Behrens 2021, transformers and the hippocampal formation (arXiv 2112.04035) **[abstract]** | Transformers with recurrent position encodings reproduce place and grid representations and perform much better than TEM. | Memory as attention over stored items; position as a code updated by actions. It supports reading out by attention over memory. |
| RotatE, Sun et al. 2019 (arXiv 1902.10197) **[abstract]** | Relations as rotations in complex space capture symmetry and antisymmetry, inversion and composition. | The same algebra (inverse = conjugate, composition = product of rotations), here with one generator scaled by k. |
| Guu, Miller and Liang 2015 (arXiv 1506.01094) **[abstract]** | Recursive path queries in embedding models suffer cascading errors; compositional training reduces them and regularizes completion. | One-pass displacement avoids the recursion. Errors in placement still scale with k. |
| Query2box, Ren, Hu and Leskovec 2020 (arXiv 2002.05969) **[abstract]** | A point embedding cannot represent many answers; boxes do. Disjunction needs dimension linear in entities. | A point target with attention gives several answers only if several items share a place. That suits renderings of one quantity, not one-to-many relations. |
| Cueva and Wei 2018 (arXiv 1803.07770) **[abstract]** | RNNs trained on velocity-based path integration develop grid-like codes. | Phase codes emerge from path integration. Here the phase form is imposed and only placement is learned. |
| Stachenfeld, Botvinick and Gershman 2017 (bioRxiv 10.1101/097170) **[abstract]** | Hippocampus as a predictive map; grid cells as a low-dimensional basis for it, with multiscale structure. | Multi-step prediction from one representation, not by iterating. |
| Fiete, Burak and Brookings 2008, J Neurosci **[familiar; fetch returned 403]** | Multi-module grid codes represent a range that grows very rapidly with the number of modules. | The scaling argument for phase over a real line; it needs checking before it is relied on. |
| TransE and its 1-to-N limitation; TransH and TransR **[familiar]** | Translation-type relations cannot model one-to-many or many-to-one relations. | M covers relations that act as transformations (orders, steps, maps). Arbitrary associations ("is the name of", "has colour") stay in memory. |

**Known failure modes:**
- **Phase wrap:** periodic phases make the ends of the line meet. Out-of-range queries return wrapped answers unless the readout abstains on low agreement.
- **Collapse:** all items at one phase satisfies k = 0. Candidate contrast prevents it only if negatives are present.
- **Uneven spacing:** if E places counts unevenly, k-step error grows with k.
- **One-to-many relations** are outside the operator.
- **Blending:** attention mixes. The design never feeds a blended state back, so this is avoided.

**Cheaper alternatives, as named reference arms (not a sweep):**
- **real translation:** z ∈ R^m, target = z + kω, score −‖target − z_j‖². The simplest version, and the closest to the rejected affine step;
- **the current split readers:** `g` and P with rollout;
- **a k-conditioned MLP predictor:** no structure, the no-pressure baseline. It is predicted not to extrapolate in k.

## Computation and scaling

- **Parameters:** E has 16·64 + 64·2m weights (m = 8: about 2.1k), ω has m, β has 1. That is fewer than P (about 7.3k) or `g` (about 6.8k).
- **Per query:** O(m) to form the target for any k. Candidate readout is O(N m). With memory, retrieval T_retrieve(M, K) + O(K m), where a full scan is O(M m).
- **Storage:** structural keys O(M m) next to memory's traces.
- **Scaling:**
  - k costs nothing extra: constant in k, against O(k) for rollout and O(k N²) for symbolic chaining.
  - The range of distinguishable positions grows with more phase units at several frequencies **[familiar; to verify]**.
  - Several independent relational dimensions need G generators and an m that grows with G.
  - One-to-many relations do not scale through M at all; memory carries them.
  - The nine-count fixture cannot show range. A small scale probe can.
- **Measure:** wall and CPU time, peak memory, training exposures and atom counts, separately from the scientific results, following [computation reporting](computation-reporting.md).

## Proposed stages (none started)

1. **Scale probe without senses** (minutes, CPU): a synthetic line of 100 positions, items with random codes, several renderings per position. Train k = 0 pairings and k = 1 sequences only, then test:
   - k = 2 to 10;
   - reverse continuation;
   - comparison;
   - a new kind taught only by k = 0.

   Arms: phase, real translation and k-conditioned MLP. This tests the mechanism, not the project's perception.
2. **Read-only fit on the step 8 checkpoints:** E, ω and β trained on frozen core codes (seeds 300–303, `delay`), core frozen, and read on the same queries as P and `g`, with memory present. It asks whether the existing core already supports the structure.
3. **The naming-only contract** from the design note, with M replacing `g` and P as readers, trained end to end. Matched against the split-reader curriculum, with relearning and memory-present reads central.

## Decisions for the user

1. Whether this module is acceptable under H, given the "relation head" section above.
2. Whether M should also replace `g`'s matching (k = 0), as designed, or sit beside `g`.
3. Whether to start with stage 1, the scale probe.
