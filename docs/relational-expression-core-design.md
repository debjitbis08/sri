# A shared relational expression interface for the existing tasks

Status: **design note and agent handoff, 2026-11-04. Not a frozen experiment contract.**
Requested by the user after the predictor-stop discussion. This proposes a direction;
it does not authorize replacing the working system or starting a training sweep.

## Goal and answer

**Yes: the existing matching, comparison, succession, sequence, composition and
memory-assisted tasks can use one relational expression interface.** Their answers
can be truth judgments or bindings for missing arguments, including the relation.
This is an interface and inference proposal, not evidence that one shared neural
scorer will learn the required abstractions or avoid interference.

The user's intended capability is: teach a new name for an understood concept,
then use existing relational knowledge through that name, in directions and tasks
not explicitly taught for that name. Preserve old abilities while learning it.
The system must represent false statements, multiple answers, and relations among
relations, with a path to composition and non-mathematical domains.

Memory and core working together online remain the goal. Consolidation and sleep
remain future work. Do not turn this into a requirement that the core memorize
arbitrary stored facts or work without memory.

## What motivates the change, and what it does not establish

**[read: local code and reports]** `src/generic_net.py` has a comparator
`g(concat(a, b, relation_token)) -> scalar` and a separate sequence predictor
`P(concat(previous, current)) -> code`. Matching ranks comparator scores; sequence
prediction ranks candidate codes by distance to P's output. See
[predictor-stop results](relearning-predictor-stop-v1.md) and
[margin attribution after candidate stop](relearning-margins-stop-v1.md).

The experiments locate substantial interference in candidate representations and
then in P. They do not prove that separate readers cause all interference, or that
a unified reader solves it. **[analogy]** Sharing relational semantics could remove
one source of inconsistency between judging a relation and using it to find an
answer. Shared parameters can still suffer catastrophic interference.

**[read: local code and reports]** Collections -> taps already reverses the learned
matching direction through the same g. `number_construction_part_a.batch_material`
adds form -> collection matching; `number_construction_part_d.batch_material` adds
taps -> collection matching (or collection -> collection in collection context).
The reverse read is not a separate predictor. In the candidate-stop trajectory's
updates 10–19, form matching contributes -6.5 of -7.8 mean lost margin on newly lost
collections -> taps decisions. Its largest cell is form matching through the
comparator (-4.9); the comparator's net contribution across paths is -1.6, while
the image groups' net contribution is -5.3. These are different decompositions,
not evidence that all remaining damage lies in g. See the
[attribution table](relearning-margins-stop-v1.md).

This is direct local evidence that reusing the same scorer does not prevent
interference. Moving continuation and composition into g places more uses on a
reader already involved in the remaining interpreter damage. The baseline must
measure this risk, rather than presenting shared scoring as a new retention mechanism.

The current form curriculum directly trains **both form -> form succession in g
and form sequence prediction through P**, as well as matching. Forms also supply
within-kind succession negatives (`relearning_mechanism.forward`). Removing only
form rows from P would therefore not isolate transfer from naming alone.

## Expression structure

Use a small, explicit expression graph. Do not require an unrestricted higher-order
theorem prover for the first implementation.

```text
Term       = EntityRef | RelationRef | Var | ExpressionRef
Atom       = Apply(relation: RelationRef or Var, arguments: role -> Term)
Formula    = Atom | And(Formula...) | Not(Formula) | Exists(Var..., Formula)
Assertion  = (formula, polarity, optional context/evidence)
Query      = (formula, requested variables, candidate domains, answer policy)
```

Roles and variable scope are structural. A repeated variable has the same binding;
renaming bound variables must not change meaning. Binary roles such as `from/to`
or `left/right` are enough for this pilot, but arity must not be fixed in the
expression format. Relation and expression references are first-class terms;
their neural evaluation is not automatically implemented merely by parsing them.

An EntityRef identifies an observation or stored trace, whose sensory code is
computed through the existing sense and core. It must not secretly identify the
true quantity. Variable identity, entity identity, and learned sameness of
quantity are different things.

```text
e = same_quantity(left: word_three_image, right: three_dot_image)
Assertion(e, true)

e_false = same_quantity(left: two_dot_image, right: three_dot_image)
Assertion(e_false, false)
```

Both expressions are valid. Falsehood is a judgment about an expression, not a
failure to construct it. `same_quantity` means equality in this respect, not equal
spelling, equal appearance, or full object identity.

The user's `[relation, a, b, T/F/?]` can be surface syntax: the first three fields
form an atom; T/F supplies assertion polarity and `?` requests its truth judgment.
Other missing fields become named variables, not anonymous holes.

For the finite pilot, use explicit positive/negative labels and evaluate truth
separately from candidate ranking. Lack of a stored fact is not evidence of
falsehood. Do not interpret a neural score as calibrated confidence, or graded
truth as epistemic uncertainty, without defining and testing that semantics.
General contextual belief and contradiction handling remain design extensions.

## Mapping the existing tasks

In this table `d3`, `t3`, and `w3` are shorthand for observations, not count IDs
available to inference. Candidate sets retain the existing held-out renderings.

| Existing task | Relational query | Output |
|---|---|---|
| Form -> collection matching | `same_quantity(w3, ?x)` with collection candidates | Bindings for `?x` |
| Collections -> taps | `same_quantity(d3, ?x)` with tap candidates | Bindings for `?x` |
| Taps -> collections | `same_quantity(t3, ?x)` with collection candidates | Bindings for `?x` |
| Same/different judgment | `same_quantity(d2, d3)` | Truth judgment |
| Succession judgment | `successor(d2, d3)` | Truth judgment |
| More-than comparison | `more_than(d3, d2)` | Truth judgment |
| Given-relation completion | `successor(d2, ?x)` | Bindings for `?x` |
| Reverse completion | `successor(?x, d3)` | Bindings for `?x` |
| Relation identification | `?r(d2, d3)` over a declared relation domain | Compatible relation bindings |
| Two-step composition | `Exists y: successor(d2,y) AND successor(y,?z)` | Endpoint bindings and witnesses |
| Memory symbol -> taps | Retrieve value `v`; query `same_quantity(v,?x)` | Tap bindings, with retrieval accounted for |
| Memory symbol comparison | Retrieve `v1,v2`; query `more_than(v1,v2)` | Truth judgment, with retrieval accounted for |

Preserve existing reverse memory reads too: use the same retrieved-value/candidate
protocol, replacing only the relational read. A memory fact is an arbitrary
association, not automatically a quantity-equality assertion about its visual key.
Retrieval abstentions and errors remain part of end-to-end results. Correct-value
bypass and memory-disabled reads are diagnostics, not substitutes for those results.

### Sequence continuation must still infer direction

There is no learned predecessor token: the current tokens are `next`, `same`, and
`more` (`src/generic_net.py`). In the checkpoint wrapper, predecessor must be the
**exact supplied inverse**, `predecessor(a,b) := successor(b,a)`. Direction is
inferred using the learned next scores, but the inverse construction is supplied
algebra in every such continuation result, not learned inverse knowledge.

The existing P receives two items, not a supplied `+1` instruction. Replacing it
with a successor query would make the task easier and change the information given.

For this fixture, express continuation as:

```text
Exists r in {successor, predecessor}:
    r(previous, current) AND r(current, ?next)
```

Keep the same relation binding for subsequent steps. For example, add
`r(?next, ?next2)` for two-step continuation. Enumerate valid bindings jointly,
or propagate the full scored alternatives, rather than silently committing to
one greedy intermediate result. Report the aggregation rule.

This restricted relation domain is an explicit fixture prior. Two observations
do not determine a unique sequence rule in a general world. Do not advertise this
as general sequence induction. Supplied-direction and inferred-direction reads
must have separate names and results.

### Training support differs between g and P

**[read: local code]** `number_construction_part_a.batch_material` and
`number_construction_part_d.batch_material` construct succession pairs within a
kind: form -> form, collection -> collection, or taps -> taps.
`relearning_mechanism.forward` also draws succession negatives within the query's
kind. P's sequence loss instead scores predicted codes against collection
candidates, including when its inputs are forms or taps. Comparison (`more`) has
a mixed collections-and-taps sampling pool, so it does not have the same restriction.

Consequently, `successor(taps, ?collection)` through g is an untrained cross-kind
combination, even when it has the same output domain as the historical P read.
A weak result does not by itself indict the expression interface. Stage 2 must
report both within-kind candidates (matching g's succession training support) and
cross-kind collection candidates (matching P's output protocol), with separate
labels. Audit every edge of a composed path, including intermediate candidates;
a within-kind start does not make later cross-kind edges trained. Report query
direction, kind pair, candidate domain, and direct supervision support per read.

### Multiple answers and domains

Keep the legacy nine-candidate top-1 measure for historical comparison, but add a
fixture with several valid observations per quantity. Return binding sets/rankings;
report precision and recall or an explicitly defined multi-answer metric. Every
valid alternative must be labeled positive, never used as a negative merely because
another rendering was selected as the target. Multiple relations may also hold
for the same pair, such as `successor` and `less_than`.

Report candidate-domain coverage: no answer in a finite domain does not establish
that no answer exists in the world. Keep out-of-range successor cases explicit.

### Scaling beyond a closed candidate domain

**[analogy/proposal]** Exhaustive enumeration is a finite diagnostic, not the
open-domain deployment mechanism. The proposed route is **memory retrieval proposes
candidates; the shared relational scorer verifies them**. Reuse frozen sensory
traces and current-core interpretation; do not store images or oracle count IDs.
Proposal retrieval for a missing argument is a distinct role from retrieving a
symbol's stored fact, and is not already implemented by naming it here.

For a memory of M items and a proposal budget K, query cost is
`T_retrieve(M,K) + O(K C)` plus code construction; C is one atom-scoring cost.
Index construction, updates, retained bytes and retrieval latency must be measured.
Do not assume sublinear search: a full-scan proposer still costs O(M) distance
evaluations. Composite queries also need intermediate witnesses; bounded candidate
sets or beams can discard a valid path even when its final answer was retrieved.

**[derived]** If the verifier can only return proposed candidates, unique-answer
end-to-end accuracy is at most the probability that the answer is proposed. For
multiple-answer queries, final answer recall is at most proposal-set recall.
Report proposal recall@K, verification conditional on answer inclusion, final
precision/recall and abstention, and coverage of complete witnesses for composition.
Do not multiply per-hop recalls without independence evidence. On a proposal miss,
return no verified answer/unknown or expand retrieval under a declared budget;
never infer falsehood merely from retrieval failure. General generation of genuinely
new entities is outside this retrieval-only mechanism.

Stage 2 retains exhaustive finite domains so reader capability can be measured
without proposal misses. A subsequent scaling contract must specify the proposer,
its query representation, M/K/chain-length scaling, and the quality-cost tradeoff.
No open-domain scaling claim follows from the nine-candidate baseline.

## Smallest useful evaluator: reuse the comparator first

**[derived]** The existing comparator already scores a completed binary atom.
The cheapest executable baseline is an adapter around it, with the senses,
projections, relation embeddings and comparator unchanged:

```text
score(r, a, b) = g(concat(code(a), code(b), embedding(r)))
```

Enumerate candidates for the missing slot using this same scorer. Asking for the
left argument instead of the right does not create a separate predictor. This
first baseline can run on existing checkpoints without retraining. It tests the
interface and measures what current relation knowledge supports; it does not
establish that a new architecture was trained.

Cross-relation scores may not be comparable: current matching uses a categorical
ranking loss while succession/comparison use binary objectives. Measure relation
selection separately and declare any calibration data. Do not hide per-task
calibration or learned completion heads behind a claim of unified inference.

For finite composition, an exact reference may use thresholded predicates with
Boolean conjunction and existential enumeration. A soft evaluator needs a declared
conjunction and existential aggregation (e.g. max-min is one possible fuzzy
semantics). Different choices change meaning and ranking; they are not interchangeable
probabilities. Do not select an aggregator by repeatedly inspecting held-out scores.

An AST wrapper alone does not supply symmetry, equality substitution, or composition
knowledge. If all that changes is syntax, identify the result as a wrapper baseline.

## Relations among relations, and what is supplied versus learned

The representation permits `inverse_of(successor, predecessor)` and
`symmetric(same_quantity)`. Their operational consequences require either an
explicit definition/rule or learned behavior; a relation name alone does nothing.

For binary relations, possible definitions are:

```text
inverse(R)(x,y) := R(y,x)
compose(R,S)(x,z) := Exists y: R(x,y) AND S(y,z)
```

**[derived]** Implementing `predecessor` as the exact inverse of `successor` makes
that consistency true by construction. It is a useful algebraic baseline, but not
evidence that reverse reasoning was learned. Likewise, symmetry imposed by averaging
`g(a,b,r)` and `g(b,a,r)` cannot count as learned symmetric transfer.

Quantity equality must not substitute expressions in every possible relation.
For example, equal quantities do not imply equal spellings. A supplied substitution
rule must name the relations/roles for which it is valid. Distinguish three sources
of capability in every result: supplied algebra, learned sensory grounding, and
learned relational generalization.

Do not introduce oracle concept IDs such as integer `3` into inference to make
grounding appear solved. A symbolic count oracle is useful only as a clearly labeled
semantic reference, not as the learned system.

## Staged work for the agent

### 1. Inventory and deterministic semantics checks

Before a new training run, map the exact loaders, losses, readers and candidate
protocols to the table above. Record any mismatch, especially inferred direction,
multi-step rollout, inverse memory reads, and negative sampling. Implement only
the finite expression/evaluator prototype needed for these tasks.

Use an explicit finite truth-table fixture to verify variable binding, argument
reversal, multiple answers, false statements, composition witnesses, and empty
domains. This checks implementation semantics; it makes no learning claim.
Keep this oracle fixture separate from perceptual evaluation.

### 2. Read-only checkpoint baseline

Wrap the current comparator and evaluate the existing tasks through completion
and finite composition on captured checkpoints for **seeds 300–303**, recording
their exact paths, hashes, arms and update indices. Compare to the current g/P
readers at the same checkpoints and with the same candidates where supported.
Add the separately labeled within-kind succession/continuation reads described
above; do not pool them with cross-kind transfer reads. Retain memory-present reads.
Report task-information
changes and new search costs separately. No architecture benefit can be claimed
from giving the evaluator direction or intermediate candidates unavailable to the
historical predictor.

### 3. Draft one training contract after the baseline

The main question is naming-only transfer. Train established quantity relationships
on collections and taps. Introduce held-out forms using **only sameness examples**.
Withhold form succession, comparison, sequence and composition supervision. Evaluate
both matching directions and the withheld relational tasks, including novel renderings.
Concretely, remove form -> form positive pairs from g's `next` loss, remove forms
from its negative pools, and remove form rows from P's sequence loss. Audit all
remaining non-matching loss arguments/candidates and generated labels for form
participation; do not merely mask positive sequence examples. Keep explicit counts
by loss, polarity and kind pair showing that forms receive only naming supervision.
Forms must not have had the withheld relational supervision earlier in the chosen
initialization either. Separate genuinely new-name transfer from relearning forms
with prior task exposure. Audit batches and labels so indirect supervision cannot
leak in.

Measure acquisition and preservation separately: taught matching, untaught transfer,
old collection/tap skills, withheld scripts, and memory-plus-core answers. Use a
long enough, predeclared schedule to assess acquisition beyond the recent ten-update
diagnostic. Report paired seed outcomes and uncertainty, not just overlapping ranges.

The old curriculum is a historical reference, not a matched naming-only baseline.
Compare the current split-reader design and the proposed shared relational evaluator
on matched examples/schedules, and disclose architecture and compute differences.
Any fixed-reader/representation-adaptation arm must be motivated and named in advance,
not assumed to be the solution. Grounding one task while harming another is still
interference. Do not train every form-task combination and call it transfer.

Before running, freeze initialization/checkpoint choice, training objectives and all
gradient paths, candidates, supplied rules, answer semantics, stopping criteria,
seeds, thresholds and computation budget. Fill these from the inventory; this note
does not invent a new loss or select a winning architecture without that evidence.

## Prior work and search record

Search/review date: 2026-11-04. Queries in the discussion included “miniKanren
relational programming run backwards”, “lambda Prolog higher order abstract syntax”,
“TensorLog differentiable deductive database”, “Logic Tensor Networks first order
logic real logic”, and “neural theorem provers end to end differentiable proving”.
Primary pages/abstracts were revisited for this note. This is a scoped review, not
a systematic survey or novelty claim.

| Source and evidence | What it establishes/reports | Difference and project hypothesis |
|---|---|---|
| [miniKanren](https://minikanren.org/), **[read: project introduction; familiar: operational relational programming semantics]** | A small extensible logic-programming core; relational programs search for bindings. | It does not learn perceptual grounding. Use its query discipline as a semantic reference, not a claimed neural implementation. |
| [λProlog](https://www.lix.polytechnique.fr/~dale/lProlog/), **[abstract: project search summary; direct fetch returned 403]** | Higher-order logic programming and lambda-tree syntax support binding and higher-order terms. | No continual sensory-learning result is established here. It is a reference for representation; review relevant implementation details before adopting higher-order search. |
| [Logic Tensor Networks](https://arxiv.org/abs/2012.13635), **[abstract]** | Differentiable many-valued first-order logic supports several learning and reasoning tasks in one formalism. | Graded truth is not uncertainty; adopting a score is not implementing LTN. Hypothesis: a consistent grounded relational interface may help this project. |
| [End-to-End Differentiable Proving](https://arxiv.org/abs/1705.11040), **[abstract]** | Differentiable symbol matching within backward chaining supports rule use/induction and multi-hop KB inference. | The reported KB setting is not this moving perceptual core. Soft matching and proof search are relevant mechanisms, not evidence of stable relearning here. |
| [TensorLog](https://arxiv.org/abs/1605.06523), **[abstract]** | Compiles logical clauses to differentiable message passing, composing inference across clauses. | Its stated complexity applies to its compilation/inference assumptions, not an unrestricted expression engine. It suggests a reference for finite relational execution. |

**[derived]** Exact argument swapping and finite existential enumeration can guarantee
algebraic consistency conditional on the supplied predicates. They cannot guarantee
correct sensory grounding. **[analogy]** Shared scoring may help learned transfer;
neither algebra nor the reviewed abstracts predicts retention during our relearning.

Known design hazards to address: search growth with free variables, spurious paths
from permissive soft matching, incompatible score scales, missing-evidence-as-false,
oracle candidate leakage, and confusing imposed identities with learned abstraction.
These are design risks **[derived/familiar]**, not reported failures of every cited
method. Cheaper alternatives are the unchanged g wrapper, exact finite truth-table
evaluation, and explicitly supplied inverse/composition rules. Implement those before
a general neural theorem prover. This stage adds an internal architectural test,
not a new logical calculus.

## Computation requirements

Follow [computation reporting](computation-reporting.md). For the proposed finite
baseline, let N be entities in the candidate domain, R candidate relations, V free
entity variables, A atoms, k chain length, C the cost of one neural atom score, and
P the trainable parameter count. Costs below exclude sensory encoding unless stated.

- Single missing entity: O(N C) time; relation completion: O(R C). Streaming top-k
  needs O(k) answer storage; returning every score needs O(N) or O(R).
- Naive enumeration: O(R^U N^V A C) for U free relation variables. This exponential
  bound is why the first pilot must be finite and bounded.
- One fixed binary relation table: O(N^2 C) scoring time and O(N^2) stored scores.
  From a fixed start, k-step max-min dynamic programming costs O(k N^2) after
  scoring, with O(N) working values; retaining witnesses costs additional storage.
  An all-start/all-end dense composition has different cost, up to O(k N^3).
  These estimates assume a simple chain and that particular aggregation, not
  arbitrary formulas or unlimited higher-order reasoning.
- A batch scoring B atoms requires O(B C) forward work; backpropagation has comparable
  order for a fixed network. Account separately for candidate expansion, encoder
  computation, activation workspace, optimizer O(P) state and retained feature caches.
  Enumeration is not free merely because parameter count falls when P is removed.
- Measure construction, feature encoding, candidate selection, training, inference,
  calibration and all rejected/tuning runs. Declare dtype, domains, batch sizes,
  parameters, budgets and enumeration limits before running. Include CPU latency
  and peak memory; no speed claims from asymptotics alone.
- Separate deterministic scientific results from runtime JSON, with revisions,
  seeds, hardware, wall/aggregate worker CPU, peak RSS/device memory, exposures,
  atom scores, query counts and stored bytes. Note missing historical cost data
  rather than retrofitting frozen reports. No numerical run was performed for this note.

## Requested handoff

The requested staging remains: **stage 1 is the inventory plus finite truth-table
fixture; stage 2 is the read-only g wrapper on captured seeds 300–303, with memory
present. Neither stage trains the model.** This document revision records the
review; it is not a record that either stage has run.

Return the inventory, fixture checks and checkpoint baseline with an explicit list
of supplied versus learned semantics and training-support mismatches. These results
should show what existing g can carry before an architecture decision. Then draft the matched
naming-only transfer contract informed by the checkpoint baseline. Keep the working
retrieval system and historical artifacts intact. Do not start a broad loss sweep,
silently supply sequence direction, or solve grounding with count IDs.
