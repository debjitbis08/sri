# Review of rule application v1

2026-11-07. Reviewed draft commit `3105e68`, the everyday-world source, and the cited primary sources below.
No contract edits or experiment runs. Findings are recommendations before freezing.

1. **[P1; read: local source; derived consequence] Category closure is already supplied.**
   Contract line 71 takes all true `kind-of` facts as input. `src/everyday_truth_v2.py` defines `kind_chain`
   through all supercategories and emits every membership as a true `kind-of` fact (lines 96–102).
   Thus `kind(x,c) -> isa(x,c)` obtains all those answers at depth 1; the proposed depth 1–4 test disappears.
   Use only immediate memberships, with an explicit identical input-domain policy for oracle and reader,
   or derive `kind-of` from a deliberately restricted immediate-membership relation. Specify which world's
   labels are comparable: existing `is-a` also covers words and categories, not just this animal hierarchy.

2. **[P1; derived] Network depth is not proof depth.** Contract lines 152–153 assume one rule application per
   layer without imposing or supervising that restriction. Triangular updates combine two previously computed
   path summaries; their potential path span can double per layer. Four layers are therefore not excluded from
   handling paths of length 5–6. This is a capacity observation, not a claim they will learn that algorithm.
   Relabel the numerical prediction as a hypothesis. Report fixed versus tied at equal four-step evaluation,
   then report the tied model's additional-loop gains separately. The 16-versus-4 comparison changes computation
   as well as tying. In-distribution 0.97 is also a hypothesis, not a deduction from fresh identities.

3. **[P1; derived] Base-fact precision/recall cannot isolate application error.** For reader inputs, also run
   the symbolic oracle on exactly the reader's supplied fact set. Compare neural answers to that closure
   (application fidelity), that closure to true-fact closure (knowledge-error propagation), and neural answers
   to world labels (end-to-end performance). A missing redundant edge may change no conclusions, while one
   spurious edge can connect whole components. Prediction 5 cannot be derived from aggregate hop metrics.
   This symbolic diagnostic remains outside the deployed neural system.

4. **[P1; derived; proposed control untested] The reads do not establish that told rules cause answers.**
   Fresh symbols prevent identity memorization but not shortcuts from topology or the rule generator. Add
   paired tests with the same facts, symbols and query but a changed head, argument order or required body
   relation; require correct answers on both sides when the oracle says they differ. Include irrelevant-rule
   changes as invariance controls and body-order/variable-renaming checks if claiming variable binding beyond
   canonical templates. Connected negatives only of path length <=3 leave long-path shortcuts untested; add
   depth-matched wrong-relation and broken-conjunction negatives. A rule-blind control is useful too.

5. **[P2; read: primary-source sections] Correct the prior-work account.**
   The Edge Transformer paper explicitly ties layer weights by default; saying it is not looped is misleading.
   Distinguish weight sharing from extrapolating the number of iterations. Fan et al. provide evidence under
   training with supplied per-example stopping counts and repeated input injection. They do not derive success
   for randomly sampled 2–4 loops in this new task. These differences should be explicit before freezing.

6. **[P2; derived] Cost accounting omits the expensive control and temporary memory.**
   Attention over n^2 pair tokens has O(n^4 d) attention work per loop, versus O(n^3 d) for the triangle.
   Conventional materialized attention stores O(H n^4) scores for pair attention versus O(H n^3) for triangular
   attention, plus O(H n^2 r) for rule attention; H is head count. Training retains loop activations unless
   recomputed, and implementation-dependent triangle values can require additional temporary storage.
   Specify actual kernels, dtype, optimizer/persistent state, activation workspace, and arm-specific MACs and
   timing estimates. Include rejection sampling/closure generation costs, smoke/selection costs, and test-set
   storage, not only n^2 recurrent states. No speed claim or measured cost is made by this review.

7. **[P2; derived] Freeze generator coverage and metric definitions.**
   Declare rule-template probabilities, seeding of recursive predicates, rejection limits, accepted-depth
   quotas, and accepted/rejected counts. Pure self-recursive rules with no seed or incoming derivation are
   vacuous. Planted chains do not guarantee shortest proof depth when other facts/rules give shortcuts.
   Require oracle-verified counts per depth and held-out orientation, and ensure the latter is necessary to
   the scored consequence rather than accompanied by a training-orientation proof. Specify the extra size/ID
   test-set counts, the negative pool used in each depth's balanced accuracy, macro/micro aggregation,
   empty-bucket handling, thresholds, and shared problem seeds across arms. Rejecting all deeper training
   graphs can change topology distributions, so report accepted distributions by template and depth.

## Source record and interpretation

Opened primary URLs directly on 2026-11-07, then read the indicated HTML sections (no new search query).

- **[read: section 3, Tied & Untied Edge Transformer]**
  [Bergen et al. 2021](https://arxiv.org/html/2112.00578v1): default tied weights are explicitly stated.
  The project adds told formula rules and fresh symbols. This source does not establish success for that extension.
- **[read: sections 4.1–4.2]**
  [Fan et al., Looped Transformers](https://arxiv.org/html/2409.15647v5): training includes a predefined number
  of steps per example, without intermediate-output targets, and the architecture injects the input each loop.
  These are materially different from random stopping counts and base facts supplied only initially.
  Transfer of their length-generalization findings here is a hypothesis.
- **[abstract]** [Clark et al. 2020](https://arxiv.org/abs/2002.05867): the abstract does report 95%+ scores
  on deeper chaining. The draft correctly discloses that per-depth numbers were not checked; no stronger
  comparison is justified by this review.

The design is a useful test of consequence-supervised application of supplied Horn-style rules. It does not
test discovering those rules, full first-order inference, or open-world truth. This scope is compatible with
the stated first pilot, provided conclusions preserve it.
