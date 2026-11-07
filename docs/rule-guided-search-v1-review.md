# Review of rule-guided search v1

2026-11-07. Reviewed draft `ce96295`, the composition-critic results and rule-application generator.
No contract changes, implementation, or experiment runs. Findings below are analytical unless marked otherwise.

## Findings before freezing

1. **P1 — Exclude answer relations from the hop graph. [read: contract/local world; derived]**
   Lines 38–45 allow all reader-trained relations as hops. That includes taught targets such as ancestor-of,
   bigger-than and grandparent-of, as well as completed kind-of memberships. Excluding their defining rules from
   critic training does not remove those edges. A direct target edge can answer without applying the supplied
   rule, especially on the oracle graph. Declare a per-task primitive-hop vocabulary, excluding target and
   intermediate derived relations and direct answer aliases. Specify immediate-kind reduction separately and
   identically for each graph condition. Keep frozen readers; this is an input/evaluation restriction.

2. **P1 — The permitted rule language exceeds the path representation. [derived]**
   Same-pair conjunction A(x,y) & B(x,y) -> H(x,y) needs two facts about the same endpoints, not a sequential
   a-to-b simple path. The no-return condition also excludes true self-relations derived through cycles.
   A pairwise scorer cannot represent arbitrary rule alternatives merely because each alternative is a chain:
   allowed sequences R,S,T and U,S,V expose local transitions accepting the invalid recombination R,S,V.
   Restrict the main pilot to a declared path/bigram-representable fragment, or add explicit proof state as a
   separate extension. Do not let structurally unsupported cases contaminate an acquisition score. Counting is
   only one limitation. For repeated P steps on an oracle graph with unit supports, the score is
   c(START,P) * c(P,P)^(k-1) * c(P,END), nonincreasing in k. Accepting exactly two and rejecting one is impossible;
   accepting length two does not mathematically force accepting length three at a fixed threshold.

3. **P1 — Proof rounds and path length are different training limits. [derived]**
   Lines 46 and 82 use depth <=2 and path length <=2 as though equivalent. Two rounds of binary composition can
   require four primitive hops. A training positive may therefore have no valid candidate even with perfect
   facts. Confirmed graphs can remove additional witnesses. The previous critic diagnostic explicitly reports
   learning spurious witnesses when valid ones are absent. Declare candidate coverage per positive, separate
   representable from unavailable proofs, and resolve absent-witness training rather than optimizing a spurious
   max. Specify empty-candidate handling. "Unlimited beam" is not exhaustive if top-k or learned c<0.5 pruning
   remains active; candidate generation must state which restrictions are disabled.

4. **P1 — Product scoring imposes an unmeasured length penalty. [derived]**
   Eleven supports of 0.9 give about 0.314 before any critic penalties. Thus even a perfect critic cannot pass
   the 0.5 answer threshold on such a valid chain. With eleven supports and twelve critic factors all at 0.95,
   the score is about 0.307. Graph membership above 0.5 is not sufficient. Measure score ceilings with valid
   proofs and a perfect critic, report ranking separately from acceptance, and freeze a justified calibration
   policy. Declare oracle support values and whether support products enter training as well as test scores.

5. **P1 — Beam search still loses witnesses. [derived]**
   W=8 and k=4 can remove the only useful branch; on unit-support oracle graphs top-k is an arbitrary tie-break.
   A graph adjacency list removes the learned proposer, not the proposal-recall problem. Include evaluation-only
   supplied valid proofs and exact/broader search on bounded cases. Report whether a proof exists, survives top-k,
   survives beam pruning, and passes scoring. Freeze tie-breaking and distinguish failed search from falsehood
   in diagnostics. Relabel the numerical ancestor/size predictions as hypotheses: the earlier critic check
   supplied valid witnesses, used different context/training, and did not establish recall of this search.

6. **P2 — Isolate graph, training and rule effects. [derived]**
   Define whether the oracle condition retrains the critic on oracle candidates or only swaps test graphs.
   Testing a critic trained on hallucinated witnesses against oracle facts does not isolate its capacity to apply
   rules on sound knowledge. Recompute exact closure for each actual graph: both-route agreement and applicability
   differ from v1's logic-reader graph, so the old reader-closure is not the new knowledge ceiling. The question
   arm has different training tasks; retain it as a legacy reference, and add a matched rule-context ablation or
   paired rule edits for causal attribution. Fresh base-symbol edits must consistently transform critic inputs;
   replacing frozen R_L vectors alone changes the learned interface rather than merely renaming a symbol.

7. **P2 — Complete coverage and cost declarations. [derived]**
   "Where the closure has them" does not ensure any deep synthetic examples. Freeze template frequencies,
   relation-family splits, accepted counts per depth, negatives and aggregation. Define depth consistently:
   grandparent is one proof round but two primitive hops. Account for graph/support storage O(R N^2), retained
   candidate lists, reader and optimizer bytes, rule-count-dependent encoder work, candidate-generation time and
   workspace, and beam selection. For Q questions, enumerating two-hop relation programs can require
   O(Q S^2 N) naive witness checks; if using all-pairs Boolean products, name their algorithm and workspace.
   Freeze graph-specific training counts, checkpoint hashes and smoke/selection cost reporting. A sentence saying
   runtime JSON is by phase does not replace the project's full reporting requirements.

## Source check

Searches on 2026-11-07: `MINERVA Go for a Walk and Arrive at the Answer reinforcement learning knowledge graphs
beam search Das 2018`; `TensorLog differentiable deductive database Cohen 2016 chain rules`.

- **[abstract]** [MINERVA](https://arxiv.org/abs/1711.05851) learns query-conditioned graph navigation by
  reinforcement learning, with an unknown destination. This proposal checks a specified endpoint using a local
  rule-conditioned critic. The abstract does not establish beam completeness or the proposed gate's performance.
- **[abstract]** [TensorLog](https://arxiv.org/abs/1605.06523) compiles clauses into factor graphs and differentiable
  message-passing functions, composed recursively. Its supplied inference structure differs from learning local
  transitions from answer labels. Differentiability alone does not make the two mechanisms equivalent.

The reuse direction is appropriate. The proposed main experiment should be described as learned rule-conditioned
path acceptance and bounded search on a specified rule fragment, not unrestricted modus ponens with variables.
