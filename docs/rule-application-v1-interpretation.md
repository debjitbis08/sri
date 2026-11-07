# Interpreting the rule application v1 negative result

2026-11-07, following results commit `ac74065`. Discussion only; no new contract or run.

**[read: local results and training code]** The first failure is acquisition: even at trained depths,
held-out synthetic performance is weak, relevant rule edits are poorly followed, and fresh identity vectors
change many answers. Clean taught problems show a narrower success at depths 1–2 in the untied model.
Additional tied iterations degrade performance. These observations do not identify the learned algorithm
or establish an architectural impossibility.

**[derived]** Missing iteration pressure is a plausible cause, not an established leading cause. Other
unseparated explanations include insufficient optimization, interference between rule forms, and difficulty
matching fresh symbols through compressed rule/pair representations. The loss includes base-fact reconstruction,
so overall training accuracy is not a direct measure of learning new consequences. Inspect derived-only and
per-template performance when judging acquisition.

**[proposal; untested]** Before another full depth-generalization run, use a small staged diagnostic:

1. Establish near-exact one-step application separately for copy, inverse, same-pair conjunction and composition,
   with fresh symbols and counterfactual rule edits. Include matched negative examples missing one conjunct or
   using the wrong relation. Use a fixed small training set as an optimization sanity check, then fresh examples.
2. If these work separately, mix them to identify interference. If they fail separately, additional recurrence
   cannot be expected to repair that; diagnose symbol matching, binding and optimization first.
3. Once acquisition works, compare final-only supervision to intermediate consequence supervision using the
   same architecture, problems and declared compute budgets. Teach preservation of established facts and the
   newly derivable consequences, then test feeding predictions into subsequent iterations. Distinguish oracle
   intermediate inputs from the model's own states to expose error accumulation.

The symbolic oracle would supply training labels, not execute inference at deployment. This supplies more
procedural information than v1 and must be reported as a changed teaching regime. Intermediate non-entailment
means "not derived yet", not final falsity. Training only on depths <=2 still does not guarantee transfer to
deeper states. Stop/go criteria and computation accounting must be frozen before a new run.

**[read: primary source, sections 4.1–4.2]** Reopened
[Fan et al., Looped Transformers](https://arxiv.org/html/2409.15647v5) directly on 2026-11-07.
They supply a required stopping count per training example but explicitly do not supply intermediate-state
targets. Thus per-step consequence supervision is a distinct proposed intervention, not what that cited study
demonstrated was necessary. No claim that it guarantees iteration is supported here.

**[read: local results]** Reader-closure separates a second problem: parent and size knowledge can support long
chains substantially better than the network realizes, while category knowledge and immediate-membership
reduction are badly damaged by false subkind links. These are separate application and evidence failures.
