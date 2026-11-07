# Learning to apply relations

Discussion note, 2026-11-07, following commit `ef4be35`. Proposals only; no contract frozen or experiment run.

## Diagnosis and first experiment

**[read: local code and results]** `relations-as-objects-v1.md` reports near-perfect fitting of rho facts but
0.52 held-out inverse/synonym accuracy in the map arm, with or without rho facts. In
`experiments/relations_as_objects.py`, dropping instances does not reset the persistent relation code.
The refit diagnostic supports underconstraint of behaviour by rho fitting; it does not establish that
relational transfer is impossible.

**[derived; proposed remedy untested]** Train actual introduction episodes. Hide a training relation's
stored code and instances, construct a temporary code solely from its rho statements (including argument
direction, truth label, and partner codes), and score withheld instance answers through that code. Backpropagate
the answer loss through the introduction encoder and reader. No target-specific persistent code or optimizer
state may enter that path. Keep ordinary fact training for known partner relations. Start with inverse and
synonym families, whose answers are determined, before partially determined implication and symmetry tasks.

**[derived]** Repeated episodes over fixed identities can still memorize mappings. Hold out entire families,
vary relation aliases, and introduce fresh synthetic extensions across episodes. If partners vary too, give
their facts through a usable memory or teach them before the introduction; random labels cannot be inferred
without evidence. Include counterfactual episodes changing rho or partner evidence while keeping the query fixed.
Changing an inverse link should change answers on asymmetric pairs; changing an irrelevant fact should not.

## Application beyond a compact code

**[analogy; untested]** Compare introduction into one code with a shared query-conditioned neural reader over
rho statements and partner facts/answers. The latter can use a known relation without requiring a new code to
reproduce its entire extension. Supply generic access to facts and argument positions, not a hard-coded rho-to-
transpose/copy dispatch. A small first diagnostic can expose both argument orders and distractor facts; that
supplies a search space, not the meaning of each rho. Report oracle evidence selection separately from learned
retrieval. Later add multi-step dependencies only after single-step application works.

**[derived]** Score correct consequences, not only agreement: two constantly false readers can agree perfectly.
For implication and symmetry, evaluate only entailed consequences as transfer targets; absence of entailment
is not evidence of falsehood. A self-inverse statement alone cannot identify arbitrary symmetric instance facts.

## Source record

Search date 2026-11-07. Queries: `site.proceedings.mlr.press Finn 2017 model agnostic meta learning`;
`site.proceedings.neurips.cc Neural Logic Machines 2019 systematic generalization`;
`site.proceedings.neurips.cc Neural LP differentiable learning logical rules knowledge base reasoning 2017`.

- **[abstract]** [Finn et al., MAML (2017)](https://proceedings.mlr.press/v70/finn17a.html) trains parameters
  for rapid adaptation by gradient descent across tasks. Here, differentiating through rho-only adaptation
  and optimizing subsequent instance answers would be a MAML-style alternative. This source does not establish
  success for relation reification or this architecture. The proposed direct introduction encoder is amortized
  inference, not MAML.
- **[abstract]** [Yang et al., Neural LP (2017)](https://proceedings.neurips.cc/paper/2017/hash/0e55666a4ad822e0e34299df3591d979-Abstract.html)
  jointly learns structure and parameters of first-order rules with a differentiable model. It motivates
  separating learning a reusable relational computation from memorizing extensions. Its supplied logical
  machinery differs from this project's restriction against supplied meanings for rho tokens; it is related
  work, not evidence that the proposed generic reader will succeed.

**[derived]** Explicit copy/transpose is a cheap diagnostic upper bound, not a compliant learned solution.
Measure transfer both with oracle partner answers and learned partner answers to separate application errors
from source-fact errors. Compare against no-rho/shuffled-rho controls and a matched instance-only baseline.
Prefer the direct encoder first to avoid unrolled fitting costs. Before any run, declare full time/storage
orders and measure training, selection, adaptation, retrieval and query costs separately under
`computation-reporting.md`; no efficiency claim is made here.
