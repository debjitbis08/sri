# Structural Resonance Intelligence — research starter

**Status: pre-theory / finite reference model.** This repository is an executable starting point for hypotheses about core–satellite intelligence, not an implemented intelligent model.

## Reproduce

Requires Python >= 3.10. Studies up to graded cellular v1 use only the standard
library. Core mediation v1 additionally needs the pinned numpy and scipy:

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
python -m unittest discover -s tests -v
python -m experiments.baseline
```

The second command overwrites `reports/baseline.json` deterministically.

## Layout

- `docs/foundations.md`: definitions, analytic proofs for the baseline, limitations and proof obligations.
- `src/model.py`: finite weighted core–satellite reachability model.
- `tests/`: executable invariants.
- `experiments/baseline.py`: reproducible parameter sweep and endogenous seed sampling.
- `reports/baseline.json`: frozen output of the sweep.
- `docs/mediation-example.md`: worked collective role-binding example, its controls and limits; reproduce with `python -m experiments.mediation`.
- `docs/comparison-v1-results.md`: four-family modeling comparison, evidence grades and limitations; reproduce with `python -m experiments.model_comparison`.
- `docs/core-mediation-v1-contract.md`: declared contract for discrete core-mediated compatibility, realizability and certified truncation.
- `docs/core-mediation-v1-results.md`: phase B realizability results (pairwise star versus ternary versus agreement); reproduce with `.venv/bin/python -m experiments.core_mediation` (about 18 minutes).
- `docs/core-mediation-v1-phase-c-results.md`: structured internals, recovery, input/output; robust codes make readout-only coupling inert. Reproduce with `.venv/bin/python -m experiments.core_structured` after phase B.
- `docs/core-graded-v1-contract.md`, `docs/core-graded-v1-results.md`: graded unit-level coupling; one worked example where recovery and input-driven core participation coexist. Reproduce with `.venv/bin/python -m experiments.core_graded`.
- `docs/core-graded-basins-contract.md`, `docs/core-graded-basins-results.md`: exact basins, tie and terminal propositions, and the joint-success window; reproduce with `.venv/bin/python -m experiments.core_graded_basins`.
- `docs/learned-compatibility-v1-contract.md`, `docs/learned-compatibility-v1-results.md`: learned core compatibility, generalization to unseen combinations, set-valued responses; reproduce with `.venv/bin/python -m experiments.learned_compatibility`.
- `docs/selective-satellite-mediation-v1-contract.md`, `docs/selective-satellite-mediation-v1-results.md`: learned context-gated repair under immutable domains; exact version space, transfer to an unseen context, matched controls (the positive-only ablation and selection prior are reported limits). Reproduce with `.venv/bin/python -m experiments.selective_mediation`.
- `docs/learned-reciprocal-participation-v1-contract.md`, `docs/learned-reciprocal-participation-v1-results.md`: signed-energy input/core/satellite dynamics; phase-A gate fails (input revision never changes responses for the declared teachers), so no learning was run. Reproduce with `.venv/bin/python -m experiments.reciprocal_participation`.
- `docs/shared-input-mediation-v1-contract.md`, `docs/shared-input-mediation-v1-results.md`: causal chain A → input → B through a shared or internally coupled input; 2 of 6 declared teachers (S1, C2) pass the full gate, mechanism only, no learning. Reproduce with `.venv/bin/python -m experiments.shared_input_mediation`.
- `docs/learned-input-mediation-v1-contract.md`, `docs/learned-input-mediation-v1-results.md`: learning couplings from outcomes and checking the learned mechanism on a held-out intervention battery; exact learners identify S1/C2 mechanisms; for a negative control, outputs alone leave the mechanism uncertain (false-chain survivors beside the correct ones). Reproduce with `.venv/bin/python -m experiments.mechanism_learning`.
- `docs/local-learning-supervision-v1-contract.md`, `docs/local-learning-supervision-v1-results.md`: fixed local rule under four supervision conditions; clamps with external readout or 12 extra mixed-input queries suffice for exact identification, but the local learner reaches the mechanism reliably only with internal readouts. Reproduce with `.venv/bin/python -m experiments.local_supervision`.
- `docs/local-update-diagnosis-v1-contract.md`, `docs/local-update-diagnosis-v1-results.md`: why the block learner fails under IB supervision (mixed causes: inexact single commits, preservation conflicts, restricted edits, plateau barriers) and three single-property variants; none suffices for both teachers, and history-aware selection (a replay comparator) suffices for C2 only. Reproduce with `.venv/bin/python -m experiments.update_diagnosis`.
- `docs/graded-cellular-v1-results.md`: nonlinear graded-cellular follow-up, recovery results and explicit scope discount; reproduce with `python -m experiments.graded_cellular`.

## Why so minimal?

The first task is a reliable research harness. Its current monotone reachability law is a **control**, not our proposed interference algebra. Replacing it with genuinely compositional, possibly nonmonotone interactions is the next research step. No distinct replay store is introduced. The baseline stores only graph structure and transient activation.

## License and Copyright

Copyright © 2026 Debjit Biswas. All rights reserved.

This repository is publicly available for viewing and scientific discussion. No license is granted to copy, modify, redistribute, or commercially use the source code or documentation, except as permitted by applicable law or GitHub's Terms of Service.

For permissions, please contact the repository owner.
