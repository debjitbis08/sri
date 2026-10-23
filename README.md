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
- `docs/graded-cellular-v1-results.md`: nonlinear graded-cellular follow-up, recovery results and explicit scope discount; reproduce with `python -m experiments.graded_cellular`.

## Why so minimal?
The first task is a reliable research harness. Its current monotone reachability law is a **control**, not our proposed interference algebra. Replacing it with genuinely compositional, possibly nonmonotone interactions is the next research step. No distinct replay store is introduced. The baseline stores only graph structure and transient activation.
