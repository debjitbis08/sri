# Structural Resonance Intelligence — research starter

**Status: pre-theory / finite reference model.** This repository is an executable starting point for hypotheses about core–satellite intelligence, not an implemented intelligent model.

## Reproduce
Requires Python >= 3.10, standard library only.

```bash
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

## Why so minimal?
The first task is a reliable research harness. Its current monotone reachability law is a **control**, not our proposed interference algebra. Replacing it with genuinely compositional, possibly nonmonotone interactions is the next research step. No distinct replay store is introduced. The baseline stores only graph structure and transient activation.
