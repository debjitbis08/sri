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
- `docs/experiment-index.md`: running list of experiments, their contracts, results and reproduction commands.

## Why so minimal?

The first task is a reliable research harness. Its current monotone reachability law is a **control**, not our proposed interference algebra. Replacing it with genuinely compositional, possibly nonmonotone interactions is the next research step. No distinct replay store is introduced. The baseline stores only graph structure and transient activation.

## License and Copyright

Copyright © 2026 Debjit Biswas. All rights reserved.

This repository is publicly available for viewing and scientific discussion. No license is granted to copy, modify, redistribute, or commercially use the source code or documentation, except as permitted by applicable law or GitHub's Terms of Service.

For permissions, please contact the repository owner.
