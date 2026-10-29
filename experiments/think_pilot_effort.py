"""Thinking-effort curve for the thinking-loop pilot (docs/think-pilot-v1-spec.md): the end-of-training D-varied+rehearsal
checkpoints re-evaluated with K = 1, 2, 4 and 8 thinking steps (trained with K = 4). Uses the runners' evaluation code
unchanged (isolated matching, cross-sense matching, comparison) and the generic judgment revision.

Usage: .venv/bin/python -m experiments.think_pilot_effort pilot
"""
import gzip
import json
import sys
import time

import numpy as np
import torch

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import standard_baseline_compare as C
from experiments import generic_relations as G
from experiments import think_pilot as T

ROOT = X.ROOT
EFFORT = (1, 2, 4, 8)
MAIN = 'D-varied+rehearsal'
MEASURES = {
    'Chinese withheld, end': lambda ev: ev['iso']['chinese/withheld'],
    'Devanagari withheld, end': lambda ev: ev['iso']['devanagari_digits/withheld'],
    'Chinese paired, end': lambda ev: ev['iso']['chinese/paired'],
    'taps → collection': lambda ev: ev['senses']['taps_to_collection'],
    'taps → Devanagari form, withheld': lambda ev: ev['senses']['cross_sense']['taps_to_form/devanagari_digits/withheld'],
    'taps → Chinese form, withheld': lambda ev: ev['senses']['cross_sense']['taps_to_form/chinese/withheld'],
    'Comparison, Chinese forms': lambda ev: ev['senses']['comparison']['forms/chinese'],
    'Comparison, Devanagari forms': lambda ev: ev['senses']['comparison']['forms/devanagari_digits'],
    'Comparison, collections': lambda ev: ev['senses']['comparison']['collections'],
    'Judgment revision, withheld: revised': lambda ev: ev['extras']['context/withheld']['two_sided_revised'],
    'Succession pairwise, collections': lambda ev: ev['extras']['succession_pairwise']['collections'],
}


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    X.prepare(); P.prepare(); D.prepare()
    SB.install()  # the standard solvers for codes
    res = {}
    for s in C.SEEDS:
        ck = torch.load(ROOT / 'runs' / T.NAME / f'{MAIN}-seed{s}.pt', weights_only=False)
        enc, model = T.make_model(s)
        model.load_state_dict(ck['sri']); enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
        for k in EFFORT:
            model.steps = k
            ev = {'iso': P.evaluate(enc, model, ck['withheld'])['isolated'], 'senses': D.evaluate_senses(enc, model, ck['withheld'], False),
                  'extras': G.generic_extras(enc, model, ck['withheld'])}
            res[f'{s}/{k}'] = {name: float(get(ev)) for name, get in MEASURES.items()}
    out = {'seeds': C.SEEDS, 'effort': EFFORT, 'trained_steps': 4, 'per_seed_and_effort': res}
    with gzip.open(ROOT / 'reports' / 'think-pilot-v1-effort.json.gz', 'wt') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    (ROOT / 'reports' / 'think-pilot-v1-effort-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing; results in think-pilot-v1-effort.json.gz', 'module_sha256': X.nc.sha256(__file__),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')), 'command': ' '.join([sys.executable, *sys.argv]),
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}, indent=1) + '\n')
    print('| Measure | ' + ' | '.join(f'K = {k}' for k in EFFORT) + ' |\n| --- |' + ' --- |' * len(EFFORT))
    for name in MEASURES:
        print(f'| {name} | ' + ' | '.join(f"{np.mean([res[f'{s}/{k}'][name] for s in C.SEEDS]):.2f}" for k in EFFORT) + ' |')


if __name__ == '__main__':
    main()
