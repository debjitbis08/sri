"""Long-term memory v1, post hoc: where is a pair held? For each query on A and B at each block's end, is it right by the readers alone, by memory alone, by both, by neither,
and what does the combined answer do with each? Pilot seeds only. Memory alone is memory's term with the readers' logits removed. Also: symbol-level agreement across renderings."""
import json
import multiprocessing as mp
import sys
import time

import numpy as np
import torch

sys.argv = [sys.argv[0], 'where'] + sys.argv[1:]
from experiments import long_term_memory as E
from src import long_term_memory as ltm

CONFIGS = {'in-budget-mlp': ('mlp', None, 0., .03, .9, 1), 'wide-k-sparse': ('k-sparse', 1024, .05, .003, .9, 1)}
WIDTH = ltm.width


def run(task):
    name, seed = task
    form, H, p, eta, beta, C = CONFIGS[name]
    torch.set_num_threads(1)
    ltm.width = (lambda f: H) if H else WIDTH
    mem = ltm.Memory(form, 32, 32, 'learned', eta, beta, C, p, seed=seed)
    cache = E.load_cache(seed)
    out = {}
    for b in E.BLOCKS:
        ltm.train_block(mem, cache['train'][b], 'frozen', 'frozen')
        ev = cache['eval'][b]
        for t in 'AB':
            nm = f'recall/{t}'
            lg = torch.from_numpy(ev[E.eval_name(nm, 'logits')])
            counts = torch.from_numpy(ev[E.eval_name(nm, 'counts')]).long()
            truth = torch.from_numpy(ev[E.eval_name(nm, 'truth')]).long()
            with torch.no_grad():
                term = mem.term(torch.from_numpy(ev[E.eval_name(nm, 'q_frozen')]), torch.from_numpy(ev[E.eval_name(nm, 'c_frozen')]))

            def right(s):
                return counts.gather(1, s.argmax(1, keepdim=True))[:, 0] == truth
            r, m, c = right(lg), right(term), right(lg + term)
            cell = {'readers_only': r & ~m, 'memory_only': m & ~r, 'both': r & m, 'neither': ~r & ~m}
            row = {k: {'share': float(v.float().mean()), 'combined_right': float(c[v].float().mean()) if v.any() else None} for k, v in cell.items()}
            E_ = E.AP.EVAL_RENDERS
            picks = counts.gather(1, term.argmax(1, keepdim=True))[:, 0].reshape(-1, E_).numpy()
            row['memory_symbol_agreement'] = float(np.mean([np.bincount(x).max() / E_ for x in picks]))
            row['expected_if_independent'] = float((r.float().mean() * m.float().mean()))
            row['term_scale'] = float(term.std(1).mean())
            row['reader_scale'] = float(lg.std(1).mean())
            row['combined'] = float(c.float().mean())
            out[f'{b}/{t}'] = row
    return name, seed, out


if __name__ == '__main__':
    t0 = time.perf_counter()
    tasks = [(n, s) for n in CONFIGS for s in E.PILOT_SEEDS]
    with mp.get_context('fork').Pool(4) as p:
        res = p.map(run, tasks)
    E.save_json(f'{E.REPORT}-where.json', {f'{n}/{s}': o for n, s, o in res})
    E.save_json(f'{E.REPORT}-where-runtime.json', E.runtime({'wall_s': time.perf_counter() - t0}))
    for name in CONFIGS:
        print(name)
        rows = [o for n, s, o in res if n == name]
        for key in ('4/A', '5/A', '5/B', '6/A', '6/B'):
            m = lambda f: round(float(np.mean([r[key][f] for r in rows])), 2)
            s = lambda k, f: round(float(np.mean([r[key][k][f] for r in rows])), 2)
            print(' ', key, 'readers-only', s('readers_only', 'share'), 'memory-only', s('memory_only', 'share'), 'both', s('both', 'share'), 'neither', s('neither', 'share'), '| combined keeps', s('readers_only', 'combined_right'), s('memory_only', 'combined_right'), s('both', 'combined_right'), s('neither', 'combined_right'),
                  '| both if independent', m('expected_if_independent'), '| agreement', m('memory_symbol_agreement'), '| scale term, readers', m('term_scale'), m('reader_scale'), '| combined', m('combined'))
