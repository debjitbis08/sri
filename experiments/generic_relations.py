"""Generic relations v1 pilot (docs/generic-relations-v1-spec.md): the Part D pilot on the standard network with the
relation-specific heads replaced by one generic comparator (src/generic_net.py).

Training is Part D's, unchanged: the matching, next and comparison losses call relation_logits, which is now the
generic comparator with a relation token. Evaluation measures that used the removed mechanisms get generic
counterparts, defined for any model with relation_logits (so they also apply to the standard network's checkpoints):
- revision: each candidate collection's score is g(same, form, candidate) plus the next judgments that place it after
  the grounded previous collection and before the grounded next one; replaces least squares through S;
- coherence: a counting run is scored by its summed next judgments; replaces successor residuals.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.generic_relations pilot [standard|sri]
Then: .venv/bin/python -m experiments.generic_relations_analysis pilot [standard|sri]
With 'sri', the SRI network (energy, modules, solvers) is kept and only its relation heads are replaced by the same
generic comparator on the core-inner code (src/generic_net.GenericSRI).
"""
import sys

import numpy as np
import torch

from experiments import number_systems as X
from experiments import number_construction_b1 as B
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from src import generic_net as gen
from src import number_core as ncore
from src import number_systems_net as net
from src import standard_net as std

nc, ns = X.nc, X.ns
ARCH = sys.argv[2] if len(sys.argv) > 2 else 'standard'
NAME = 'generic-relations-v1-' + ('sri-' if ARCH == 'sri' else '') + X.MODE + ('-smoke' if X.SMOKE else '')
ARMS = ('D-base', 'D-rehearsal', 'D-varied+rehearsal')
ORIG_EVALUATE, ORIG_SENSES = P.evaluate, D.evaluate_senses


def make_model(seed):
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(net.SIZES[0]), 'sequence': ncore.MorseEncoder(net.SIZES[0])}  # as X.make_model
    return enc, gen.GenericNet(('image', 'sequence'))  # projections draw the same initial values as the standard network's


def make_sri_model(seed):
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(net.SIZES[0]), 'sequence': ncore.MorseEncoder(net.SIZES[0])}
    return enc, gen.GenericSRI(('image', 'sequence'), torch.Generator().manual_seed(seed))  # SRI modules as in X.make_model


def codes_fn(enc, model):
    if hasattr(model, 'proj'):
        return lambda x, m: std.StandardSolver(model, m).code(enc[m](torch.from_numpy(np.ascontiguousarray(x))))
    return lambda x, m: model.split(B.Solver(model, m).solve(enc[m](torch.from_numpy(np.ascontiguousarray(x)))))['core_inner']


def pair(model, a, b, head):
    """Relation logits for every (row of a, row of b) within matching leading shapes: a (n, d), b (n, k, d) -> (n, k)."""
    n, k, d = b.shape
    return model.relation_logits(a[:, None].expand(-1, k, -1).reshape(-1, d), b.reshape(-1, d), head).reshape(n, k)


@torch.no_grad()
def generic_extras(enc, model, withheld, langs=P.LANGS):
    """Judgment-combined revision with grounded neighbours, and next-judgment succession, for any relation_logits model."""
    enc['image'].eval(); enc['sequence'].eval()
    code, E = codes_fn(enc, model), P.EVAL_PER
    rows = []
    for lang in langs:
        for n in ns.NUMBERS:
            q = code(X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]], 'image')
            n_swap = int(nc.stream('partA_swap', lang, n).choice([k for k in ns.NUMBERS if k != n]))
            qs = code(X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n_swap, 'idx')]], 'image')
            cc = code(np.stack([P.EVAL[(lang, n, i)][0] for i in range(E)]).reshape(-1, 28, 28), 'image').reshape(E, 9, -1)
            counts = np.stack([P.EVAL[(lang, n, i)][1] for i in range(E)])
            ctx = 0.
            for s in (-1, 1):
                if 1 <= n + s <= 9:
                    cs = code(np.stack([P.EVAL[(lang, n, i)][3][s] for i in range(E)]), 'image')
                    # previous neighbour: next(prev, candidate); next neighbour: next(candidate, nxt)
                    ctx = ctx + (pair(model, cs, cc, 0) if s < 0 else
                                 model.relation_logits(cc.reshape(-1, cc.shape[-1]), cs[:, None].expand(-1, 9, -1).reshape(-1, cc.shape[-1]), 0).reshape(E, 9))
            pick = lambda s: counts[np.arange(E), s.argmax(1).numpy()]
            iso = pair(model, q, cc, 1)
            rows.append({'lang': lang, 'n': n, 'n_swap': n_swap, 'withheld': n in withheld.get(lang, ()),
                         'iso': pick(iso), 'rev': pick(iso + ctx), 'swap': pick(pair(model, qs, cc, 1) + ctx)})
    out = {}
    for grp in ('paired', 'withheld'):
        rs = [r for r in rows if r['withheld'] and r['lang'] != P.LANGS[0]] if grp == 'withheld' else \
             [r for r in rows if not r['withheld'] and 2 <= r['n'] <= 8]
        if rs:
            num = np.concatenate([[r['n']] * E for r in rs]); nsw = np.concatenate([[r['n_swap']] * E for r in rs])
            cat = lambda k: np.concatenate([r[k] for r in rs])
            out[f'context/{grp}'] = {'isolated': float((cat('iso') == num).mean()), 'two_sided_revised': float((cat('rev') == num).mean()),
                                     'two_sided_swap_follows_context': float((cat('swap') == num).mean()),
                                     'two_sided_swap_follows_item': float((cat('swap') == nsw).mean())}
    succ = {}
    for name, get in (('collections', lambda n: P.DOTS['succ'][n]),) + tuple((L, (lambda L: lambda n: X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]])(L)) for L in langs):
        c = {n: code(get(n), 'image') for n in ns.NUMBERS}
        wins = [(model.relation_logits(c[n], c[n + 1], 0) > model.relation_logits(c[n], c[k], 0)).float().mean().item()
                for n in range(1, 9) for k in ns.NUMBERS if k not in (n, n + 1)]
        succ[name] = float(np.mean(wins))
    out['succession_pairwise'] = succ
    enc['image'].train(); enc['sequence'].train()
    return out


@torch.no_grad()
def generic_coherence(enc, model):
    """Part D's coherence runs (same streams), scored by summed next judgments along the run (higher = less coherent)."""
    code, E = codes_fn(enc, model), P.EVAL_PER
    kinds = {'collections': {n: P.DOTS['succ'][n] for n in ns.NUMBERS}, 'taps': {n: D.TAPS[('eval', n)] for n in ns.NUMBERS},
             **{f'forms/{L}': {n: X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]] for n in ns.NUMBERS} for L in (*P.LANGS, P.TRANSFER)}}
    det = {}
    for name, items in kinds.items():
        codes = {n: code(x, 'sequence' if name == 'taps' else 'image') for n, x in items.items()}
        rng = nc.stream('partD_coherence', name)
        scores, labels = [], []
        for start in range(1, 6):
            run = [codes[start + k] for k in range(5)]
            for bad in (False, True):
                seq = list(run)
                if bad:
                    pos = int(rng.integers(1, 4))
                    alt = [m for m in ns.NUMBERS if m != start + pos]
                    seq[pos] = torch.stack([codes[int(rng.choice(alt))][i] for i in range(E)])
                scores.append(-sum(model.relation_logits(seq[k], seq[k + 1], 0) for k in range(4)).numpy()); labels.append(np.full(E, bad))
        det[name] = B.auroc(np.concatenate(scores), np.concatenate(labels))
    return det


def evaluate(enc, model, withheld, langs=P.LANGS, full=False):
    out = ORIG_EVALUATE(enc, model, withheld, langs=langs, full=False)
    if full:
        out.update(generic_extras(enc, model, withheld, langs))
    return out


def evaluate_senses(enc, model, withheld, full):
    out = ORIG_SENSES(enc, model, withheld, full=False)
    if full:
        out['coherence_auroc_next_judgment'] = generic_coherence(enc, model)
    return out


def main():
    if ARCH == 'standard':
        SB.install()
        X.make_model = make_model
    elif ARCH == 'sri':
        X.make_model = make_sri_model
    else:
        raise SystemExit(f'unknown architecture {ARCH}')
    P.evaluate, D.evaluate_senses = evaluate, evaluate_senses
    D.ARMS = {a: D.ARMS[a] for a in ARMS}
    D.NAME = NAME
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, X.ROOT / 'runs' / NAME
    D.main()


if __name__ == '__main__':
    main()
