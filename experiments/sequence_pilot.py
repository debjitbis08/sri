"""Sequence prediction pilot (docs/sequence-pilot-v1-spec.md): the generic network plus one next-item predictor
P(c_prev, c_cur) conditioned on the two most recent items of a sequence. No action or relation tokens; the direction of
a counting episode (up or down) is inferred from its history.

Training is the transition pilot's update (generic Part D losses, same batches and draws) with its prediction loss
replaced: for every three consecutive steps t-1, t, t+1 of an episode whose step t+1 shows a collection, and for each
item kind present at both t-1 and t (forms, taps, or collections), P(c_{t-1}, c_t) must pick the collection seen at
t+1 among one collection per count, scored by -exp(scale) * squared distance. Withheld forms take part as history.

Evaluation: the generic pilot's measures, plus composition: from two real items of counts (n-1, n) (ascending) or
(n+1, n) (descending), P is rolled forward k = 1-3 steps and the result matched among the 9 collections of the same
evaluation item index, by g(same) and by distance.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.sequence_pilot pilot
Then: .venv/bin/python -m experiments.transition_pilot_analysis pilot sequence
"""
import numpy as np
import torch
import torch.nn.functional as F

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import generic_relations as G
from experiments import transition_pilot as TP
from src import generic_net as gen
from src import number_core as ncore
from src import number_systems_net as net

nc, ns = X.nc, X.ns
NAME = 'sequence-pilot-v1-' + X.MODE + ('-smoke' if X.SMOKE else '')
KS = (1, 2, 3)
ORIG_MATERIAL = D.batch_material


def make_model(seed):
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(net.SIZES[0]), 'sequence': ncore.MorseEncoder(net.SIZES[0])}
    return enc, gen.GenericSequenceNet(('image', 'sequence'))  # shared parameters start as in the generic pilot


def batch_material(lang_eps, ctx_eps, context):
    """D.batch_material plus each episode's steps in presentation order: per step, {kind: item index} and the count."""
    items, pairs, match = ORIG_MATERIAL(lang_eps, ctx_eps, context)
    seqs, pos = [], 0
    for ep in lang_eps:
        steps = []
        for n, e, fi in ep['steps']:
            st = {'form': pos}; pos += 1
            if e >= 0:
                st['coll'] = pos; pos += 1
            steps.append((n, st))
        seqs.append(steps)
    for ep in ctx_eps:
        seqs.append([(n, {'other': pos + 2 * j, 'coll': pos + 2 * j + 1}) for j, (n, *_) in enumerate(ep['steps'])])
        pos += 2 * len(ep['steps'])
    assert pos == len(items) and all(items[i][2] == n for s in seqs for n, st in s for i in st.values())
    return items, pairs, match, seqs


def sequence_loss(model, material, c, cand):
    items, _, _, seqs = material
    rows = []  # (prev item, current item, target collection item, target count)
    for steps in seqs:
        for t in range(1, len(steps) - 1):
            n_next, nxt = steps[t + 1]
            if 'coll' not in nxt:
                continue
            for kind in steps[t][1]:
                if kind in steps[t - 1][1]:
                    rows.append((steps[t - 1][1][kind], steps[t][1][kind], nxt['coll'], n_next))
    if not rows:
        return torch.zeros(())
    prev, cur, tgt = (torch.tensor([r[j] for r in rows]) for j in range(3))
    cnt = torch.tensor([r[3] for r in rows]) - 1
    C = cand[None].expand(len(rows), -1, -1).clone()
    C[torch.arange(len(rows)), cnt] = c[tgt]
    return F.cross_entropy(TP.predict_logits(model, model.predict(c[prev], c[cur]), C), cnt)


@torch.no_grad()
def composition(enc, model):
    code, E = G.codes_fn(enc, model), P.EVAL_PER
    kinds = {'collections': {n: P.DOTS['succ'][n] for n in ns.NUMBERS}, 'taps': {n: D.TAPS[('eval', n)] for n in ns.NUMBERS},
             **{f'forms/{L}': {n: X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]] for n in ns.NUMBERS} for L in P.LANGS}}
    colls = torch.stack([code(P.DOTS['succ'][n], 'image') for n in ns.NUMBERS], 1)
    out = {}
    for name, items in kinds.items():
        codes = {n: code(x, 'sequence' if name == 'taps' else 'image') for n, x in items.items()}
        for direction, sign in (('up', 1), ('down', -1)):
            for k in KS:
                hits = {'same': [], 'distance': []}
                for n in ns.NUMBERS:
                    m = n + sign * k
                    if not (1 <= n - sign <= 9 and 1 <= m <= 9):
                        continue
                    prev, cur = codes[n - sign], codes[n]
                    for _ in range(k):
                        prev, cur = cur, model.predict(prev, cur)
                    hits['same'].append((G.pair(model, cur, colls, 1).argmax(1) == m - 1).float().numpy())
                    hits['distance'].append((TP.predict_logits(model, cur, colls).argmax(1) == m - 1).float().numpy())
                for how, h in hits.items():
                    out[f'{name}/{direction}/{k}/{how}'] = float(np.concatenate(h).mean())
    return out


def evaluate_senses(enc, model, withheld, full):
    out = G.evaluate_senses(enc, model, withheld, full)
    if full:
        out['composition'] = composition(enc, model)
    return out


def main():
    SB.install()
    X.make_model = make_model
    TP.EXTRA_LOSS = sequence_loss
    P.evaluate, D.evaluate_senses, D.train_update, D.batch_material = G.evaluate, evaluate_senses, TP.train_update, batch_material
    D.ARMS = {a: D.ARMS[a] for a in G.ARMS}
    D.NAME = NAME
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, X.ROOT / 'runs' / NAME
    D.main()


if __name__ == '__main__':
    main()
