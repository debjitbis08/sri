"""Transition network pilot (docs/transition-pilot-v1-spec.md): the generic relations pilot's standard network plus one
generic action-conditioned transition network T(c, action), trained to predict the next item's code.

Training is the generic pilot's (Part D losses through the generic comparator g, same batches and draws) plus a
prediction loss: for every counting transition u -> v in the batch (v one count above u), T(c_u, add) must pick the
collection seen at v's step among one collection per count (the batch's shared candidates, with the true slot replaced
by that collection), scored by -exp(scale) * squared distance in code space; T(c_v, remove) likewise picks u's
collection. Sources are any kind (forms, collections, taps), withheld forms included; targets are visible collections
only. The loss draws no random numbers, so every batch matches the generic pilot.

Evaluation: the generic pilot's measures, plus composition: T applied k = 1-3 times to an item's code, the result
matched to the 9 collections of one evaluation item set, by g(same) and by distance.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.transition_pilot pilot
Then: .venv/bin/python -m experiments.transition_pilot_analysis pilot
"""
import numpy as np
import torch
import torch.nn.functional as F

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import generic_relations as G
from src import generic_net as gen
from src import number_core as ncore
from src import number_systems_net as net

nc, ns = X.nc, X.ns
NAME = 'transition-pilot-v1-' + X.MODE + ('-smoke' if X.SMOKE else '')
KS = (1, 2, 3)


def make_model(seed):
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(net.SIZES[0]), 'sequence': ncore.MorseEncoder(net.SIZES[0])}
    return enc, gen.GenericTransitionNet(('image', 'sequence'))  # shared parameters start as in the generic pilot


def predict_logits(model, pred, cand):
    """pred (n, d), cand (n, 9, d) -> (n, 9) logits -exp(scale) * ||pred - cand||^2."""
    return -model.scale.exp() * (pred[:, None] - cand).pow(2).sum(-1)


def transition_loss(model, items, pairs, match, c, cand):
    coll_of = {i: i for i, it in enumerate(items) if it[0] == 'coll'}
    coll_of.update({a: b for a, b, _ in match if items[b][0] == 'coll'})
    rows = []  # (source, action, target collection item, target count)
    for u, v in pairs:
        if v in coll_of:
            rows.append((u, 'add', coll_of[v], items[v][2]))
        if u in coll_of:
            rows.append((v, 'remove', coll_of[u], items[u][2]))
    loss, n = torch.zeros(()), 0
    for action in gen.ACTIONS:
        r = [x for x in rows if x[1] == action]
        if not r:
            continue
        src = torch.tensor([x[0] for x in r]); tgt = torch.tensor([x[2] for x in r]); cnt = torch.tensor([x[3] for x in r]) - 1
        C = cand[None].expand(len(r), -1, -1).clone()
        C[torch.arange(len(r)), cnt] = c[tgt]
        loss = loss + F.cross_entropy(predict_logits(model, model.transition(c[src], action), C), cnt, reduction='sum')
        n += len(r)
    return loss / max(n, 1)


def extra_transition_loss(model, material, c, cand):
    return transition_loss(model, *material[:3], c, cand)


EXTRA_LOSS = extra_transition_loss  # replaced by runners that reuse train_update with another prediction loss


def train_update(enc, model, opt, material, rng, compare):
    """D.train_update (copied; same losses and draws) with EXTRA_LOSS (default: the transition loss) added to the total."""
    items, pairs, match = material[:3]
    solvers = {'image': P.Coupled(model)}
    if any(it[0] == 'taps' for it in items):
        solvers['sequence'] = D.Sequence(model)
    c = D.encode_items(enc, model, items, solvers)
    ci = rng.integers(len(P.DOTS['train']), size=9)
    cand = solvers['image'].solve(enc['image'](torch.from_numpy(np.stack([P.DOTS['train'][ci[k], k] for k in range(9)]))))[:, P.INNER]
    fi = torch.tensor([m[0] for m in match]); cj = torch.tensor([m[1] for m in match]); nn_ = torch.tensor([m[2] for m in match]) - 1
    C = cand[None].expand(len(match), -1, -1).clone()
    C[torch.arange(len(match)), nn_] = c[cj]
    logits = P.same_logits(model, c[fi][:, None].expand_as(C).reshape(-1, D.SIZES[2]), C.reshape(-1, D.SIZES[2])).reshape(len(match), 9)
    loss_match = F.cross_entropy(logits, nn_)
    a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
    kinds = [items[p[0]][0] for p in pairs]
    by_kind = {k: [i for i, it in enumerate(items) if it[0] == k] for k in ('form', 'coll', 'taps')}
    na, nb = [], []
    for k in kinds:
        pool = by_kind[k]
        for _ in range(20):
            u, v = rng.choice(pool, 2)
            if items[v][2] != items[u][2] + 1:
                na.append(u); nb.append(v)
                break
    a2 = torch.cat([a, torch.tensor(na, dtype=torch.long)]); b2 = torch.cat([b, torch.tensor(nb, dtype=torch.long)])
    target = torch.cat([torch.ones(len(a)), torch.zeros(len(na))])
    loss_next = F.binary_cross_entropy_with_logits(model.relation_logits(c[a2], c[b2], 0), target)
    loss_cmp = torch.zeros(())
    if compare:
        pool = np.array(by_kind['coll'] + by_kind['taps'])
        u, v = pool[rng.integers(len(pool), size=4 * D.N_COMPARE)], pool[rng.integers(len(pool), size=4 * D.N_COMPARE)]
        cnt = np.array([it[2] for it in items])
        keep = np.flatnonzero(cnt[u] != cnt[v])[:D.N_COMPARE]
        u, v = u[keep], v[keep]
        loss_cmp = F.binary_cross_entropy_with_logits(model.relation_logits(c[u], c[v], 2), torch.from_numpy((cnt[u] > cnt[v]).astype(np.float32)))
    loss_t = EXTRA_LOSS(model, material, c, cand)
    loss = loss_match + loss_next + loss_cmp + loss_t
    if not torch.isfinite(loss):
        raise nc.NumericalFailure('transition pilot update')
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 5.0)
    opt.step()
    return [round(float(v.detach()), 4) for v in (loss_match, loss_next, loss_cmp, loss_t)]


@torch.no_grad()
def composition(enc, model):
    """T^k from each item's code (k = 1-3, add and remove), matched among the 9 collections of the same evaluation item
    index, by g(same) and by distance. Kinds: collections, taps, forms of each language."""
    code, E = G.codes_fn(enc, model), P.EVAL_PER
    kinds = {'collections': {n: P.DOTS['succ'][n] for n in ns.NUMBERS}, 'taps': {n: D.TAPS[('eval', n)] for n in ns.NUMBERS},
             **{f'forms/{L}': {n: X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]] for n in ns.NUMBERS} for L in P.LANGS}}
    colls = torch.stack([code(P.DOTS['succ'][n], 'image') for n in ns.NUMBERS], 1)  # (E, 9, d)
    out = {}
    for name, items in kinds.items():
        codes = {n: code(x, 'sequence' if name == 'taps' else 'image') for n, x in items.items()}
        for action, sign in (('add', 1), ('remove', -1)):
            for k in KS:
                hits = {'same': [], 'distance': []}
                for n in ns.NUMBERS:
                    m = n + sign * k
                    if not 1 <= m <= 9:
                        continue
                    s = codes[n]
                    for _ in range(k):
                        s = model.transition(s, action)
                    hits['same'].append((G.pair(model, s, colls, 1).argmax(1) == m - 1).float().numpy())
                    hits['distance'].append((predict_logits(model, s, colls).argmax(1) == m - 1).float().numpy())
                for how, h in hits.items():
                    out[f'{name}/{action}/{k}/{how}'] = float(np.concatenate(h).mean())
    return out


def evaluate_senses(enc, model, withheld, full):
    out = G.evaluate_senses(enc, model, withheld, full)
    if full:
        out['composition'] = composition(enc, model)
    return out


def main():
    SB.install()
    X.make_model = make_model
    P.evaluate, D.evaluate_senses, D.train_update = G.evaluate, evaluate_senses, train_update
    D.ARMS = {a: D.ARMS[a] for a in G.ARMS}
    D.NAME = NAME
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, X.ROOT / 'runs' / NAME
    D.main()


if __name__ == '__main__':
    main()
