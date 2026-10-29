"""Thinking-loop pilot (docs/think-pilot-v1-spec.md): the sequence pilot's network (next-item predictor, no tokens for
actions) with its one-pass comparator replaced by a recurrent relational loop that thinks for K steps before answering
(src/generic_net.ThinkingSequenceNet).

Training is the sequence pilot's (Part D losses and draws, plus the next-item prediction loss), except that every
relational loss (matching, next, more) is averaged over the K = 4 thinking steps, so an answer is trained at every
step (deep supervision, as in recurrent relational networks). Evaluation is the sequence pilot's, at K = 4; the
analysis re-evaluates the checkpoints at K = 1, 2, 4 and 8.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.think_pilot pilot
Then: .venv/bin/python -m experiments.think_pilot_analysis pilot
"""
import numpy as np
import torch
import torch.nn.functional as F

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import generic_relations as G
from experiments import sequence_pilot as SP
from src import generic_net as gen
from src import number_core as ncore
from src import number_systems_net as net

nc, ns = X.nc, X.ns
NAME = 'think-pilot-v1-' + X.MODE + ('-smoke' if X.SMOKE else '')


def make_model(seed):
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(net.SIZES[0]), 'sequence': ncore.MorseEncoder(net.SIZES[0])}
    return enc, gen.ThinkingSequenceNet(('image', 'sequence'))  # other parameters start as in the sequence pilot


def stepwise(losses):
    return sum(losses) / len(losses)


def train_update(enc, model, opt, material, rng, compare):
    """The sequence pilot's update (same losses and draws) with each relational loss averaged over thinking steps."""
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
    d = D.SIZES[2]
    steps = model.relation_steps(c[fi][:, None].expand_as(C).reshape(-1, d), C.reshape(-1, d), 1)
    loss_match = stepwise([F.cross_entropy(s.reshape(len(match), 9), nn_) for s in steps])
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
    loss_next = stepwise([F.binary_cross_entropy_with_logits(s, target) for s in model.relation_steps(c[a2], c[b2], 0)])
    loss_cmp = torch.zeros(())
    if compare:
        pool = np.array(by_kind['coll'] + by_kind['taps'])
        u, v = pool[rng.integers(len(pool), size=4 * D.N_COMPARE)], pool[rng.integers(len(pool), size=4 * D.N_COMPARE)]
        cnt = np.array([it[2] for it in items])
        keep = np.flatnonzero(cnt[u] != cnt[v])[:D.N_COMPARE]
        u, v = u[keep], v[keep]
        tgt = torch.from_numpy((cnt[u] > cnt[v]).astype(np.float32))
        loss_cmp = stepwise([F.binary_cross_entropy_with_logits(s, tgt) for s in model.relation_steps(c[u], c[v], 2)])
    loss_seq = SP.sequence_loss(model, material, c, cand)
    loss = loss_match + loss_next + loss_cmp + loss_seq
    if not torch.isfinite(loss):
        raise nc.NumericalFailure('think pilot update')
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 5.0)
    opt.step()
    return [round(float(v.detach()), 4) for v in (loss_match, loss_next, loss_cmp, loss_seq)]


def main():
    SB.install()
    X.make_model = make_model
    P.evaluate, D.evaluate_senses, D.train_update, D.batch_material = G.evaluate, SP.evaluate_senses, train_update, SP.batch_material
    D.ARMS = {a: D.ARMS[a] for a in G.ARMS}
    D.NAME = NAME
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, X.ROOT / 'runs' / NAME
    D.main()


if __name__ == '__main__':
    main()
