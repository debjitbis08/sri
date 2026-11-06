"""Composition critic check v1 (docs/composition-critic-check-v1.md): can a step-local critic, learned from question
labels only, tell valid proof programs from spurious ones? Variants: pairwise, carried summary, pairwise + look-back.
Frozen phase-1 readers of conflict search v1; no walker training.

Usage: EV_WORKERS=4 .venv/bin/python -m experiments.composition_critic_check        (rerun to resume)
       .venv/bin/python -m experiments.composition_critic_check summary
"""
import collections
import json
import os
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import conflict_search as CS
from experiments import conflict_search_diag as D
from experiments import durable as DUR
from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from src import everyday_truth_v2 as T2

ROOT, nc = V2.ROOT, V2.nc
NAME = 'composition-critic-check-v1'
CONTRACT = ROOT / 'docs' / f'{NAME}.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
SEEDS = (450, 451, 452, 453)
VARIANTS = ('pairwise', 'summary', 'lookback')
STEPS, LR, EMB, HID = 1500, 3e-3, 16, 32
WALKS, LONG = 30, (3, 4, 5)
EVAL_TESTS = ('P6-grandparent', 'P6-colour', 'L1-kind', 'L1-anc', 'L1-anc-d45', 'L1-size', 'L1-size-le5', 'L2-named')


class Critic(nn.Module):
    def __init__(self, n_tok, n_rel, variant):
        super().__init__()
        self.variant = variant
        self.START, self.END, self.PAD = n_tok, n_tok + 1, n_tok + 2
        self.tok = nn.Embedding(n_tok + 3, EMB)
        self.q = nn.Embedding(n_rel, EMB)
        if variant == 'summary':
            self.h0 = nn.Linear(EMB, HID)
            self.gru = nn.GRU(EMB, HID, batch_first=True)
            inp = EMB + HID + EMB
        else:
            inp = 3 * EMB
            if variant == 'lookback':
                self.wq, self.wk, self.wv = nn.Linear(2 * EMB, EMB), nn.Linear(EMB, EMB), nn.Linear(EMB, EMB)
                inp += EMB + 1
        self.mlp = nn.Sequential(nn.Linear(inp, HID), nn.Tanh(), nn.Linear(HID, 1))

    def forward(self, q, toks):
        """q (U,), toks (U, L) = START, steps..., END, PAD...; returns log score (U,) = sum of log local checks."""
        U, L = toks.shape
        e = self.tok(toks)
        qe = self.q(q)
        prev, cur = e[:, :-1], e[:, 1:]                                  # checks at positions 1..L-1
        valid = (toks[:, 1:] != self.PAD)
        qx = qe[:, None].expand(-1, L - 1, -1)
        if self.variant == 'summary':
            h0 = torch.tanh(self.h0(qe))[None]
            out, _ = self.gru(e[:, :-1], h0)                             # state after consuming tokens 0..p-1
            x = torch.cat([qx, out, cur], 2)
        else:
            x = torch.cat([qx, prev, cur], 2)
            if self.variant == 'lookback':
                k, v = self.wk(e), self.wv(e)
                qq = self.wq(torch.cat([qx, cur], 2))                    # (U, L-1, EMB), position p = index+1
                att = qq @ k.transpose(1, 2) / EMB ** .5                 # (U, L-1, L)
                p = torch.arange(1, L)[:, None]
                j = torch.arange(L)[None]
                allowed = (j >= 1) & (j <= p - 2)                        # real steps before the previous one
                allowed = allowed[None] & (toks[:, None, :] < self.START)
                any_ = allowed.any(2, keepdim=True)
                w = torch.softmax(att.masked_fill(~allowed, -1e9), 2) * any_
                x = torch.cat([x, w @ v, any_.float()], 2)
        logit = self.mlp(x)[..., 0]
        return (F.logsigmoid(logit) * valid).sum(1)


# ---------------------------------------------------------------- candidates

class Graph:
    def __init__(self, st, conf, permitted):
        rels = list(st.relations)
        self.rels = rels
        self.pid = [rels.index(x) for x in permitted]
        ne = len(st.entities)
        self.n_tok = 2 * len(self.pid)
        self.C = np.zeros((self.n_tok, ne, ne), bool)
        slot = {r: k for k, r in enumerate(self.pid)}
        for x, r, y in conf:
            if r in slot:
                self.C[2 * slot[r], x, y] = True
                self.C[2 * slot[r] + 1, y, x] = True
        self.Cf = self.C.astype(np.float32)
        self.adj = [np.argwhere(self.C[:, x, :]) for x in range(ne)]    # rows (tok, y)
        self.tok_of = {(rels[r], fwd): 2 * k + (0 if fwd else 1) for k, r in enumerate(self.pid) for fwd in (True, False)}

    def tok_name(self, t):
        return self.rels[self.pid[t // 2]], t % 2 == 0

    def short(self, a, b):
        out = [(int(t),) for t in np.flatnonzero(self.C[:, a, b])]
        A = self.Cf[:, a, :].copy()
        A[:, [a, b]] = 0
        P2 = (A @ self.Cf[:, :, b].T) > 0
        out += [(int(t1), int(t2)) for t1, t2 in np.argwhere(P2)]
        return out

    def sampled(self, a, b, rng):
        out = set()
        for L in LONG:
            for _ in range(WALKS):
                x, path, toks = a, {a, b}, []
                for _ in range(L - 1):
                    nb = self.adj[x]
                    nb = nb[~np.isin(nb[:, 1], list(path))] if len(nb) else nb
                    if not len(nb):
                        break
                    t, y = nb[rng.integers(len(nb))]
                    toks.append(int(t))
                    path.add(int(y))
                    x = int(y)
                if len(toks) == L - 1:
                    for t in np.flatnonzero(self.C[:, x, b]):
                        out.add(tuple(toks) + (int(t),))
        return list(out)


def valid_program(G, rel, prog):
    return D.accepts(rel, [D.token(*G.tok_name(t)) for t in prog])


def encode(progs, n_tok):
    L = max(len(p) for p in progs) + 2
    toks = np.full((len(progs), L), n_tok + 2, dtype=np.int64)
    toks[:, 0] = n_tok
    for i, p in enumerate(progs):
        toks[i, 1:len(p) + 1] = p
        toks[i, len(p) + 1] = n_tok + 1
    return torch.as_tensor(toks)


class Bank:
    """Unique (q, program) pairs and row memberships."""

    def __init__(self):
        self.key, self.items, self.mem_row, self.mem_item = {}, [], [], []

    def add(self, row, q, prog):
        k = (q, prog)
        if k not in self.key:
            self.key[k] = len(self.items)
            self.items.append(k)
        self.mem_row.append(row)
        self.mem_item.append(self.key[k])

    def tensors(self, n_tok):
        q = torch.tensor([k[0] for k in self.items])
        return q, encode([k[1] for k in self.items], n_tok), torch.tensor(self.mem_row), torch.tensor(self.mem_item)


def row_max(ls, mem_row, mem_item, n_rows):
    out = torch.full((n_rows,), -1e9)
    return out.scatter_reduce(0, mem_row, ls[mem_item], 'amax')


# ---------------------------------------------------------------- run

def run_seed(seed):
    torch.set_num_threads(1)
    cost = {}
    t0 = time.process_time()
    st, _ = T2.build()
    train_rows, _ = CS.splits(st, seed)
    rels = list(st.relations)
    world = CS.World(st)
    permitted = D.permitted_relations(st, train_rows)
    base = D.load_model(st, D.SRC_RUNS / f'phase1-seed{seed}.pt', full=False)
    pool0, conf, gstats = D.confirmed_graph(base)
    G = Graph(st, conf, permitted)
    masks = D.test_masks(st, train_rows)
    cost['graph'] = {'cpu_seconds': time.process_time() - t0, **gstats}

    # training candidates
    t0 = time.process_time()
    tr_idx = masks['withheld-fit']
    trb = Bank()
    for k, i in enumerate(tr_idx):
        for p in G.short(int(st.a[i]), int(st.b[i])):
            trb.add(k, int(st.r[i]), p)
    ytr = torch.as_tensor(st.world_y[tr_idx].astype(np.float32))
    tq, ttok, tmr, tmi = trb.tensors(G.n_tok)
    has_valid_tr = np.zeros(len(tr_idx), bool)
    for r_, it in zip(trb.mem_row, trb.mem_item):
        q, p = trb.items[it]
        if valid_program(G, rels[q], p):
            has_valid_tr[r_] = True
    # evaluation candidates
    rng = np.random.default_rng([seed, 77])
    adj_all = D.adjacency(conf, lambda r: True)
    ev_idx = np.unique(np.concatenate([masks[t] for t in EVAL_TESTS]))
    evb, ev_oracle = Bank(), Bank()
    for k, i in enumerate(ev_idx):
        a, q, b = int(st.a[i]), int(st.r[i]), int(st.b[i])
        for p in G.short(a, b) + G.sampled(a, b, rng):
            evb.add(k, q, p)
        facts = D.pattern_path(D.bfs_pattern(adj_all, a, D.AUT[rels[q]], rels), b, D.AUT[rels[q]][1])
        if facts is not None:
            ch = D.chain([(x, rels[r], z) for x, r, z in facts], a, b)
            ev_oracle.add(k, q, tuple(G.tok_of[(f[1], fwd)] for f, fwd in ch))
    eq_, etok, emr, emi = evb.tensors(G.n_tok)
    cost['candidates'] = {'cpu_seconds': time.process_time() - t0, 'train_rows': int(len(tr_idx)),
                          'train_unique_programs': len(trb.items), 'train_memberships': len(trb.mem_row),
                          'eval_rows': int(len(ev_idx)), 'eval_unique_programs': len(evb.items),
                          'eval_memberships': len(evb.mem_row), 'oracle_programs': len(ev_oracle.items)}
    ev_valid = np.array([valid_program(G, rels[q], p) for q, p in evb.items])
    ev_len = np.array([len(p) for _, p in evb.items])
    yev = st.world_y[ev_idx]
    pos = {int(i): k for k, i in enumerate(ev_idx)}

    # probes
    P, IS, SUB, NAMES = (G.tok_of[(x, True)] for x in ('parent-of', 'is-a', 'subkind-of', 'names'))
    probes = {f'{q}|parent-of x{k}': (rels.index(q), (P,) * k) for q in ('grandparent-of', 'ancestor-of') for k in range(1, 6)}
    probes.update({f'kind-of|is-a subkind-of x{k}': (rels.index('kind-of'), (IS,) + (SUB,) * k) for k in range(0, 5)})
    probes.update({f'named-kind|names is-a subkind-of x{k}': (rels.index('named-kind'), (NAMES, IS) + (SUB,) * k) for k in range(0, 4)})

    res = {'seed': seed, 'train_rows_with_valid_candidate': {'T': float(has_valid_tr[ytr.numpy() == 1].mean()),
                                                           'F': float(has_valid_tr[ytr.numpy() == 0].mean())},
           'variants': {}}
    wpos = float((ytr == 0).sum() / (ytr == 1).sum())
    for vi, var in enumerate(VARIANTS):
        t0 = time.process_time()
        torch.manual_seed(seed * 10 + vi)
        model = Critic(G.n_tok, len(rels), var)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        log = []
        for step in range(1, STEPS + 1):
            opt.zero_grad()
            ls = model(tq, ttok)
            rm = row_max(ls, tmr, tmi, len(tr_idx)).clamp(min=-30., max=-1e-6)
            loss_t = -rm
            loss_f = -torch.log1p(-rm.exp().clamp(max=1 - 1e-6))
            loss = (ytr * loss_t * wpos + (1 - ytr) * loss_f).sum() / (ytr * wpos + (1 - ytr)).sum()
            loss.backward()
            opt.step()
            if step % 500 == 0:
                log.append({'step': step, 'loss': float(loss)})
        train_cpu = time.process_time() - t0
        t0 = time.process_time()
        with torch.no_grad():
            p_tr = row_max(model(tq, ttok), tmr, tmi, len(tr_idx)).exp().numpy()
            ls_ev = model(eq_, etok)
            sc_ev = ls_ev.exp().numpy()
            rm_ev = row_max(ls_ev, emr, emi, len(ev_idx)).exp().numpy()
            if ev_oracle.items:
                oq, otok, omr, omi = ev_oracle.tensors(G.n_tok)
                ro = row_max(model(oq, otok), omr, omi, len(ev_idx)).exp().numpy()
            else:
                ro = np.zeros(len(ev_idx))
            rm_or = np.maximum(rm_ev, ro)
            pr = {k: float(model(torch.tensor([q]), encode([p], G.n_tok)).exp()) for k, (q, p) in probes.items()}
        out = {'train_cpu_seconds': train_cpu, 'log': log, 'probes': pr,
               'train_fit': {'T_recall': float((p_tr[ytr.numpy() == 1] > .5).mean()),
                             'F_recall': float((p_tr[ytr.numpy() == 0] <= .5).mean()),
                             'auroc': R1.auroc(p_tr, ytr.numpy())}, 'tests': {}}
        mr, mi = np.array(evb.mem_row), np.array(evb.mem_item)
        for t in EVAL_TESTS:
            rows = np.array([pos[int(i)] for i in masks[t]])
            if not len(rows):
                continue
            y = yev[rows].astype(bool)
            rec = {}
            for tag, sc in (('with_valid', rm_or), ('without_valid', rm_ev)):
                s = sc[rows]
                rec[tag] = {'T_recall': float((s[y] > .5).mean()) if y.any() else None,
                            'F_recall': float((s[~y] <= .5).mean()) if (~y).any() else None,
                            'auroc': R1.auroc(s, y) if y.any() and (~y).any() else None}
                rec[tag]['bal'] = None if rec[tag]['T_recall'] is None or rec[tag]['F_recall'] is None else \
                    .5 * (rec[tag]['T_recall'] + rec[tag]['F_recall'])
            # program level, over this test's candidate memberships plus the valid proofs
            sel = np.isin(mr, rows)
            items = mi[sel]
            s_p, v_p, l_p = list(sc_ev[items]), list(ev_valid[items]), list(ev_len[items])
            if ev_oracle.items:
                omr_, omi_ = np.array(ev_oracle.mem_row), np.array(ev_oracle.mem_item)
                osel = np.isin(omr_, rows)
                with torch.no_grad():
                    osc = model(oq, otok).exp().numpy()
                for it in omi_[osel]:
                    s_p.append(osc[it])
                    v_p.append(True)
                    l_p.append(len(ev_oracle.items[it][1]))
            s_p, v_p, l_p = np.array(s_p), np.array(v_p, bool), np.array(l_p)
            rec['program_auroc'] = R1.auroc(s_p, v_p) if v_p.any() and (~v_p).any() else None
            rec['by_length'] = {int(L): {'n_valid': int((v_p & (l_p == L)).sum()), 'n_spurious': int((~v_p & (l_p == L)).sum()),
                                         'accept_valid': float((s_p[v_p & (l_p == L)] > .5).mean()) if (v_p & (l_p == L)).any() else None,
                                         'accept_spurious': float((s_p[~v_p & (l_p == L)] > .5).mean()) if (~v_p & (l_p == L)).any() else None}
                                for L in np.unique(l_p)}
            out['tests'][t] = rec
        out['eval_cpu_seconds'] = time.process_time() - t0
        cost[var] = {'train_cpu_seconds': train_cpu, 'eval_cpu_seconds': out['eval_cpu_seconds'],
                     'params': int(sum(p.numel() for p in model.parameters()))}
        res['variants'][var] = out
    return res, cost


def task(t):
    seed = t[0]
    t0, c0 = time.perf_counter(), time.process_time()
    res, cost = run_seed(seed)
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
          'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'stages': cost}
    return f'seed{seed}', res, rt


def main():
    t0 = time.perf_counter()
    workers = int(os.environ.get('EV_WORKERS', 4))
    tasks = [(s,) for s in SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(min(workers, len(tasks)), maxtasksperchild=1) as pool:
        out = pool.map(DUR.durable(task, RUNS / 'done', 'seed{0}'), tasks, chunksize=1)
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'workers': workers,
                     'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(r['cpu_seconds'] for _, _, r in out)),
                     'tasks': {k: r for k, _, r in out}})
    V2.save_json(f'{REPORT}.json', {'status': 'composition critic check v1', 'seeds': {k: r for k, r, _ in out}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    rep = list(json.loads(open(f'{REPORT}.json').read())['seeds'].values())
    m = lambda v: None if not [x for x in v if x is not None] else round(float(np.mean([x for x in v if x is not None])), 3)
    print('train rows with a valid candidate', [r['train_rows_with_valid_candidate'] for r in rep])
    for var in VARIANTS:
        V = [r['variants'][var] for r in rep]
        print(f'\n===== {var}   train fit', {k: m([v['train_fit'][k] for v in V]) for k in ('T_recall', 'F_recall', 'auroc')})
        print('test'.ljust(16) + 'progAUC  | with valid: T    F    bal  AUC  | without: T    F    bal  AUC')
        for t in EVAL_TESTS:
            g = lambda tag, k: m([v['tests'][t][tag][k] for v in V if t in v['tests']])
            print(t.ljust(16) + f"{m([v['tests'][t]['program_auroc'] for v in V]) or float('nan'):.2f}     | " +
                  '  '.join(f"{g('with_valid', k) if g('with_valid', k) is not None else float('nan'):.2f}" for k in ('T_recall', 'F_recall', 'bal', 'auroc')) + '  | ' +
                  '  '.join(f"{g('without_valid', k) if g('without_valid', k) is not None else float('nan'):.2f}" for k in ('T_recall', 'F_recall', 'bal', 'auroc')))
        for t in ('L1-anc', 'L1-kind', 'L1-size-le5', 'L2-named', 'P6-grandparent', 'P6-colour'):
            Ls = sorted({L for v in V for L in v['tests'][t]['by_length']}, key=int)
            print('  ' + t + ' accept valid/spurious by length: ' + ', '.join(
                f"{L}: {m([v['tests'][t]['by_length'].get(L, {}).get('accept_valid') for v in V])}/"
                f"{m([v['tests'][t]['by_length'].get(L, {}).get('accept_spurious') for v in V])}" for L in Ls))
        print('  probes: ' + ', '.join(f'{k} {m([v["probes"][k] for v in V]):.2f}' for k in V[0]['probes']))
    out = {var: {'train_fit': {k: m([r['variants'][var]['train_fit'][k] for r in rep]) for k in ('T_recall', 'F_recall', 'auroc')},
                 'probes': {k: m([r['variants'][var]['probes'][k] for r in rep]) for k in rep[0]['variants'][var]['probes']},
                 'probes_per_seed': {k: [r['variants'][var]['probes'][k] for r in rep] for k in rep[0]['variants'][var]['probes']}}
           for var in VARIANTS}
    V2.save_json(f'{REPORT}-summary.json', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
