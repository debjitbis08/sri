"""Conflict search diagnostics v1 (docs/conflict-search-diagnostics-v1.md): proof validity, restricted hops, oracle and
exhaustive controls on the frozen readers and controllers of conflict search v1. No training.

The exhaustive and oracle searches are measuring tools (user consent, 2026-11-06), not part of the system.

Usage: EV_WORKERS=4 .venv/bin/python -m experiments.conflict_search_diag        (rerun to resume)
       .venv/bin/python -m experiments.conflict_search_diag summary
"""
import collections
import json
import os
import resource
import sys
import time

import numpy as np
import torch

from experiments import conflict_search as CS
from experiments import dual_route_truth as DR
from experiments import durable as DUR
from experiments import everyday_relations_v2 as V2
from experiments import taught_truth as TT
from src import everyday_truth_v2 as T2

ROOT, nc = V2.ROOT, V2.nc
NAME = 'conflict-search-diagnostics-v1'
CONTRACT = ROOT / 'docs' / f'{NAME}.md'
SRC_RUNS = ROOT / 'runs' / 'conflict-search-v1'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
SEEDS = (450, 451, 452, 453)
TRAINED_ARMS = ('bt-forward', 'bt-bidir', 'bt-forward-notrace')
RANDOM_REPEATS = 3
GRAMMAR = ('good-phrase', 'makes-sense', 'not', 'and', 'or')
TESTS = ('P6-grandparent', 'P6-colour', 'L1-kind', 'L1-anc', 'L1-anc-d45', 'L1-size', 'L1-size-le5', 'L2-named',
         'withheld-fit')

# proof definitions as automata over tokens (relation, forward); state 0 is the start
P, J = 'parent-of', 'just-bigger-than'
AUT = {
    'grandparent-of': ({0: {(P, True): 1}, 1: {(P, True): 2}}, {2}),
    'ancestor-of': ({0: {(P, True): 1}, 1: {(P, True): 1}}, {1}),
    'kind-of': ({0: {('is-a', True): 1}, 1: {('subkind-of', True): 1}}, {1}),
    'bigger-than': ({0: {(J, True): 1}, 1: {(J, True): 1}}, {1}),
    'smaller-than': ({0: {(J, False): 1}, 1: {(J, False): 1}}, {1}),
    'named-can': ({0: {('names', True): 1}, 1: {('can', True): 2}}, {2}),
    'named-has': ({0: {('names', True): 1}, 1: {('has', True): 2}}, {2}),
    'named-kind': ({0: {('names', True): 1}, 1: {('is-a', True): 2}, 2: {('subkind-of', True): 2}}, {2}),
    'same-colour-as': ({0: {('colour-of', True): 1}, 1: {('colour-of', False): 2}}, {2}),
}


def permitted_relations(st, train_rows):
    rels = np.array(st.relations)
    taught = set(rels[st.r[train_rows & (st.kind == 'individual') & (st.world_y == 1)]].tolist())
    return sorted(x for x in taught if x not in CS.WITHHELD_RELS and x not in GRAMMAR)


# ---------------------------------------------------------------- proof reads

def chain(facts, a, b):
    """Order facts (x, rel, y) into a chain a -> b; returns [(fact, forward)] or None."""
    adj = collections.defaultdict(list)
    for f in facts:
        adj[f[0]].append((f, f[2], True))
        adj[f[2]].append((f, f[0], False))
    prev = {a: None}
    q = collections.deque([a])
    while q:
        x = q.popleft()
        if x == b and x != a:
            break
        for f, y, fwd in adj[x]:
            if y not in prev:
                prev[y] = (x, f, fwd)
                q.append(y)
    if b not in prev or b == a:
        return None
    out, x = [], b
    while prev[x] is not None:
        px, f, fwd = prev[x]
        out.append((f, fwd))
        x = px
    return out[::-1]


def token(rel, fwd):
    return (P, not fwd) if rel == 'child-of' else (rel, fwd)


def accepts(rel, toks):
    if rel not in AUT:
        return False
    trans, acc = AUT[rel]
    s = 0
    for t in toks:
        s = trans.get(s, {}).get(t)
        if s is None:
            return False
    return s in acc


def proof_reads(facts, a, rel, b, world, permitted):
    ch = chain(facts, a, b)
    factual = all(f in world.true for f in facts)
    perm = all(f[1] in permitted for f in facts)
    if ch is None:
        return factual, perm, False, '(broken)'
    toks = [token(f[1], fwd) for f, fwd in ch]
    valid = factual and accepts(rel, toks)
    prog = ' '.join(('>' if fw else '<') + r for r, fw in toks)
    return factual, perm, valid, prog


# ---------------------------------------------------------------- graphs

@torch.no_grad()
def confirmed_graph(model):
    """Logic's edge table (the search's candidate pools) and the both-route confirmed facts."""
    pool0, _, _ = CS.edge_table(model)
    d = model.dual
    E_I, E_L, R_I, R_L = d.E_I.detach(), d.E_L.detach(), d.R_I.detach(), d.R_L.detach()
    tri = pool0.nonzero()
    conf = []
    for s in range(0, len(tri), 50000):
        x, r, y = tri[s:s + 50000].T
        sl = model.logic_codes(E_L[x], R_L[r], E_L[y])
        sig = torch.sigmoid(model.intuition_codes(E_I[x], R_I[r], E_I[y]))
        conf.append((CS.disagreement(sig, sl) <= CS.TAU) & (sl[:, 0] > .5))
    conf = torch.cat(conf) if conf else torch.zeros(0, dtype=torch.bool)
    facts = tri[conf].tolist()
    ne, nr = pool0.shape[0], pool0.shape[1]
    return pool0, {tuple(f) for f in facts}, {'edge_table_reader_calls': int(ne * nr * ne),
                                               'confirm_reader_calls': int(2 * len(tri)),
                                               'pool_triples': int(len(tri)), 'confirmed_facts': int(len(facts))}


def adjacency(facts, rel_ok):
    adj = collections.defaultdict(list)
    for x, r, y in facts:
        if rel_ok(r):
            adj[x].append((r, y, True))
            adj[y].append((r, x, False))
    return adj


def bfs_pattern(adj, a, aut, rels):
    """Depth-limited BFS over (node, proof state) from a; returns {(node, state): parent} with parents for paths."""
    trans, _ = aut
    par = {(a, 0): None}
    frontier = [(a, 0)]
    for _ in range(CS.D_MAX):
        nxt = []
        for x, s in frontier:
            tr = trans.get(s)
            if not tr:
                continue
            for r, y, fwd in adj.get(x, ()):
                ns = tr.get(token(rels[r], fwd))
                if ns is None or (y, ns) in par:
                    continue
                par[(y, ns)] = (x, s, r, fwd)
                nxt.append((y, ns))
        frontier = nxt
    return par


def pattern_path(par, b, acc):
    """A proof is a simple path: b == a (e.g. same-colour-as(x, x), false in the fixture) is never proved."""
    if (b, 0) in par and par[(b, 0)] is None:
        return None
    for s in acc:
        if (b, s) in par and par[(b, s)] is not None:
            facts, key = [], (b, s)
            while par[key] is not None:
                x, ps, r, fwd = par[key]
                y = key[0]
                facts.append((x, r, y) if fwd else (y, r, x))
                key = (x, ps)
            return facts[::-1]
    return None


def bfs_plain(adj, a):
    par = {a: None}
    frontier = [a]
    for _ in range(CS.D_MAX):
        nxt = []
        for x in frontier:
            for r, y, fwd in adj.get(x, ()):
                if y in par:
                    continue
                par[y] = (x, r, fwd)
                nxt.append(y)
        frontier = nxt
    return par


def plain_path(par, a, b):
    if b == a or b not in par:
        return None
    facts, y = [], b
    while par[y] is not None:
        x, r, fwd = par[y]
        facts.append((x, r, y) if fwd else (y, r, x))
        y = x
    return facts[::-1]


def oracle_random(a, b, aut, rels, pool0, conf, rng):
    """The v1 walk's budget and branching (B checks, k = 3 then 2, depth <= D_max) with the oracle's relation and
    direction at each step and a random order over the reader's candidate pool. Returns (facts or None, checks)."""
    trans, acc = aut
    rid = {x: i for i, x in enumerate(rels)}

    class Fr:
        __slots__ = ('node', 'state', 'depth', 'left', 'tried', 'parent', 'fact')

    root = Fr()
    root.node, root.state, root.depth, root.left, root.tried, root.parent, root.fact = a, 0, 0, CS.K_FIRST, set(), None, None
    cur, checks = root, 0
    while checks < CS.B:
        on_path, f = {a}, cur
        while f is not None:
            on_path.add(f.node)
            f = f.parent
        tr = trans.get(cur.state, {})
        cands, choice = [], None
        if tr and cur.left > 0:
            opts = list(tr.items())
            (rel, fwd), ns = opts[rng.integers(len(opts))]
            r = rid[rel]
            pool = pool0[cur.node, r] if fwd else pool0[:, r, cur.node]
            cands = [m for m in np.flatnonzero(pool) if m not in on_path and m not in cur.tried]
            choice = (r, fwd, ns)
        if not cands:
            if cur.parent is None:
                return None, checks
            cur.parent.tried.add(cur.node)
            cur = cur.parent
            continue
        r, fwd, ns = choice
        m = int(cands[rng.integers(len(cands))])
        cur.left -= 1
        checks += 1
        fact = (cur.node, r, m) if fwd else (m, r, cur.node)
        if fact in conf:
            if m == b and ns in acc:
                facts, f = [fact], cur
                while f.parent is not None:
                    facts.append(f.fact)
                    f = f.parent
                return facts[::-1], checks
            if m != b and cur.depth + 1 < CS.D_MAX and trans.get(ns):
                ch = Fr()
                ch.node, ch.state, ch.depth, ch.left, ch.tried, ch.parent, ch.fact = m, ns, cur.depth + 1, CS.K_DEEP, set(), cur, fact
                cur = ch
                continue
        cur.tried.add(m)
    return None, checks


# ---------------------------------------------------------------- reads

def answer_reads(found, y, proofs, checks=None):
    """found (bool array), y (0/1), proofs: list per row of (factual, permitted, valid, prog) or None."""
    found, y = np.asarray(found, bool), np.asarray(y).astype(bool)
    f = lambda v: float(np.mean(v)) if len(v) else None
    nT, nF = int(y.sum()), int((~y).sum())
    t_rec = f(found[y])
    f_rec_a = f(~found[~y])
    rec = {'n_true': nT, 'n_false': nF, 'T_recall': t_rec, 'found_false': f(found[~y]),
           'A_F_recall': f_rec_a, 'A_bal': None if t_rec is None or f_rec_a is None else .5 * (t_rec + f_rec_a),
           'B_F_recall': 0.0 if nF else None, 'B_U_rate': f(~found),
           'B_bal': None if t_rec is None else .5 * t_rec}
    valid = np.array([bool(p and p[2]) for p in proofs])
    rec['valid_T_recall'] = f(valid[y])
    for tag, m in (('T', y), ('F', ~y)):
        fm = found & m
        ps = [proofs[i] for i in np.flatnonzero(fm)]
        rec[f'factual_{tag}'] = f([p[0] for p in ps])
        rec[f'permitted_{tag}'] = f([p[1] for p in ps])
        rec[f'valid_{tag}'] = f([p[2] for p in ps])
    progs = collections.Counter(p[3] for p, fd in zip(proofs, found) if fd and p)
    rec['programs'] = progs.most_common(4)
    if checks is not None:
        rec['checks_true'] = f(np.asarray(checks)[y])
        rec['checks_false'] = f(np.asarray(checks)[~y])
    return rec


def tfu_reads(ans, y):
    y = np.asarray(y).astype(bool)
    f = lambda v: float(np.mean(v)) if len(v) else None
    return {'T_recall': f(ans[y] == 'T'), 'F_recall': f(ans[~y] == 'F'), 'U_rate': f(ans == 'U'),
            'bal': DR.tf_bal(ans, y)}


def test_masks(st, train_rows):
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    wh = CS.withheld_mask(st)
    out = {}
    for t in TESTS:
        if t == 'withheld-fit':
            m = train_rows & ind & wh
        elif t == 'L1-anc-d45':
            m = ind & (st.test == 'L1-anc') & np.isin(st.sub, ('4', '5'))
        elif t == 'L1-size-le5':
            m = ind & (st.test == 'L1-size') & np.isin(st.sub, ('3', '4', '5'))
        else:
            m = ind & (st.test == t) & wh
        out[t] = np.flatnonzero(m)
    return out


def load_model(st, path, full):
    g = torch.Generator().manual_seed(0)
    model = CS.Model(len(st.entities), len(st.relations), g)
    sd = torch.load(path)
    (model if full else model.dual).load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_seed(seed):
    torch.set_num_threads(1)
    st, _ = T2.build()
    TT.value_idx_cache = TT.value_idx(st)
    train_rows, _ = CS.splits(st, seed)
    rels = list(st.relations)
    world = CS.World(st)
    permitted = set(permitted_relations(st, train_rows))
    perm_idx = torch.tensor([x in permitted for x in rels])
    masks = test_masks(st, train_rows)
    allidx = np.unique(np.concatenate(list(masks.values())))
    pos = {int(i): k for k, i in enumerate(allidx)}
    A, R, Bn = st.a[allidx], st.r[allidx], np.maximum(st.b[allidx], 0)
    Y = st.world_y[allidx]
    relname = [rels[r] for r in R]
    cost, res = {}, {'seed': seed, 'permitted_relations': sorted(permitted), 'controls': {}, 'baselines': {}}

    def store(name, found, proofs, checks=None):
        res['controls'][name] = {t: answer_reads(found[[pos[int(i)] for i in idx]], Y[[pos[int(i)] for i in idx]],
                                                 [proofs[pos[int(i)]] for i in idx],
                                                 None if checks is None else checks[[pos[int(i)] for i in idx]])
                                 for t, idx in masks.items()}

    # reader graph from the phase-1 reader
    t0 = time.process_time()
    base = load_model(st, SRC_RUNS / f'phase1-seed{seed}.pt', full=False)
    pool0, conf, gstats = confirmed_graph(base)
    pool0_np = pool0.numpy()
    cost['graph_construction'] = {'cpu_seconds': time.process_time() - t0, **gstats}
    res['graph'] = gstats

    # world ceiling: valid proof of <= D_max hops among true facts
    rid = {x: i for i, x in enumerate(rels)}
    wfacts = [(a, rid[r], b) for a, r, b in world.true]
    wadj = adjacency(wfacts, lambda r: True)
    t0 = time.process_time()
    cache = {}
    found, proofs = np.zeros(len(allidx), bool), [None] * len(allidx)
    for k in range(len(allidx)):
        key = (int(A[k]), relname[k])
        if key not in cache:
            cache[key] = bfs_pattern(wadj, key[0], AUT[relname[k]], rels)
        facts = pattern_path(cache[key], int(Bn[k]), AUT[relname[k]][1])
        if facts is not None:
            found[k] = True
            proofs[k] = proof_reads([(a, rels[r], b) for a, r, b in facts], int(A[k]), relname[k], int(Bn[k]), world, permitted)
    store('world-ceiling', found, proofs)
    cost['world-ceiling'] = {'cpu_seconds': time.process_time() - t0}

    # oracle exhaustive and plain BFS on the confirmed graph
    for name, rel_ok in (('oracle-exhaustive', lambda r: True), ('bfs-permitted', lambda r: rels[r] in permitted),
                         ('bfs-all', lambda r: True)):
        t0 = time.process_time()
        adj = adjacency(conf, rel_ok)
        cache = {}
        found, proofs = np.zeros(len(allidx), bool), [None] * len(allidx)
        for k in range(len(allidx)):
            a, b = int(A[k]), int(Bn[k])
            if name == 'oracle-exhaustive':
                key = (a, relname[k])
                if key not in cache:
                    cache[key] = bfs_pattern(adj, a, AUT[relname[k]], rels)
                facts = pattern_path(cache[key], b, AUT[relname[k]][1])
            else:
                if a not in cache:
                    cache[a] = bfs_plain(adj, a)
                facts = plain_path(cache[a], a, b)
            if facts is not None:
                found[k] = True
                proofs[k] = proof_reads([(x, rels[r], z) for x, r, z in facts], a, relname[k], b, world, permitted)
        store(name, found, proofs)
        cost[name] = {'cpu_seconds': time.process_time() - t0, 'searches': len(cache)}

    # oracle relation, random entity order, v1 budget
    t0 = time.process_time()
    reps, total_checks = [], 0
    for rep in range(RANDOM_REPEATS):
        rng = np.random.default_rng([seed, 31, rep])
        found, proofs, checks = np.zeros(len(allidx), bool), [None] * len(allidx), np.zeros(len(allidx))
        for k in range(len(allidx)):
            a, b = int(A[k]), int(Bn[k])
            facts, c = oracle_random(a, b, AUT[relname[k]], rels, pool0_np, conf, rng)
            checks[k] = c
            if facts is not None:
                found[k] = True
                proofs[k] = proof_reads([(x, rels[r], z) for x, r, z in facts], a, relname[k], b, world, permitted)
        total_checks += checks.sum()
        store(f'oracle-random-r{rep}', found, proofs, checks)
        reps.append(f'oracle-random-r{rep}')
    cost['oracle-random'] = {'cpu_seconds': time.process_time() - t0, 'repeats': RANDOM_REPEATS,
                             'checks_total': float(total_checks), 'reader_calls_per_check': 2}

    # baselines: intuition alone, always-F
    with torch.no_grad():
        u = base.dual.intuition(torch.as_tensor(A), torch.as_tensor(R), torch.as_tensor(Bn)).numpy()
    intu = np.where(u > 0, 'T', 'F')
    for t, idx in masks.items():
        ii = [pos[int(i)] for i in idx]
        res['baselines'].setdefault(t, {})['intuition'] = tfu_reads(intu[ii], Y[ii])
        res['baselines'][t]['always-F'] = tfu_reads(np.array(['F'] * len(ii)), Y[ii])

    # trained controllers: as run, restricted to permitted relations, restricted to the question's base relations
    nr = len(rels)
    base_tab = torch.ones(nr, nr, dtype=torch.bool)
    for q, bl in CS.UNFOLD.items():
        base_tab[rid[q]] = torch.tensor([x in bl for x in rels])
    restr_tab = perm_idx[None].expand(nr, nr).clone()
    a_t, r_t, b_t = torch.as_tensor(A), torch.as_tensor(R), torch.as_tensor(Bn)
    for arm in TRAINED_ARMS:
        model = load_model(st, SRC_RUNS / f'{arm}-seed{seed}.pt', full=True)
        assert all(torch.equal(v, base.dual.state_dict()[k]) for k, v in model.dual.state_dict().items())
        model._edges = (('edges', id(model.dual)), CS.edge_table(base))
        mode = 'bidir' if arm == 'bt-bidir' else 'forward'
        for tag, tab in (('', None), ('-restricted', restr_tab), ('-base', base_tab)):
            t0 = time.process_time()
            with torch.no_grad():
                o = CS.search(model, a_t, r_t, b_t, mode, arm != 'bt-forward-notrace', rel_oracle=tab)
            found = o['found'].numpy()
            proofs = [None] * len(allidx)
            for k, row in enumerate(o['rows']):
                if row.status == 'found':
                    facts, _ = CS.path_facts(row, int(A[k]), int(Bn[k]), rels)
                    proofs[k] = proof_reads(facts, int(A[k]), relname[k], int(Bn[k]), world, permitted)
            store(f'trained{tag}/{arm}', found, proofs, o['checks'].astype(float))
            cost[f'trained{tag}/{arm}'] = {'cpu_seconds': time.process_time() - t0, 'checks_total': float(o['checks'].sum()),
                                          'rows': int(len(allidx))}
        # v1 answers as run (search only on conflicts; learned give-up; arbiter)
        for t, idx in masks.items():
            m = np.zeros(len(st.a), bool)
            m[idx] = True
            oa = CS.answer(model, st, m, arm)
            outs = CS.outputs(model, oa)
            res['baselines'][t][f'v1-logic-final/{arm}'] = tfu_reads(outs['logic-final'], st.world_y[m])
            res['baselines'][t][f'v1-arbiter/{arm}'] = tfu_reads(outs['arbiter'], st.world_y[m])
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
    V2.save_json(f'{REPORT}.json', {'status': 'conflict search diagnostics v1', 'seeds': {k: r for k, r, _ in out}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(float(np.mean(vals)), 3) if vals else None


def summary():
    rep = json.loads(open(f'{REPORT}.json').read())['seeds']
    seeds = list(rep.values())
    ctrls = list(seeds[0]['controls'])
    keys = ('T_recall', 'found_false', 'A_bal', 'B_bal', 'B_U_rate', 'valid_T_recall', 'factual_T', 'permitted_T',
            'valid_T', 'factual_F', 'valid_F', 'checks_true')
    out = {'controls': {}, 'baselines': {}, 'per_seed': {}}
    names = [c for c in ctrls if not c.startswith('oracle-random-r')] + ['oracle-random']
    for c in names:
        src = [c] if c != 'oracle-random' else [f'oracle-random-r{i}' for i in range(RANDOM_REPEATS)]
        out['controls'][c] = {t: {k: mean([s['controls'][x][t].get(k) for s in seeds for x in src]) for k in keys}
                              for t in TESTS}
        out['per_seed'][c] = {t: {s['seed']: {k: mean([s['controls'][x][t].get(k) for x in src])
                                              for k in ('T_recall', 'valid_T_recall', 'found_false')} for s in seeds}
                              for t in TESTS}
    for t in TESTS:
        out['baselines'][t] = {b: {k: mean([s['baselines'][t][b][k] for s in seeds]) for k in ('T_recall', 'F_recall', 'U_rate', 'bal')}
                               for b in seeds[0]['baselines'][t]}
    V2.save_json(f'{REPORT}-summary.json', out)
    show = ('T_recall', 'valid_T_recall', 'found_false', 'A_bal', 'B_bal', 'valid_T', 'factual_T', 'permitted_T')
    for t in TESTS:
        print(f'\n== {t}  (n true {seeds[0]["controls"]["world-ceiling"][t]["n_true"]}, n false {seeds[0]["controls"]["world-ceiling"][t]["n_false"]})')
        print('control'.ljust(34) + ''.join(k[:11].rjust(12) for k in show))
        for c in names:
            v = out['controls'][c][t]
            print(c.ljust(34) + ''.join(('-' if v[k] is None else f'{v[k]:.2f}').rjust(12) for k in show))
        for b, v in out['baselines'][t].items():
            print(('  ' + b).ljust(34) + f"T {v['T_recall']:.2f}  F {v['F_recall']:.2f}  U {v['U_rate']:.2f}  bal {v['bal']:.2f}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
