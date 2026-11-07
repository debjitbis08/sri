"""Rule critic diagnostic v1 (docs/rule-critic-diagnostic-v1.md): the rule-guided search v1 rule encoder and MLP
critic on an ancestor-style fragment (R -> H; R o H -> H or H o R -> H), trained three ways: confirmed-mil,
oracle-mil (question labels, max over candidate programs) and direct (valid / invalid program labels, a diagnostic
control). Program-level reads on held-out relations and instances, with checkpoints.

Usage: EV_WORKERS=5 .venv/bin/python -m experiments.rule_critic_diagnostic       (rerun to resume)
       .venv/bin/python -m experiments.rule_critic_diagnostic summary
"""
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import resource
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import durable as DUR
from experiments import everyday_relations_v2 as V2
from experiments import rule_guided_search as G

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'rule-critic-diagnostic-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'rule-critic-diagnostic-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
CONDS = ('confirmed-mil', 'oracle-mil', 'direct')
N_RS = 30 if SMOKE else 300
STEPS = 100 if SMOKE else 3000
CKPTS = (50, 100) if SMOKE else (500, 1000, 2000, 3000)
SPLIT_SEED = 2029
COMBOS = [(o, s) for o in ('fwd', 'rev') for s in ('lin-l', 'lin-r')]
LENS = range(1, 13)
NEG_KINDS = ('reverse-step', 'replace-step', 'splice', 'wrong-orientation', 'random')


# ---------------------------------------------------------------- fragment

def make_rs(rd, R, orient, side, rng):
    a = (0, 1) if orient == 'fwd' else (1, 0)
    seed = (1, (0, 1), [(0, a)], 'sub')
    if side == 'lin-l':
        rec = (1, (0, 2), [(0, a), (1, (1, 2))], 'lin-l')
    else:
        rec = (1, (0, 2), [(1, (0, 1)), (0, (1, 2) if orient == 'fwd' else (2, 1))], 'lin-r')
    return dict(glob=[R], nb=1, m=2, rules=[seed, rec], ask=1, sym=G.unitv(rng, 2), R=R, orient=orient, side=side)


def star(rd, rs):
    V = len(rd.vocab)
    return rd.vidx[rs['R']] + (0 if rs['orient'] == 'fwd' else V)


def split(rd, orc):
    rich = [r for r in rd.vocab if orc[r].sum() >= 5]
    g = np.random.default_rng(SPLIT_SEED)
    forced = {rd.W.R['parent-of'], rd.W.R['just-bigger-than']}
    rest = [r for r in rich if r not in forced]
    n_held = max(0, int(round(.2 * len(rich))) - len(forced))
    held = set(g.choice(rest, n_held, replace=False).tolist()) | forced
    train = [r for r in rich if r not in held]
    g2 = np.random.default_rng(SPLIT_SEED + 1)
    held_combo = {r: COMBOS[g2.integers(4)] for r in train}
    return train, sorted(held), held_combo


# ---------------------------------------------------------------- programs and negatives

def neg_program(kind, k, s, rd, rng):
    V, S = len(rd.vocab), rd.S
    flip = lambda x: (x + V) % S
    other = lambda: int((s % V + 1 + rng.integers(V - 1)) % V) + (s // V) * V
    p = [s] * k
    if kind == 'reverse-step':
        i = int(rng.integers(k)); p[i] = flip(s)
    elif kind == 'replace-step':
        i = int(rng.integers(k)); p[i] = other()
    elif kind == 'splice':
        if k < 2:
            return None
        i = int(rng.integers(1, k)); o = other()
        p = [s] * i + [o] * (k - i)
    elif kind == 'wrong-orientation':
        p = [flip(s)] * k
    else:
        p = [int(x) for x in rng.integers(S, size=k)]
        if p == [s] * k:
            return None
    return tuple(p)


def first_fail(C, prog, S, perfect=None):
    """Index of the first check that fails (C <= .5), or that a perfect critic would reject (perfect given)."""
    seq = (S,) + tuple(prog)
    checks = [(seq[t], prog[t]) for t in range(len(prog))] + [(prog[-1], S)]
    for t, (l, r) in enumerate(checks):
        ok = perfect(l, r, t, len(checks)) if perfect else C[l, r] > .5
        if not ok:
            return 'START' if t == 0 else ('END' if t == len(checks) - 1 else 'internal')
    return None


# ---------------------------------------------------------------- training

def mil_data(rd, graph, orc, rsets, rng):
    succ = G.succ_lists(graph, rd)
    succT = []
    for s in range(rd.S):
        A = G.step_adj(graph, rd, s).T
        succT.append([np.flatnonzero(A[x]) for x in range(rd.N)])
    conn = G.connected(graph, rd)
    eye = np.eye(rd.N, dtype=bool)
    qs, st = [], {'oracle_T_short': 0, 'covered': 0, 'F_excluded': 0}
    for i, rs in enumerate(rsets):
        yo, ho = G.labels(rs, orc, K=2)
        yc, hc = G.labels(rs, graph, K=2)
        shortT = yo & (ho >= 1) & (ho <= 2) & ~eye
        cov = shortT & (hc >= 1) & (hc <= 2)
        st['oracle_T_short'] += int(shortT.sum()); st['covered'] += int(cov.sum())
        Fm = ~yo & ~eye
        st['F_excluded'] += int((Fm & yc).sum())
        Fm &= ~yc
        T, Fc, Fo = np.argwhere(cov), np.argwhere(Fm & conn), np.argwhere(Fm & ~conn)
        pick = lambda A, n: A[rng.choice(len(A), min(n, len(A)), replace=False)] if len(A) else A
        ch = [(a, b, 1) for a, b in pick(T, 20)] + [(a, b, 0) for a, b in pick(Fc, 10)] + \
             [(a, b, 0) for a, b in pick(Fo, 20 - min(10, len(Fc)))]
        for a, b, y in ch:
            pr = G.preds(succT, int(b), rd)
            qs.append(dict(rs=rs, rsid=i, a=int(a), b=int(b), y=y, progs=G.candidates(int(a), int(b), succ, pr, rd)))
    st['questions'] = len(qs)
    st['programs'] = int(sum(len(q['progs']) for q in qs))
    return qs, st


def prog_logp(critic, rs, E, P):
    """log prod c over START, steps, END for programs P (n, L) padded with -1 (vectorized)."""
    S, K = critic.S, critic.S + 2
    P = np.asarray(P, dtype=np.int64)
    n, Lm = P.shape
    lengths = (P >= 0).sum(1)
    Ls, Rs, Os = [], [], []
    rows = np.arange(n)
    for t in range(Lm):
        has = P[:, t] >= 0
        last = np.full(n, S) if t == 0 else P[:, t - 1]
        Ls.append(last[has]); Rs.append(P[has, t]); Os.append(rows[has])
    lastv = P[rows, lengths - 1]
    Ls.append(lastv); Rs.append(np.full(n, S + 1)); Os.append(rows)
    L, R, O = np.concatenate(Ls), np.concatenate(Rs), np.concatenate(Os)
    uniq, inv = np.unique(L * K + R, return_inverse=True)
    c = critic.score(critic.tokens(rs), E, torch.as_tensor(uniq // K), torch.as_tensor(uniq % K))
    lc = torch.log(c.clamp_min(1e-6))[torch.as_tensor(inv)]
    return torch.zeros(n).index_add(0, torch.as_tensor(O), lc)


def train(critic, cond, rsets, qs, rd, seed, on_ckpt):
    opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    S, K = critic.S, critic.S + 2
    rng = np.random.default_rng([seed, 11])
    log = []
    if cond != 'direct':
        by = {}
        for i, q in enumerate(qs):
            if len(q['progs']):
                by.setdefault(q['rsid'], {0: [], 1: []})[q['y']].append(i)
        rsids = [r for r, d in by.items() if d[0] and d[1]]
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        E = critic.emb_all()
        total, n = 0., 0
        if cond == 'direct':
            for i in rng.choice(len(rsets), min(16, len(rsets)), replace=False):
                rs = rsets[i]
                s = star(rd, rs)
                progs = [(s, -1), (s, s)]
                labels = [1., 1.]
                for kind in ('random', NEG_KINDS[rng.integers(4)]):
                    for _ in range(10):
                        k = int(rng.integers(1, 3))
                        p = neg_program(kind, k, s, rd, rng)
                        if p is not None:
                            break
                    progs.append(tuple(p) + (-1,) * (2 - len(p))); labels.append(0.)
                lp = prog_logp(critic, rs, E, progs)
                total = total + F.binary_cross_entropy(lp.exp().clamp(1e-6, 1 - 1e-6), torch.as_tensor(labels), reduction='sum')
                n += len(progs)
        else:
            for rsid in rng.choice(rsids, min(16, len(rsids)), replace=False):
                d = by[rsid]
                ii = list(rng.choice(d[1], min(16, len(d[1])), replace=False)) + \
                    list(rng.choice(d[0], min(16, len(d[0])), replace=False))
                P = np.concatenate([qs[i]['progs'] for i in ii]).astype(np.int64)
                owner = np.concatenate([np.full(len(qs[i]['progs']), j) for j, i in enumerate(ii)])
                lp = prog_logp(critic, qs[ii[0]]['rs'], E, P)
                qmax = torch.full((len(ii),), -1e9).scatter_reduce(0, torch.as_tensor(owner), lp, 'amax')
                y = torch.as_tensor([float(qs[i]['y']) for i in ii])
                total = total + F.binary_cross_entropy(qmax.exp().clamp(1e-6, 1 - 1e-6), y, reduction='sum')
                n += len(ii)
        loss = total / n
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == STEPS:
            log.append({'step': step, 'loss': loss.item()})
        if step in CKPTS:
            on_ckpt(step)
    return log


# ---------------------------------------------------------------- reads

def eval_sets(rd, train_rel, held_rel, held_combo, seed):
    rng = np.random.default_rng([seed, 31])
    rel = [make_rs(rd, R, o, s, rng) for R in held_rel for o, s in COMBOS]
    inst = [make_rs(rd, R, *held_combo[R], rng) for R in train_rel]
    return {'held_relations': rel, 'held_instances': inst}


def program_reads(critic, rd, rsets, seed, full=True):
    rng = np.random.default_rng([seed, 41])
    S = rd.S
    acc = {k: [0, 0] for k in LENS}
    neg = {kind: {k: [0, 0] for k in LENS} for kind in NEG_KINDS}
    where_valid = {'START': 0, 'internal': 0, 'END': 0}
    where_neg = {'START': 0, 'internal': 0, 'END': 0}
    inv = {'fresh-symbols': [0, 0], 'variable-renaming': [0, 0], 'rule-order': [0, 0]}
    for rs in rsets:
        C = critic.table(rs)
        s = star(rd, rs)
        perfect = lambda l, r, t, n: (l == S and r == s) if t == 0 else ((l == s and r == S) if t == n - 1 else (l == s and r == s))
        progs = []
        for k in LENS:
            p = (s,) * k
            a = G.program_accepts(C, p, S)
            acc[k][0] += int(a); acc[k][1] += 1
            progs.append(p)
            if not a and full:
                where_valid[first_fail(C, p, S)] += 1
            for kind in NEG_KINDS:
                for _ in range(3 if full else 1):
                    q = neg_program(kind, k, s, rd, rng)
                    if q is None:
                        continue
                    a = G.program_accepts(C, q, S)
                    neg[kind][k][0] += int(a); neg[kind][k][1] += 1
                    progs.append(q)
                    if a and full:
                        f = first_fail(C, q, S, perfect)
                        if f:
                            where_neg[f] += 1
        if full:
            base = np.array([G.program_accepts(C, p, S) for p in progs])
            sw = lambda v: {0: 2, 2: 0}.get(v, v)
            variants = {'fresh-symbols': dict(rs, sym=G.unitv(rng, rs['m'])),
                        'variable-renaming': dict(rs, rules=[(h, (sw(hv[0]), sw(hv[1])), [(r, (sw(v[0]), sw(v[1]))) for r, v in body], f)
                                                            for h, hv, body, f in rs['rules']]),
                        'rule-order': dict(rs, rules=list(reversed(rs['rules'])))}
            for key, e in variants.items():
                Ce = critic.table(e)
                new = np.array([G.program_accepts(Ce, p, S) for p in progs])
                inv[key][0] += int((new != base).sum()); inv[key][1] += len(progs)
    r = lambda h: None if h[1] == 0 else h[0] / h[1]
    out = {'valid_accepted': {k: r(v) for k, v in acc.items()},
           'negative_accepted': {kind: {k: r(v) for k, v in d.items()} for kind, d in neg.items()}}
    if full:
        out.update(where_valid_rejected=where_valid, where_negative_accepted=where_neg,
                   invariance_changed={k: r(v) for k, v in inv.items()})
    return out


def witness_quality(critic, qs, rd, n=500):
    rng = np.random.default_rng(5)
    T = [q for q in qs if q['y'] == 1 and len(q['progs'])]
    T = [T[i] for i in rng.choice(len(T), min(n, len(T)), replace=False)]
    good = 0
    with torch.no_grad():
        E = critic.emb_all()
        for q in T:
            lp = prog_logp(critic, q['rs'], E, q['progs'].astype(np.int64))
            p = q['progs'][int(lp.argmax())]
            s = star(rd, q['rs'])
            good += int(all(x == s for x in p if x >= 0))
    return good / max(len(T), 1)


def answer_reads(critic, rd, g, orc, seed):
    specs = G.taught_specs(rd)
    out = {}
    for name in ('ancestor-of', 'bigger-than'):
        rs = dict(specs[name])
        rs['sym'] = G.unitv(np.random.default_rng([seed, 71, len(name)]), rs['m'])
        ents = G.task_entities(rs, orc)
        yo, ho = G.labels(rs, orc)
        conn = G.connected(g, rd)
        q = [(a, b, int(yo[a, b]), int(ho[a, b]), bool(conn[a, b])) for a in ents for b in ents if a != b]
        C = critic.table(rs)
        out[name] = {}
        for gname, graph in (('oracle', orc), ('confirmed', g)):
            ctx = G.graph_ctx(graph, rd, seed)
            out[name][gname] = G.summarize_tally(G.evaluate(rs, q, C, ctx, G.new_tally(), full=False))
    return out


# ---------------------------------------------------------------- run

def run(cond, seed):
    torch.set_num_threads(1)
    ph = {}
    t0 = time.process_time()
    rd = G.Readers(seed)
    g, ng, sup, orc = rd.graphs()
    del ng, sup
    train_rel, held_rel, held_combo = split(rd, orc)
    ph['graph_cpu_seconds'] = time.process_time() - t0
    rng = np.random.default_rng([seed, 21])
    rsets = []
    for _ in range(N_RS):
        R = int(train_rel[rng.integers(len(train_rel))])
        allowed = [c for c in COMBOS if c != held_combo[R]]
        o, s = allowed[rng.integers(len(allowed))]
        rsets.append(make_rs(rd, R, o, s, rng))
    t0 = time.process_time()
    qs, qstats = ([], {}) if cond == 'direct' else mil_data(rd, g if cond == 'confirmed-mil' else orc, orc, rsets, rng)
    ph['candidates_cpu_seconds'] = time.process_time() - t0
    ev = eval_sets(rd, train_rel, held_rel, held_combo, seed)
    torch.manual_seed(seed)
    critic = G.Critic(rd, 'rules')
    ckpts = {}
    ck_cpu = [0.]

    def on_ckpt(step):
        c0 = time.process_time()
        with torch.no_grad():
            ckpts[step] = {k: program_reads(critic, rd, v, seed, full=False) for k, v in ev.items()}
            if qs:
                ckpts[step]['witness_quality'] = witness_quality(critic, qs, rd)
        ck_cpu[0] += time.process_time() - c0
    t0 = time.process_time()
    log = train(critic, cond, rsets, qs, rd, seed, on_ckpt)
    ph['training_cpu_seconds'] = time.process_time() - t0 - ck_cpu[0]
    ph['checkpoint_reads_cpu_seconds'] = ck_cpu[0]
    for p in critic.parameters():
        p.requires_grad_(False)
    t0 = time.process_time()
    res = {'cond': cond, 'seed': seed, 'log': log, 'mil': qstats, 'checkpoints': ckpts,
           'split': {'train_relations': len(train_rel), 'held_relations': len(held_rel)},
           'final': {k: program_reads(critic, rd, v, seed) for k, v in ev.items()},
           'witness_quality': witness_quality(critic, qs, rd) if qs else None,
           'answers': answer_reads(critic, rd, g, orc, seed)}
    ph['final_reads_cpu_seconds'] = time.process_time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{cond}-seed{seed}.pt'
    torch.save(critic.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {**ph, 'steps': STEPS, 'critic_params': int(sum(p.numel() for p in critic.parameters())), 'mil': qstats}


def task(t):
    t0, c0 = time.perf_counter(), time.process_time()
    res, rt = run(*t)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{t[0]}/seed{t[1]}', res, rt


def main():
    t0 = time.perf_counter()
    workers = int(os.environ.get('EV_WORKERS', 5))
    tasks = [(c, s) for c in CONDS for s in SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(min(workers, len(tasks)), maxtasksperchild=1) as pool:
        out = pool.map(DUR.durable(task, RUNS / 'done', '{0}-seed{1}'), tasks, chunksize=1)
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'workers': workers,
                     'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(r['cpu_seconds'] for _, _, r in out)),
                     'tasks': {k: r for k, _, r in out}})
    V2.save_json(f'{REPORT}.json', {'status': 'rule critic diagnostic v1' + (' (smoke)' if SMOKE else ''),
                                    'results': {k: r for k, r, _ in out}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    res = json.loads(open(f'{REPORT}.json').read())['results']
    m = lambda v: None if not [x for x in v if x is not None] else round(float(np.mean([x for x in v if x is not None])), 3)
    out = {}
    for cond in CONDS:
        R = [r for r in res.values() if r['cond'] == cond]
        if not R:
            continue
        o = {'final': {}, 'checkpoints': {}, 'witness_quality': m([r['witness_quality'] for r in R]), 'answers': {}}
        for ev in ('held_relations', 'held_instances'):
            xs = [r['final'][ev] for r in R]
            o['final'][ev] = {
                'valid_accepted': {k: m([x['valid_accepted'][k] for x in xs]) for k in xs[0]['valid_accepted']},
                'negative_accepted': {kind: {k: m([x['negative_accepted'][kind][k] for x in xs]) for k in xs[0]['negative_accepted'][kind]}
                                      for kind in NEG_KINDS},
                'where_valid_rejected': {w: int(sum(x['where_valid_rejected'][w] for x in xs)) for w in ('START', 'internal', 'END')},
                'where_negative_accepted': {w: int(sum(x['where_negative_accepted'][w] for x in xs)) for w in ('START', 'internal', 'END')},
                'invariance_changed': {k: m([x['invariance_changed'][k] for x in xs]) for k in xs[0]['invariance_changed']}}
        for step in R[0]['checkpoints']:
            cs = [r['checkpoints'][step] for r in R]
            o['checkpoints'][step] = {
                'held_relations_valid_3_12': m([np.mean([c['held_relations']['valid_accepted'][k] for k in map(str, range(3, 13))]) for c in cs]),
                'held_relations_hard_negative_2_12': m([np.mean([c['held_relations']['negative_accepted'][kind][k] for kind in NEG_KINDS[:4]
                                                                 for k in map(str, range(2, 13)) if c['held_relations']['negative_accepted'][kind][k] is not None]) for c in cs]),
                'witness_quality': m([c.get('witness_quality') for c in cs])}
        for name in R[0]['answers']:
            o['answers'][name] = {gname: {'bal': {k: m([r['answers'][name][gname]['bal'][k] for r in R]) for k in G.BUCKETS},
                                          'F_recall': m([r['answers'][name][gname]['F_recall'] for r in R])}
                                  for gname in ('oracle', 'confirmed')}
        out[cond] = o
        f = o['final']['held_relations']
        print(f"\n===== {cond}: witness {o['witness_quality']}")
        print(f"  held relations valid {f['valid_accepted']}")
        print(f"  held relations hard neg (len 2-12 mean) " + str({k: m(list(v.values())[1:]) for k, v in f['negative_accepted'].items()}))
        print(f"  where valid rejected {f['where_valid_rejected']} where neg accepted {f['where_negative_accepted']} invariance {f['invariance_changed']}")
        fi = o['final']['held_instances']
        print(f"  held instances valid {fi['valid_accepted']}")
        print(f"  checkpoints {o['checkpoints']}")
        print(f"  answers {o['answers']}")
    V2.save_json(f'{REPORT}-summary.json', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
