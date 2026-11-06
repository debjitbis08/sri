"""Stage 2 of docs/relational-expression-core-design.md: read-only baseline of relational expression reads through the existing
comparator g, against the current readers (g as read now, and P's rollout), on the step 8 sparse48 checkpoints (seeds 300-303,
`delay` and `relearn`). No training. Plan and inventory: docs/relational-expression-inventory-v1.md.

Every expression read goes through src/relational_expression.rank (max-min semantics, declared in stage 1) with the scorer
g(code(a), code(b), token(relation)); predecessor := inverse(successor) is supplied. Entities are opaque indices; counts are
used only to score answers. Top-1 credit is shared among exactly tied top bindings (expected accuracy under a random tie
break), and the tie rate is reported.

Reads per checkpoint:
  completion      same_quantity(q, ?x) and same_quantity(?x, q); successor(a, ?x) and successor(?x, b); for every
                  (query kind, answer kind) pair, 9 candidates (one rendering per count; never the query item itself)
  truth           AUROC of same, next and more over the same pairs; accuracy at logit 0 for next and more
  multi           same_quantity(q, ?x) and successor(a, ?x) over 5 renderings per count; R-precision (precision at 5)
  continuation    inferred direction: Exists r in {successor, predecessor}, y_1..y_{k-1}:
                  r(prev, cur) AND r(cur, y_1) AND ... AND r(y_{k-1}, ?x), k = 1, 2, 3, up and down, with intermediates and
                  answers in the `collections` domain (P's protocol: the 9 evaluation collections at the query's rendering
                  index) or the `within` domain (the query's own kind at that index); and supplied direction (r given,
                  no premise). Historical: sequence_pilot.composition (P rollout) on the same queries and final candidates.
  memory          relearning_branches.Reads (memory present, actual retrieval and abstention), plus each relational read
                  in both slots with the same retrieved values, abstentions and candidates; the historical slot must
                  reproduce exactly.
Usage: NFI_DEVICE=cuda RX_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relational_expression_baseline
"""
import hashlib
import os
import resource
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import relearning_mechanism as RM
sys.argv = _argv
import numpy as np
import torch

from src import relational_expression as RE

RB, SC, S0, P, D, X, LM, SP, DEV, nc = RM.RB, RM.SC, RM.S0, RM.P, RM.D, RM.X, RM.LM, RM.SP, RM.DEV, RM.S0.nc
SMC = RB.SMC
NAME = 'relational-expression-baseline-v1'
REPORT = X.ROOT / 'reports' / NAME
ENC = 'joined-mlp-calibrated'
POINTS = ('delay', 'relearn')
KS, MULTI = (1, 2, 3), 5
HEAD = {'successor': 0, 'same_quantity': 1, 'more_than': 2}
DEFS = {'predecessor': RE.Inverse('successor')}
KINDS = ('collections', 'taps', *(f'forms/{L}' for L in P.LANGS))
N = tuple(range(1, 10))
XV, R = RE.Var('x'), RE.Var('r')


class Store:
    """Codes by kind, opaque entities, and lazily computed g score matrices per (relation, kind, kind)."""

    def __init__(self, model):
        self.model, self.codes, self.mats, self.kind, self.loc = model, {}, {}, [], []
        self.atoms = 0

    def add(self, kind, Z):
        base = len(self.kind)
        self.codes[kind] = Z
        self.kind += [kind] * len(Z)
        self.loc += range(len(Z))
        return [RE.Ent(base + j) for j in range(len(Z))]

    @torch.no_grad()
    def mat(self, rel, ka, kb):
        key = (rel, ka, kb)
        if key not in self.mats:
            A, B = self.codes[ka], self.codes[kb]
            out = np.empty((len(A), len(B)), np.float32)
            for s in range(0, len(A), 64):
                a = A[s:s + 64]
                out[s:s + len(a)] = self.model.relation_logits(a[:, None].expand(-1, len(B), -1).reshape(-1, A.shape[1]),
                                                               B[None].expand(len(a), -1, -1).reshape(-1, A.shape[1]), HEAD[rel]).reshape(len(a), len(B)).cpu().numpy()
            self.mats[key] = out
            self.atoms += out.size
        return self.mats[key]

    def __call__(self, rel, args):
        a, b = args
        return self.mat(rel, self.kind[a.name], self.kind[b.name])[self.loc[a.name], self.loc[b.name]]


def credit(ranking, correct):
    """Top-1 credit shared among exactly tied top bindings; returns (credit, tied)."""
    top = ranking.ranked[0][1]
    tied = [b for b, s in ranking.ranked if s == top]
    return sum(correct(b) for b in tied) / len(tied), len(tied) > 1


def load(seed, point):
    path = SC.RUNS / ENC / 'checkpoints' / f'sparse48-seed{seed}-{point}.pt'
    enc, model = DEV.build(lambda: SC.make_model(seed, 'sparse48', ENC))
    ck = torch.load(path, weights_only=False, map_location='cpu')
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence']); model.load_state_dict(ck['sri'])
    return enc, model, ck['withheld'], {'path': str(path.relative_to(X.ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def eval_mode(enc, model):
    for m in (enc['image'], enc['sequence'], model):
        m.eval()


@torch.no_grad()
def kind_codes(enc, model):
    code, E = SP.G.codes_fn(enc, model), P.EVAL_PER
    src = {'collections': ({n: P.DOTS['succ'][n] for n in N}, 'image'), 'taps': ({n: D.TAPS[('eval', n)] for n in N}, 'sequence'),
           **{f'forms/{L}': ({n: X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]] for n in N}, 'image') for L in P.LANGS}}
    out = {}
    for k, (items, m) in src.items():
        Z = torch.stack([code(np.ascontiguousarray(items[n], dtype=np.float32), m) for n in N])  # (9, E, d)
        assert Z.shape[1] == E
        out[k] = Z.reshape(9 * E, -1)
    return out


def draws(seed, qk, xk, per):
    """Candidate renderings for each query (count n, rendering i): `per` per count, never the query item itself."""
    E = P.EVAL_PER
    rng = nc.stream('rex_candidates', seed, qk, xk, per)
    J = np.stack([rng.permutation(E)[:per + 1] for _ in range(9 * E * 9)]).reshape(9, E, 9, per + 1)
    out = np.empty((9, E, 9, per), int)
    for n in range(9):
        for i in range(E):
            for m in range(9):
                j = J[n, i, m]
                if qk == xk and m == n:
                    j = j[j != i]
                out[n, i, m] = j[:per]
    return out


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    s = np.concatenate([pos, neg]); r = np.argsort(np.argsort(s)) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def completion_reads(st, ents, seed):
    E, out, truth, multi = P.EVAL_PER, {}, {}, {}
    cnt = lambda e: st.loc[e.name] // E + 1
    for qk in KINDS:
        for xk in KINDS:
            dr = draws(seed, qk, xk, 1)
            res = {k: [] for k in ('same/right', 'same/left', 'successor/right', 'successor/left')}
            ties = {k: 0 for k in res}
            sc = {'same': ([], []), 'next': ([], []), 'more_acc': []}
            nxt_acc = []
            for n in N:
                for i in range(E):
                    q = ents[qk][(n - 1) * E + i]
                    dom = RE.Domain(tuple(ents[xk][(m - 1) * E + dr[n - 1, i, m - 1, 0]] for m in N))
                    reads = {'same/right': (RE.atom('same_quantity', q, XV), n), 'same/left': (RE.atom('same_quantity', XV, q), n)}
                    if n < 9:
                        reads['successor/right'] = (RE.atom('successor', q, XV), n + 1)
                    if n > 1:
                        reads['successor/left'] = (RE.atom('successor', XV, q), n - 1)
                    for k, (f, target) in reads.items():
                        c, t = credit(RE.rank(f, (XV,), {'x': dom}, st), lambda b: cnt(b[0]) == target)
                        res[k].append(c); ties[k] += t
                    for x in dom.items:
                        m = cnt(x)
                        s_same, s_next, s_more = st('same_quantity', (q, x)), st('successor', (q, x)), st('more_than', (q, x))
                        sc['same'][m != n].append(s_same)
                        sc['next'][m != n + 1].append(s_next)
                        nxt_acc.append((s_next > 0) == (m == n + 1))
                        if m != n:
                            sc['more_acc'].append((s_more > 0) == (n > m))
            key = f'{qk}->{xk}'
            out[key] = {k: float(np.mean(v)) for k, v in res.items()}
            out[key]['tie_rate'] = {k: ties[k] / len(res[k]) for k in res}
            truth[key] = {'same_auroc': auroc(*sc['same']), 'next_auroc': auroc(*sc['next']), 'next_accuracy_at_0': float(np.mean(nxt_acc)),
                          'more_accuracy_at_0': float(np.mean(sc['more_acc']))}
            # several valid answers: MULTI renderings per count, R-precision
            drm = draws(seed, qk, xk, MULTI)
            rp = {'same/right': [], 'successor/right': []}
            for n in N:
                for i in range(E):
                    q = ents[qk][(n - 1) * E + i]
                    dom = RE.Domain(tuple(ents[xk][(m - 1) * E + j] for m in N for j in drm[n - 1, i, m - 1]))
                    for k, f, target in (('same/right', RE.atom('same_quantity', q, XV), n), ('successor/right', RE.atom('successor', q, XV), n + 1)):
                        if target > 9:
                            continue
                        top = RE.rank(f, (XV,), {'x': dom}, st).ranked[:MULTI]
                        rp[k].append(np.mean([cnt(b[0]) == target for b, _ in top]))
            multi[key] = {k: float(np.mean(v)) for k, v in rp.items()}
    return out, truth, multi


def continuation_formula(prev, cur, k, rel=None):
    ys = [RE.Var(f'y{h}') for h in range(k - 1)]
    chain = [cur, *ys, XV]
    r = R if rel is None else RE.Rel(rel)
    atoms = [RE.atom(r, chain[h], chain[h + 1]) for h in range(k)]
    if rel is None:
        return RE.exists((R, *ys), RE.conj(RE.atom(R, prev, cur), *atoms)), ys
    return (RE.exists(ys, RE.conj(*atoms)) if ys else atoms[0]), ys


def continuation_reads(st, ents):
    E, out = P.EVAL_PER, {}
    cnt = lambda e: st.loc[e.name] // E + 1
    rels = RE.Domain((RE.Rel('successor'), RE.Rel('predecessor')))
    for K in KINDS:
        for proto in ('collections', 'within'):
            if proto == 'within' and K == 'collections':
                continue
            dk = 'collections' if proto == 'collections' else K
            for direction, sign in (('up', 1), ('down', -1)):
                for k in KS:
                    acc, tie, dirn, sup, sup_tie = [], 0, [], [], 0
                    for n in N:
                        m = n + sign * k
                        if not (1 <= n - sign <= 9 and 1 <= m <= 9):
                            continue
                        for i in range(E):
                            prev, cur = ents[K][(n - sign - 1) * E + i], ents[K][(n - 1) * E + i]
                            dom = RE.Domain(tuple(ents[dk][(c - 1) * E + i] for c in N))
                            f, ys = continuation_formula(prev, cur, k)
                            doms = {'x': dom, 'r': rels, **{y.name: dom for y in ys}}
                            rk = RE.rank(f, (XV,), doms, st, DEFS)
                            c, t = credit(rk, lambda b: cnt(b[0]) == m)
                            acc.append(c); tie += t
                            dirn.append(rk.witnesses[rk.ranked[0][0]][0].name == ('successor' if sign > 0 else 'predecessor'))
                            f, ys = continuation_formula(prev, cur, k, 'successor' if sign > 0 else 'predecessor')
                            rs = RE.rank(f, (XV,), {'x': dom, **{y.name: dom for y in ys}}, st, DEFS)
                            c, t = credit(rs, lambda b: cnt(b[0]) == m)
                            sup.append(c); sup_tie += t
                    out[f'{K}/{proto}/{direction}/{k}'] = {'inferred': float(np.mean(acc)), 'inferred_tie_rate': tie / len(acc),
                                                          'direction_witness_correct': float(np.mean(dirn)),
                                                          'supplied': float(np.mean(sup)), 'supplied_tie_rate': sup_tie / len(sup), 'queries': len(acc)}
    return out


@torch.no_grad()
def memory_reads(st, ents, reads, enc, model):
    """Each memory relational read in both slots, with the historical slot reproduced from the same values and draws."""
    E = P.EVAL_PER
    host = SMC.Host.__new__(SMC.Host)
    host.enc, host.model, host.frozen, host.cache = enc, model, True, {}
    out = {}
    for name, s in reads.sets.items():
        Q = host.raw(s['value'])
        hist, _ = SMC.score(host, Q, s['ok'], s['counts'], s['draws'])
        V = st.add(f'value/{name}', Q)
        dr, ok, counts = s['draws'], s['ok'], s['counts']
        res = {k: [] for k in ('r2/right', 'r2/left', 'r3/right', 'r3/left', "r3'/right", "r3'/left")}
        for i in range(len(Q)):
            for tag, kind, J in (('r2', 'collections', dr['dj']), ('r3', 'taps', dr['tj'])):
                dom = RE.Domain(tuple(ents[kind][int(c) * E + J[i, c]] for c in dr['order'][i]))
                pos = {e: p for p, e in enumerate(dom.items)}
                for slot, f in (('right', RE.atom('same_quantity', V[i], XV)), ('left', RE.atom('same_quantity', XV, V[i]))):
                    best = RE.rank(f, (XV,), {'x': dom}, st).ranked[0][0][0]
                    res[f'{tag}/{slot}'].append(bool(ok[i]) and dr['order'][i][pos[best]] + 1 == counts[i])
        for j, (k, it) in enumerate(zip(dr['t_count'], dr['t_item'])):
            t = ents['taps'][(int(k) - 1) * E + int(it)]
            cands = [c for c in dr['t_cand'][j] if ok[c]]
            if not ok[dr['t_cand'][j][k - 1]] or not cands:
                res["r3'/right"].append(False); res["r3'/left"].append(False)
                continue
            dom = RE.Domain(tuple(V[c] for c in cands))
            for slot, f in (('right', RE.atom('same_quantity', t, XV)), ('left', RE.atom('same_quantity', XV, t))):
                best = RE.rank(f, (XV,), {'x': dom}, st).ranked[0][0][0]
                res[f"r3'/{slot}"].append(best == V[dr['t_cand'][j][k - 1]])
        out[name] = {k: float(np.mean(v)) for k, v in res.items()}
        out[name]['r4'] = float(hist['r4'].mean())
        out[name]['historical_slot_identical'] = {
            'r2': bool(np.array_equal(np.array(res['r2/right']), hist['r2'])), 'r3': bool(np.array_equal(np.array(res['r3/right']), hist['r3'])),
            "r3'": bool(np.array_equal(np.array(res["r3'/right"]), hist['r3_taps_to_symbol']))}
    return out


def task(job):
    seed, point = job
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(LM.FI.THREADS)
    DEV.install()
    enc, model, withheld, ck = load(seed, point)
    reads = RB.Reads(seed, withheld)
    eval_mode(enc, model)
    hist = reads(enc, model)
    eval_mode(enc, model)
    hist['composition'] = SP.composition(enc, model)
    t_hist = time.perf_counter() - t0
    eval_mode(enc, model)
    st = Store(model)
    ents = {k: st.add(k, Z) for k, Z in kind_codes(enc, model).items()}
    t1 = time.perf_counter()
    comp, truth, multi = completion_reads(st, ents, seed)
    t2 = time.perf_counter()
    cont = continuation_reads(st, ents)
    t3 = time.perf_counter()
    mem = memory_reads(st, ents, reads, enc, model)
    t4 = time.perf_counter()
    out = {'checkpoint': ck, 'historical': hist, 'completion': comp, 'truth': truth, 'multi': multi, 'continuation': cont, 'memory': mem}
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'historical_reads_seconds': t_hist,
          'codes_seconds': t1 - t0 - t_hist, 'completion_seconds': t2 - t1, 'continuation_seconds': t3 - t2, 'memory_seconds': t4 - t3,
          'g_atom_scores': st.atoms, 'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    print(seed, point, f'{rt["wall_seconds"]:.0f} s', flush=True)
    return seed, point, out, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    jobs = [(s, p) for s in RB.SEEDS for p in POINTS]
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('RX_WORKERS', 3))) as pool:
        res = pool.map(task, jobs, chunksize=1)
    results = {}
    for s, p, o, _ in res:
        results.setdefault(f'seed{s}', {})[p] = o
    S0.save_json(f'{REPORT}.json', {'status': 'read-only baseline, no training', 'seeds': RB.SEEDS, 'points': POINTS, 'kinds': KINDS, 'ks': KS,
                                    'multi_per_count': MULTI, 'semantics': 'max-min; ties share top-1 credit; predecessor := inverse(successor) supplied',
                                    'results': results})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'relational expression baseline', 'device': os.environ.get('NFI_DEVICE', 'cpu'), 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}-{p}': t for s, p, _, t in res},
                                                      'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for *_, t in res))}))


if __name__ == '__main__':
    main()
