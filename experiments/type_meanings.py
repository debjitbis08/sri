"""Type meanings v1 (docs/type-meanings-v1.md): category is-a facts and a learned applicability ("meaningless") output
on the logic route, taught by example with a held-out category (sounds) and held-out relations. Phase-1 readers only.

Usage: EV_WORKERS=6 .venv/bin/python -m experiments.type_meanings        (rerun to resume)
       .venv/bin/python -m experiments.type_meanings summary
"""
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
from experiments import dual_route_truth as DR
from experiments import durable as DUR
from experiments import everyday_relations as R1
from experiments import everyday_relations_v2 as V2
from experiments import taught_truth as TT
from src import everyday_truth as T
from src import everyday_truth_v2 as T2
from src import everyday_world as W1
from src import everyday_world_v2 as W2

ROOT, nc = V2.ROOT, V2.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'type-meanings-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs' / 'type-meanings-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
STEPS = 60 if SMOKE else CS.STEPS
SEEDS = (450,) if SMOKE else (450, 451, 452, 453)
ARMS = ('base', 'isa', 'meaningless', 'isa+meaningless')
HELD_REL = ('lives-with', 'brighter-than', 'slower-than', 'eats', 'sibling-of', 'mixed-from')
HELD_CAT = 'sound'
N_TRAIN_PER_REL, N_EVAL = 100, 3000
D_, RD = DR.D, DR.RD

CATEGORIES = ('person', 'sound', 'colour', 'body-part', 'property', 'ability', 'food', 'habitat', 'substance', 'object')


def category_members(st0):
    names = set(st0.entities)
    people = list(W1.TREE_A) + list(W1.TREE_B) + [p for line in T2.LINEAGES for p in line]
    m = {
        'person': people,
        'sound': sorted({s for v in W1.SOUNDS.values() for s in v.split()}),
        'colour': list(W1.COLOURS),
        'body-part': list(W1.PARTS),
        'property': list(W2.ADJECTIVES) + ['warm', 'cool', 'male', 'female'],
        'ability': list(W1.ABILITIES),
        'food': list(W1.FOODS),
        'habitat': list(W1.HOMES),
        'substance': ['water', 'fire', 'soup', 'tea', 'ice', 'milk'],
    }
    m = {k: [x for x in v if x in names] for k, v in m.items()}
    rels = np.array(st0.relations)
    E = np.array(st0.entities)
    isa = (rels[st0.r] == 'is-a') & (st0.kind == 'individual') & (st0.world_y == 1)
    has_isa = set(E[st0.a[isa]].tolist())
    skip = set(T.VALUES) | {'nil'} | set(T2.KINDS) | set(CATEGORIES) | has_isa | set(W1.BABIES)
    skip |= {x for x in names if '-' in x and x.split('-')[-1].isdigit()}          # unnamed individuals (mammal-1)
    skip |= {x for x in names if x.endswith('-w')}                                 # words
    placed = set().union(*m.values())
    m['object'] = sorted(x for x in names if x not in placed and x not in skip and x not in ('noun', 'verb', 'adjective'))
    return m


def build(arm, seed):
    st0, _ = T2.build()
    tr0, val0 = CS.splits(st0, seed)
    members = category_members(st0)
    ents = list(st0.entities) + list(CATEGORIES)
    E = {x: i for i, x in enumerate(ents)}
    rels = list(st0.relations)
    V = {x: i for i, x in enumerate(T.VALUES)}
    rows = []
    if arm.startswith('isa'):
        cat_of = {}
        for c, xs in members.items():
            for x in xs:
                cat_of.setdefault(x, set()).add(c)
        for x in W1.BABIES:
            if x in E:
                cat_of.setdefault(x, set()).add('animal')
        for x, cs in sorted(cat_of.items()):
            for c in CATEGORIES + ('animal',):
                y = c in cs
                rows.append((E[x], rels.index('is-a'), E[c], V['T' if y else 'F'], 0, 'individual', '', '', int(y)))
    cols = [st0.a, st0.r, st0.b, st0.v, st0.gap, st0.kind, st0.test, st0.sub, st0.world_y]
    if rows:
        add = list(zip(*rows))
        cols = [np.concatenate([c, np.array(x, dtype=c.dtype)]) for c, x in zip(cols, add)]
    st = T.Statements(ents, rels, *cols)
    st.fill_answers = st0.fill_answers
    n_new = len(st.a) - len(st0.a)
    train_rows = np.concatenate([tr0, np.ones(n_new, bool)])
    val = np.concatenate([val0, np.zeros(n_new, bool)])
    return st, st0, train_rows, val, members


def entity_types(st0, members, n_ent):
    """Category sets per entity index of the extended entity list (animals 'animal', words 'word', kinds 'kind')."""
    E = {x: i for i, x in enumerate(st0.entities)}
    rels = np.array(st0.relations)
    cats = [set() for _ in range(n_ent)]
    for c, xs in members.items():
        for x in xs:
            cats[E[x]].add(c)
    isa = (rels[st0.r] == 'is-a') & (st0.kind == 'individual') & (st0.world_y == 1)
    for a, b in zip(st0.a[isa], st0.b[isa]):
        nm = st0.entities[b]
        cats[a].add('word' if nm in ('noun', 'verb', 'adjective') else 'animal')
    for x in W1.BABIES:
        if x in E:
            cats[E[x]].add('animal')
    for x in T2.KINDS:
        if x in E:
            cats[E[x]].add('kind')
    return cats


def meaningless_sets(st0, train0, members, n_ent, seed):
    rels = list(st0.relations)
    permitted = D.permitted_relations(st0, train0)
    cats = entity_types(st0, members, n_ent)
    catd = np.array([bool(c) for c in cats])
    sound = np.zeros(n_ent, bool)
    sound[[st0.index(x) for x in members[HELD_CAT]]] = True
    ind = (st0.kind == 'individual') & (st0.world_y >= 0)
    rng = np.random.default_rng([seed, 5])
    train, ev = [], {'M-in': [], 'M-cat': [], 'M-rel': []}
    masks = {}
    for rel in permitted:
        r = rels.index(rel)
        m = ind & (st0.r == r)
        dom = set().union(*[cats[a] for a in st0.a[m]])
        rnge = set().union(*[cats[b] for b in st0.b[m]])
        dok = np.array([bool(c & dom) for c in cats])
        rok = np.array([bool(c & rnge) for c in cats])
        M = catd[:, None] & catd[None, :] & (~dok[:, None] | ~rok[None, :])
        np.fill_diagonal(M, False)
        M[st0.a[m], st0.b[m]] = False                     # taught facts are meaningful by definition
        masks[r] = M
        has_sound = sound[:, None] | sound[None, :]
        clean = np.argwhere(M & ~has_sound)
        snd = np.argwhere(M & has_sound)
        if rel in HELD_REL:
            ev['M-rel'] += [(x, r, y) for x, y in clean]
        else:
            pick = rng.choice(len(clean), min(N_TRAIN_PER_REL, len(clean)), replace=False) if len(clean) else []
            chosen = {tuple(clean[i]) for i in pick}
            train += [(x, r, y) for x, y in chosen]
            ev['M-in'] += [(x, r, y) for x, y in clean if (x, y) not in chosen]
            ev['M-cat'] += [(x, r, y) for x, y in snd]
    out = {}
    for k, v in ev.items():
        v = np.array(v)
        out[k] = v[rng.choice(len(v), min(N_EVAL, len(v)), replace=False)] if len(v) else v.reshape(0, 3)
    return np.array(train).reshape(-1, 3), out, masks, permitted


class TM(nn.Module):
    def __init__(self, n_ent, n_rel, g):
        super().__init__()
        self.dual = DR.Dual('dual-separate', n_ent, n_rel, g)
        self.A_L = DR.mlp(D_ + RD, 1)

    def applic_logit(self, a, r, b):
        d = self.dual
        rc = d.R_L[r]
        return self.A_L(torch.cat([d.G_L(torch.cat([d.E_L[a], d.E_L[b], rc], 1)), rc], 1))[:, 0]


def phase1(model, st, seed, train_rows, mless):
    """CS.phase1 plus the applicability term."""
    d = model.dual
    opt = torch.optim.Adam(model.parameters(), lr=TT.LR, weight_decay=TT.WD)
    w = torch.as_tensor(TT.weights(st, train_rows))
    vnames = np.array(T.VALUES)
    wh = CS.withheld_mask(st)
    wrel = torch.tensor([i for i, x in enumerate(st.relations) if x in CS.WITHHELD_RELS])
    r_init = d.R_L_own.detach()[wrel].clone()
    sup = np.flatnonzero(train_rows & np.isin(st.kind, ('individual', 'anon', 'anon-isa')))
    sup_t = torch.tensor([DR.SUPPORT[vnames[v]] for v in st.v[sup]])
    sup_t[torch.as_tensor(wh[sup])] = 0.
    kin = np.flatnonzero(train_rows & (st.kind == 'kind'))
    qidx = torch.tensor([st.index(q) for q in DR.QUANT])
    kin_t = torch.tensor([DR.QUANT.index(vnames[v]) for v in st.v[kin]])
    fil = np.flatnonzero(train_rows & (st.kind == 'fill') & ~wh)
    fmask = torch.zeros(len(fil), len(st.entities), dtype=torch.bool)
    for k, i in enumerate(fil):
        fmask[k, st.fill_answers[int(i)]] = True
    con = np.flatnonzero(st.kind == 'connective')
    names = np.array(st.entities)
    con_p1 = torch.tensor([DR.SUPPORT.get(names[a], (0., 0.)) for a in st.a[con]])
    con_p2 = torch.tensor([DR.SUPPORT.get(names[b], (0., 0.)) for b in st.b[con]])
    con_t = torch.tensor([DR.SUPPORT[vnames[v]] for v in st.v[con]])
    con_r = torch.as_tensor(st.r[con])
    ii = np.flatnonzero(train_rows & (st.kind == 'individual') & (st.world_y >= 0))
    iy = torch.as_tensor(st.world_y[ii].astype(np.float32))
    iw = V2.rel_weights(st.r[ii], st.world_y[ii], len(st.relations))
    A = lambda idx: (torch.as_tensor(st.a[idx]), torch.as_tensor(st.r[idx]), torch.as_tensor(np.maximum(st.b[idx], 0)))
    sa, sr, sb = A(sup)
    ka, kr, kb = A(kin)
    fa, fr_, fb = A(fil)
    ia, ir, ib = A(ii)
    ma, mr, mb = (torch.as_tensor(mless[:, k]) for k in range(3)) if len(mless) else (None, None, None)
    w_app = float(w[sup].sum())
    log, t0 = [], time.perf_counter()
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        li = (F.binary_cross_entropy_with_logits(d.intuition(ia, ir, ib), iy, reduction='none') * iw).sum()
        ls = (F.binary_cross_entropy(d.logic(sa, sr, sb), sup_t, reduction='none').sum(1) * w[sup]).sum()
        lk = (F.cross_entropy(d.code_logits(d.code_out(ka, kr, kb, torch.zeros(len(kin), dtype=torch.long)), d.E_L[qidx]),
                              kin_t, reduction='none') * w[kin]).sum()
        lp = torch.log_softmax(d.code_logits(d.code_out(fa, fr_, fb, torch.ones(len(fil), dtype=torch.long)), d.E_L), 1)
        lf = (-torch.logsumexp(lp.masked_fill(~fmask, -1e9), 1) * w[fil]).sum()
        lc = (F.binary_cross_entropy(d.connective(con_p1, con_p2, con_r), con_t, reduction='none').sum(1) * w[con]).sum()
        pos = F.binary_cross_entropy_with_logits(model.applic_logit(ia, ir, ib), torch.ones(len(ii)))
        if ma is not None:
            neg = F.binary_cross_entropy_with_logits(model.applic_logit(ma, mr, mb), torch.zeros(len(ma)))
            la = w_app * .5 * (pos + neg)
        else:
            la = w_app * pos
        loss = li + ls + lk + lf + lc + la
        loss.backward()
        opt.step()
        with torch.no_grad():
            d.R_L_own[wrel] = r_init
        if step % CS.LOG_EVERY == 0 or step == STEPS:
            log.append({'step': step, 'loss': float(loss.detach()), 'applicability': float(la.detach()),
                        'wall': time.perf_counter() - t0})
    return log


# ---------------------------------------------------------------- reads

@torch.no_grad()
def judge(model, trip):
    """trip (n, 3) -> dict of arrays: applicability, logic support, intuition sigmoid, confirmed (both routes)."""
    d = model.dual
    a, r, b = (torch.as_tensor(np.asarray(trip)[:, k]) for k in range(3))
    out = {k: [] for k in ('app', 'sp', 'sm', 'sig')}
    for s in range(0, len(a), 50000):
        sl = slice(s, s + 50000)
        out['app'].append(torch.sigmoid(model.applic_logit(a[sl], r[sl], b[sl])))
        lg = d.logic(a[sl], r[sl], b[sl])
        out['sp'].append(lg[:, 0])
        out['sm'].append(lg[:, 1])
        out['sig'].append(torch.sigmoid(d.intuition(a[sl], r[sl], b[sl])))
    o = {k: torch.cat(v).numpy() if v else np.zeros(0) for k, v in out.items()}
    s = np.stack([o['sp'], o['sm']], 1)
    dis = np.abs(o['sig'] - s[:, 0]) + np.abs((1 - o['sig']) - s[:, 1])
    o['confirmed'] = (dis <= CS.TAU) & (s[:, 0] > .5)
    return o


@torch.no_grad()
def full_graph(model, masks, permitted, rels, st0, n_ent):
    """Confirmed facts over all (x, r, y) of permitted relations, by meaningful/meaningless, with and without the gate."""
    ind = (st0.kind == 'individual') & (st0.world_y >= 0)
    res = {'meaningless_confirmed': 0, 'meaningless_confirmed_gated': 0, 'true_confirmed': 0, 'true_confirmed_gated': 0,
           'true_total': 0, 'meaningless_total': 0}
    xs, ys = np.meshgrid(np.arange(n_ent), np.arange(n_ent), indexing='ij')
    xs, ys = xs.ravel(), ys.ravel()
    for rel in permitted:
        r = rels.index(rel)
        M = masks[r]
        Tm = np.zeros((n_ent, n_ent), bool)
        m = ind & (st0.r == r) & (st0.world_y == 1)
        Tm[st0.a[m], st0.b[m]] = True
        keep = M.ravel() | Tm.ravel()
        trip = np.stack([xs[keep], np.full(keep.sum(), r), ys[keep]], 1)
        o = judge(model, trip)
        isM = M.ravel()[keep]
        gate = o['app'] >= .5
        res['meaningless_total'] += int(isM.sum())
        res['true_total'] += int((~isM).sum())
        res['meaningless_confirmed'] += int((o['confirmed'] & isM).sum())
        res['meaningless_confirmed_gated'] += int((o['confirmed'] & gate & isM).sum())
        res['true_confirmed'] += int((o['confirmed'] & ~isM).sum())
        res['true_confirmed_gated'] += int((o['confirmed'] & gate & ~isM).sum())
    return res


def reads(model, st, st0, train0, members, mless_ev, masks, permitted):
    rels = list(st.relations)
    n_ent = len(st.entities)
    out = {'M_sets': {}}
    for k, trip in mless_ev.items():
        o = judge(model, trip)
        out['M_sets'][k] = {'n': int(len(trip)), 'recognized': float((o['app'] < .5).mean()),
                            'confirmed': float(o['confirmed'].mean()),
                            'confirmed_gated': float((o['confirmed'] & (o['app'] >= .5)).mean())}
    ind0 = (st0.kind == 'individual') & (st0.world_y >= 0) & np.isin(np.array(rels)[st0.r], permitted)
    taught = np.stack([st0.a[ind0], st0.r[ind0], st0.b[ind0]], 1)
    o = judge(model, taught)
    out['taught_kept_meaningful'] = float((o['app'] >= .5).mean())
    out['taught_kept_meaningful_true'] = float((o['app'][st0.world_y[ind0] == 1] >= .5).mean())
    out['full_graph'] = full_graph(model, masks, permitted, rels, st0, n_ent)
    # truth on v1's tests
    ind = (st.kind == 'individual') & (st.world_y >= 0)
    out['truth'] = {}
    d = model.dual
    for tst in DR.TF_TESTS:
        m = ind & (st.test == 'P1') & (st.sub == 'default') if tst == 'P1-default' else ind & (st.test == tst)
        idx = np.flatnonzero(m)
        trip = np.stack([st.a[idx], st.r[idx], st.b[idx]], 1)
        o = judge(model, trip)
        y = st.world_y[idx]
        lv = DR.logic_state(np.stack([o['sp'], o['sm']], 1))
        out['truth'][tst] = {'intuition': DR.tf_bal(np.where(o['sig'] > .5, 'T', 'F'), y), 'logic': DR.tf_bal(lv, y),
                             'logic_with_M': DR.tf_bal(np.where(o['app'] < .5, 'M', lv), y),
                             'meaningless_rate': float((o['app'] < .5).mean())}
    # categories learned: sounds' is-a support (taught in isa arms)
    E = {x: i for i, x in enumerate(st.entities)}
    snd = [E[x] for x in members[HELD_CAT]]
    isa = rels.index('is-a')
    t_true = np.array([(x, isa, E['sound']) for x in snd])
    t_false = np.array([(x, isa, E[c]) for x in snd for c in CATEGORIES if c != 'sound'])
    out['sound_isa'] = {'support_true': float(judge(model, t_true)['sp'].mean()),
                        'support_false': float(judge(model, t_false)['sp'].mean())}
    out['cow_parent_of_moo'] = {k: float(v[0]) for k, v in judge(model, np.array([[E['cow'], rels.index('parent-of'), E['moo']]])).items()}
    return out


# ---------------------------------------------------------------- tasks

def run(arm, seed):
    torch.set_num_threads(1)
    st, st0, train_rows, val, members = build(arm, seed)
    TT.value_idx_cache = TT.value_idx(st)
    train0, _ = CS.splits(st0, seed)
    mless_train, mless_ev, masks, permitted = meaningless_sets(st0, train0, members, len(st.entities), seed)
    use = mless_train if arm.endswith('meaningless') else np.zeros((0, 3), int)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = TM(len(st.entities), len(st.relations), g)
    t0 = time.process_time()
    log = phase1(model, st, seed, train_rows, use)
    train_cpu = time.process_time() - t0
    t1 = time.process_time()
    res = {'arm': arm, 'seed': seed, 'log': log, 'n_meaningless_train': int(len(use)),
           'n_isa_rows': int(len(st.a) - len(st0.a)), 'categories': {k: len(v) for k, v in members.items()},
           'object_members': members['object'], 'reads': reads(model, st, st0, train0, members, mless_ev, masks, permitted)}
    reads_cpu = time.process_time() - t1
    RUNS.mkdir(parents=True, exist_ok=True)
    ck = RUNS / f'{arm}-seed{seed}.pt'
    torch.save(model.state_dict(), ck)
    res['checkpoint'] = {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)}
    return res, {'train_cpu_seconds': train_cpu, 'reads_cpu_seconds': reads_cpu, 'steps': STEPS,
                 'params': int(sum(p.numel() for p in model.parameters())), 'checkpoint_bytes': ck.stat().st_size}


def task(t):
    t0, c0 = time.perf_counter(), time.process_time()
    res, rt = run(*t)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{t[0]}/seed{t[1]}', res, rt


def main():
    t0 = time.perf_counter()
    workers = int(os.environ.get('EV_WORKERS', 6))
    tasks = [(a, s) for a in ARMS for s in SEEDS]
    with __import__('multiprocessing').get_context('fork').Pool(min(workers, len(tasks)), maxtasksperchild=1) as pool:
        out = pool.map(DUR.durable(task, RUNS / 'done', '{0}-seed{1}'), tasks, chunksize=1)
    rt = V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'workers': workers,
                     'wall_seconds': time.perf_counter() - t0,
                     'aggregate_worker_cpu_seconds': float(sum(r['cpu_seconds'] for _, _, r in out)),
                     'tasks': {k: r for k, _, r in out}})
    V2.save_json(f'{REPORT}.json', {'status': 'type meanings v1' + (' (smoke)' if SMOKE else ''),
                                    'results': {k: r for k, r, _ in out}})
    V2.save_json(f'{REPORT}-runtime.json', rt)
    summary()


def summary():
    res = json.loads(open(f'{REPORT}.json').read())['results']
    by = {}
    for r in res.values():
        by.setdefault(r['arm'], []).append(r['reads'])
    m = lambda v: round(float(np.mean(v)), 3)
    out = {}
    for arm in ARMS:
        R = by.get(arm, [])
        if not R:
            continue
        o = {'M_sets': {k: {q: m([r['M_sets'][k][q] for r in R]) for q in ('recognized', 'confirmed', 'confirmed_gated')}
                        for k in R[0]['M_sets']},
             'taught_kept_meaningful': m([r['taught_kept_meaningful'] for r in R]),
             'full_graph': {k: m([r['full_graph'][k] for r in R]) for k in R[0]['full_graph']},
             'truth': {t: {k: m([r['truth'][t][k] for r in R]) for k in R[0]['truth'][t]} for t in R[0]['truth']},
             'sound_isa': {k: m([r['sound_isa'][k] for r in R]) for k in R[0]['sound_isa']},
             'cow_parent_of_moo': {k: m([r['cow_parent_of_moo'][k] for r in R]) for k in ('app', 'sp', 'sig')},
             'per_seed_recognized': {k: [round(r['M_sets'][k]['recognized'], 3) for r in R] for k in R[0]['M_sets']}}
        out[arm] = o
        print(f'\n== {arm}')
        print('   recognized meaningless:', {k: v['recognized'] for k, v in o['M_sets'].items()}, ' per seed', o['per_seed_recognized'])
        print('   confirmed (ungated / gated):', {k: (v['confirmed'], v['confirmed_gated']) for k, v in o['M_sets'].items()})
        print('   taught kept meaningful', o['taught_kept_meaningful'], ' full graph', o['full_graph'])
        print('   sound is-a support', o['sound_isa'], ' cow parent-of moo', o['cow_parent_of_moo'])
        print('   truth (intuition / logic / logic with M):', {t: (v['intuition'], v['logic'], v['logic_with_M']) for t, v in o['truth'].items()})
    V2.save_json(f'{REPORT}-summary.json', out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
