"""Local update diagnosis v1: phase A diagnoses the IB block learner; phase B runs single-property variants.

Run `.venv/bin/python -m experiments.update_diagnosis [--out-dir DIR] [--workers N]`.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import unittest

from experiments.local_supervision import (
    Projected, RowIndex, atom_of, behaviour, feedback, min_prefix_accuracy, timeline_summaries,
)
from experiments.mechanism_learning import (
    NEGATIVE, POSITIVE, TEACHERS, Evaluator, Index, atom_hash, build_battery, canonical, completion_status, dump,
    ev_fwd, mirror, ratio, sha, write_rows_report,
)
from src.local_supervision import make_simulator, observation_keys, pool_timelines
from src.mechanism_learning import BLOCKS, N_VECTORS, QUERIES, BlockLearner, hypothesis_index, popcount, vector
from src.update_diagnosis import (
    VARIANTS, VariantLearner, b0_choice, class_distance, classify, n1, n2, proximity, repetition_probe,
    reverse_reachable, set_hash, witness,
)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'local-update-diagnosis-v1'
CONTRACT = 'docs/local-update-diagnosis-v1-contract.md'
LLS = 'local-learning-supervision-v1'
V1 = 'learned-input-mediation-v1'
SPLITS = 'reports/learned-reciprocal-participation-v1-splits.json'
CHECKPOINTS = (0, 5, 10, 15)
CLASSES = ('E0', 'E1a', 'E1b', 'E2a', 'E2b', 'E2c')
RULES = [
    'supervision: local-learning-supervision v1 condition IB unchanged (B1, B6x ascending A, B7x ascending B)',
    'blocks (w_1A,w_2A),(w_1B,w_2B),(w_KA,w_KB),(w_12); N1 = 26 single-block edits (v1 order); '
    'N2 = 240 edits changing exactly two blocks; N<=2 = N1 + N2',
    'loss: sum over keys of |predicted symmetric-difference feedback| after projection; unit weight per element',
    'B0/SCOPE/REPEAT tie-break: (current loss, changed coefficients, lexicographic vector); commit on strict improvement',
    'SCOPE: candidates unchanged + N1 + N2 (267); REPEAT: repeat the B0 decision until no strict improvement; '
    'cost 27 x (commits + 1)',
    'REPLAY: eligible = N1 edits strictly improving the current map (commit iff B0 would at that state); '
    'select by (summed loss over stored earlier maps, current loss, changed, lexicographic); '
    'all 27 candidates scored on every stored map; current map appended after the decision',
    'previously correct items: earlier training items reproduced on the own map by the state before the decision',
    'failing experience: after the experience the current example is not exact, or an item correct at its start is broken',
    'd(w): blocks differing from the nearer of teacher and core-sign mirror; G-down/G-le: N1 edges with strictly '
    'decreasing / non-increasing frozen-example loss; reachability by exact reverse BFS from the teacher class',
    'cost units: candidate vectors evaluated, (candidate, key) predictions, history predictions, decisions, commits, '
    'coefficients changed, simulation requests (total, unique), stored maps',
    'targets L1-L3 as local-learning-supervision v1 at checkpoint 15, raw-count pooling over 25 streams',
]


def manifest():
    lls_m = json.loads((ROOT / f'reports/{LLS}-manifest.json').read_text())
    splits = json.loads((ROOT / SPLITS).read_text())
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                split_file=SPLITS, split_file_sha256=sha(SPLITS),
                coefficient_order=['w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12'],
                blocks=[list(b) for b in BLOCKS],
                teachers={k: list(v) for k, v in TEACHERS.items()}, positive_teachers=list(POSITIVE),
                negative_controls=list(NEGATIVE), variants=list(VARIANTS), checkpoints=list(CHECKPOINTS),
                supervision='IB', streams={str(s['seed']): dict(train=s['train'], orders=s['orders'],
                                                              mixed24=lls_m['coverage'][str(s['seed'])]['mixed24'])
                                          for s in splits['splits']},
                targets=dict(L1='recovered >= 23 of 25 and >= 4 per split',
                             L2='exact >= 0.8 n on within and mixed24 and exact > zero baseline exact',
                             L3='own-map final_correct >= 0.95 presented and forgotten <= 0.05 ever_acquired'),
                rules=RULES,
                prior_artifacts={p: sha(p) for p in [
                    f'reports/{LLS}-manifest.json', f'reports/{LLS}-ceiling.json', f'reports/{LLS}-learning.json',
                    f'docs/{LLS}-contract.md', f'reports/{V1}-battery-summary.json', f'reports/{V1}-manifest.json',
                    SPLITS, 'src/local_supervision.py', 'experiments/local_supervision.py',
                    'src/mechanism_learning.py', 'experiments/mechanism_learning.py',
                    'src/shared_input_mediation.py']})
    return dict(body, sha256=hashlib.sha256(canonical(body).encode()).hexdigest())


def freeze_manifest(out):
    m = manifest()
    path = out / f'{PREFIX}-manifest.json'
    frozen = ROOT / 'reports' / f'{PREFIX}-manifest.json'
    ref = path if path.exists() else frozen if frozen.exists() else None
    if ref is not None and json.loads(ref.read_text()) != m:
        sys.exit('manifest differs from the frozen file; refusing to run')
    if not path.exists():
        dump(path, m)
    return m


# ---------------------------------------------------------------- evaluator tables

class Tables:
    """Evaluator-only per-(teacher, query) IB loss vectors over all 2,187 hypotheses, and exact sets."""
    def __init__(self, ix, pj, bat, queries):
        self.ix, self.pj = ix, pj
        self.loss, self.exact = {}, {}
        for n, th in bat['teacher_idx'].items():
            for qi in queries:
                x = QUERIES[qi]
                cols = []
                for key in observation_keys(x, 'IB'):
                    a = atom_of(ix, qi, key)
                    if key[0] == 'B1':
                        T = ix.table[th][a]
                        cols.append([popcount(ix.table[h][a] ^ T) for h in range(N_VECTORS)])
                    else:
                        T = pj.value(a, th)
                        cols.append([popcount(pj.value(a, h) ^ T) for h in range(N_VECTORS)])
                L = [sum(c[h] for c in cols) for h in range(N_VECTORS)]
                self.loss[(n, qi)] = L
                self.exact[(n, qi)] = sum(1 << h for h in range(N_VECTORS) if L[h] == 0)

    def fn(self, n, qi):
        L = self.loss[(n, qi)]
        return lambda u: L[hypothesis_index(u)]

    def ok(self, n, qi, h):
        return bool(self.exact[(n, qi)] >> h & 1)


def adjacency():
    return [[hypothesis_index(u) for u in n1(vector(h))] for h in range(N_VECTORS)]


def graphs(tables, adj, bat, keys):
    """Exact frozen-example reachability to the teacher class, per (teacher, query)."""
    out, certs = {}, {}
    for n, qi in keys:
        th = bat['teacher_idx'][n]
        targets = sorted({th, hypothesis_index(mirror(vector(th)))})
        L = tables.loss[(n, qi)]
        down = reverse_reachable(L, adj, targets, strict=True)
        le = reverse_reachable(L, adj, targets, strict=False)
        out[(n, qi)] = (down, le)
        certs[f'{n}:{qi}'] = dict(loss_sha256=hashlib.sha256(','.join(map(str, L)).encode()).hexdigest(),
                                  G_down=dict(reach=len(down), sha256=set_hash(down)),
                                  G_le=dict(reach=len(le), sha256=set_hash(le)),
                                  down_subset_le=set(down) <= set(le))
    return out, certs


# ---------------------------------------------------------------- per-stream diagnosis

def prior_correct(tables, n, earlier, h):
    return [s for s in earlier if tables.ok(n, s, h)]


def broken_fn(tables, n, items):
    return lambda u: sum(not tables.ok(n, s, hypothesis_index(u)) for s in items)


def broken_set_fn(tables, n, items):
    return lambda u: {s for s in items if not tables.ok(n, s, hypothesis_index(u))}


def graph_category(g, h):
    down, le = g
    if h in down:
        return 1, witness(down, h)
    if h in le:
        return 2, witness(le, h)
    return 3, None


def state_diagnostics(tables, gr, n, qi, h, members, earlier):
    """Proximity, frozen-example graph category and repetition probe at state h before experience qi."""
    w = vector(h)
    loss = tables.fn(n, qi)
    rec = {}
    in_class = h in {hypothesis_index(m) for m in members}
    rec['in_teacher_class'] = in_class
    if not in_class:
        rec['proximity'] = proximity(w, loss, members)
        cat, path = graph_category(gr[(n, qi)], h)
        rec['graph'] = dict(category=cat, witness=path)
    pc = prior_correct(tables, n, earlier, h)
    probe = repetition_probe(w, loss, broken_set_fn(tables, n, pc))
    term = probe.pop('terminal')
    th_idx = {hypothesis_index(m) for m in members}
    rec['repetition'] = dict(probe, terminal_exact=loss(term) == 0, terminal_in_class=hypothesis_index(term) in th_idx,
                             d_change=class_distance(term, members) - class_distance(w, members))
    return rec


def run_variant_stream(variant, n, th, order, sim, tables, gr, members, lockstep_sim=None):
    """Run one stream; return the learner, per-experience records and (B0) lockstep result."""
    L = VariantLearner(sim, variant)
    ref = BlockLearner(lockstep_sim) if lockstep_sim is not None else None
    lock_ok = True
    states = [hypothesis_index(L.w)]
    experiences = []
    for t, qi in enumerate(order):
        x = QUERIES[qi]
        fb = feedback(tables.ix, th, qi, 'IB')
        earlier = order[:t]
        h0 = hypothesis_index(L.w)
        loss = tables.fn(n, qi)
        pc0 = prior_correct(tables, n, earlier, h0)
        # counterfactual B0 classification at this experience's starting state (never applied)
        w0 = vector(h0)
        cf_choice = b0_choice(w0, loss)
        cf = classify(w0, loss, broken_fn(tables, n, pc0), n1(w0), cf_choice)
        cf['b0_choice'] = None if cf_choice is None else hypothesis_index(cf_choice)
        diag = state_diagnostics(tables, gr, n, qi, h0, members, earlier)
        trace = L.update(x, fb)
        if ref is not None:
            ref.update(x, fb)
            lock_ok &= ref.w == L.w and ref.evaluations == L.evaluations and ref.commits == L.commits
        decisions = []
        lost_any = set()
        for dec in trace:
            wb = vector(dec['before'])
            pcd = prior_correct(tables, n, earlier, dec['before'])
            neigh = n1(wb) + (n2(wb) if variant == 'SCOPE' else [])
            lab = classify(wb, loss, broken_fn(tables, n, pcd), neigh, vector(dec['after']) if dec['committed'] else None,
                           scope_n2=variant == 'SCOPE')
            decisions.append(dict(dec, **lab))
            if dec['committed']:
                lost_any |= broken_set_fn(tables, n, pc0)(vector(dec['after']))
        h1 = hypothesis_index(L.w)
        broken_end = sorted(broken_set_fn(tables, n, pc0)(L.w))
        experiences.append(dict(
            step=t + 1, query=qi, before=h0, after=h1, loss_before=loss(w0), loss_after=loss(L.w),
            commits=sum(d['committed'] for d in trace), prior_correct=len(pc0),
            broken_at_end=broken_end, lost_at_any_commit=sorted(lost_any),
            d_before=class_distance(w0, members), d_after=class_distance(L.w, members),
            failing=loss(L.w) > 0 or bool(broken_end),
            counterfactual_b0=cf, decisions=decisions, **diag))
        states.append(h1)
    return L, states, experiences, lock_ok


def stream_end(tables, n, order, h, members, ix_members):
    w = vector(h)
    cum = lambda u: sum(tables.loss[(n, qi)][hypothesis_index(u)] for qi in order)
    c0 = cum(w)
    n1_improves = any(cum(u) < c0 for u in n1(w))
    n2_improves = n1_improves or any(cum(u) < c0 for u in n2(w))
    return dict(final=h, d=class_distance(w, members), in_teacher_class=h in ix_members, cumulative_loss=c0,
                N1_local_minimum=not n1_improves, N_le2_local_minimum=not n2_improves)


def timelines(tables, n, order, states, ix, th):
    """Per-item own-map (IB) and B1 timelines after each experience, as local-learning-supervision v1."""
    out = []
    for t, qi in enumerate(order):
        a = ix.key_index[('B1', qi)]
        b1 = lambda h: ix.table[h][a] == ix.table[th][a]
        out.append(dict(step=t + 1, query=qi,
                        own=dict(before=tables.ok(n, qi, states[t]),
                                 after=''.join('1' if tables.ok(n, qi, h) else '0' for h in states[t + 1:])),
                        b1=dict(before=b1(states[t]), after=''.join('1' if b1(h) else '0' for h in states[t + 1:]))))
    return out


def attribution(tl, experiences):
    """Never-acquired and forgotten items with the classes of their presentation and loss experiences."""
    out = []
    for item in tl:
        after = item['own']['after']
        t_s = item['step']
        pres = experiences[t_s - 1]
        rec = dict(query=item['query'], step=t_s, presentation_class=pres['decisions'][0]['cls'],
                   presentation_counterfactual_b0=pres['counterfactual_b0']['cls'])
        if '1' not in after:
            rec.update(status='never_acquired', losses=None)
        else:
            losses = [t_s + i for i in range(1, len(after)) if after[i - 1] == '1' and after[i] == '0']
            rec['losses'] = [dict(step=s, classes=[d['cls'] for d in experiences[s - 1]['decisions']],
                                  counterfactual_b0=experiences[s - 1]['counterfactual_b0']['cls']) for s in losses]
            rec['status'] = 'forgotten' if after[-1] == '0' else 'recovered' if losses else 'held'
            rec['last_loss'] = rec['losses'][-1] if rec['status'] == 'forgotten' else None
        out.append(rec)
    return out


# ---------------------------------------------------------------- summaries

def decision_summary(experiences, key='counterfactual_b0'):
    """Class counts among all experiences and among failing ones (one label per experience)."""
    c, cf = Counter(), Counter()
    kinds, minb = Counter(), Counter()
    e0_outside = 0
    for e in experiences:
        lab = e[key] if key == 'counterfactual_b0' else e['decisions'][0]
        c[lab['cls']] += 1
        if e['failing']:
            cf[lab['cls']] += 1
        if lab['cls'] == 'E2b':
            kinds[lab['e2b_kind']] += 1
        if lab['cls'] == 'E1b':
            minb[str(lab['min_blocks_to_lower'])] += 1
        if lab['cls'] == 'E0' and not e['in_teacher_class']:
            e0_outside += 1
    nf = sum(e['failing'] for e in experiences)
    return dict(experiences=len(experiences), failing=nf,
                counts={k: c[k] for k in CLASSES},
                failing_counts={k: cf[k] for k in CLASSES},
                fraction_of_all={k: ratio(c[k], len(experiences)) for k in CLASSES},
                fraction_of_failing={k: ratio(cf[k], nf) for k in CLASSES},
                e2b_kind=dict(kinds), e1b_min_blocks=dict(minb), e0_outside_teacher_class=e0_outside)


def all_decisions_summary(experiences):
    c = Counter()
    for e in experiences:
        for d in e['decisions']:
            c[d['cls']] += 1
    return dict(decisions=sum(c.values()), counts={k: c[k] for k in CLASSES})


def proximity_summary(experiences):
    c = Counter()
    for e in experiences:
        if e['in_teacher_class']:
            continue
        c['states'] += 1
        dd = e['d_after'] - e['d_before']
        c['actual_d_' + ('decrease' if dd < 0 else 'same' if dd == 0 else 'increase')] += 1
        for nb in ('N1', 'N2'):
            p = e['proximity'][nb]
            if p['edits']:
                c[f'{nb}_states_with_teacherward_edit'] += 1
                c[f'{nb}_best_{p["best"]}'] += 1
                for k, v in p['counts'].items():
                    c[f'{nb}_edits_{k}'] += v
    return dict(c)


def graph_summary(experiences):
    c = Counter()
    for e in experiences:
        if e['in_teacher_class']:
            continue
        cat = e['graph']['category']
        c[f'category_{cat}'] += 1
        if e['failing']:
            c[f'failing_category_{cat}'] += 1
        if cat == 1:
            c['category_1_repetition_reaches_class'] += e['repetition']['terminal_in_class']
            c['category_1_repetition_misses_class'] += not e['repetition']['terminal_in_class']
    return dict(c)


def repetition_summary(experiences):
    c = Counter()
    extra = Counter()
    for e in experiences:
        r = e['repetition']
        b0_commits = int(e['counterfactual_b0']['cls'].startswith('E2'))
        x = r['commits'] - b0_commits
        extra[str(x)] += 1
        c['probes'] += 1
        c['decisions'] += r['decisions']
        c['candidate_losses'] += r['candidate_losses']
        c['terminal_exact'] += r['terminal_exact']
        c['terminal_in_class'] += r['terminal_in_class']
        c['terminal_exact_no_prior_wrong_at_end'] += r['terminal_exact'] and not r['wrong_at_end']
        c['lost_at_any_commit'] += bool(r['lost_at_any_commit'])
        c['lost_and_restored_within'] += bool(set(r['lost_at_any_commit']) - set(r['wrong_at_end']))
        if e['failing']:
            c['failing_probes'] += 1
            c['failing_terminal_exact_no_prior_wrong_at_end'] += r['terminal_exact'] and not r['wrong_at_end']
    return dict(c, extra_commits_histogram=dict(sorted(extra.items(), key=lambda kv: int(kv[0]))))


def end_summary(ends):
    c = Counter()
    for e in ends:
        c['streams'] += 1
        c['in_teacher_class'] += e['in_teacher_class']
        if e['cumulative_loss'] > 0:
            c['positive_loss'] += 1
            if e['N_le2_local_minimum']:
                c['positive_loss_N_le2_local_minimum'] += 1
            elif e['N1_local_minimum']:
                c['positive_loss_N1_but_not_N_le2_local_minimum'] += 1
            else:
                c['positive_loss_not_local_minimum'] += 1
        c[f'd_{e["d"]}'] += 1
    return dict(c)


def attribution_summary(attrs):
    c = Counter()
    for a in attrs:
        c[f'status_{a["status"]}'] += 1
        if a['status'] == 'never_acquired':
            c[f'never_acquired_presented_{a["presentation_counterfactual_b0"]}'] += 1
        for loss in a['losses'] or []:
            c[f'loss_caused_by_{loss["counterfactual_b0"]}'] += 1
        if a['status'] == 'forgotten':
            c[f'forgotten_last_loss_{a["last_loss"]["counterfactual_b0"]}'] += 1
    return dict(sorted(c.items()))


def summaries(streams):
    out = {}
    for n in TEACHERS:
        S = [s for s in streams if s['teacher'] == n]
        ex = [e for s in S for e in s['experiences']]
        attrs = [a for s in S for a in s['attribution']]
        out[n] = dict(counterfactual_b0=decision_summary(ex), actual_first_decision=decision_summary(ex, 'actual'),
                      all_decisions=all_decisions_summary(ex), proximity=proximity_summary(ex),
                      graphs=graph_summary(ex), repetition=repetition_summary(ex),
                      end_states=end_summary([s['end'] for s in S]), attribution=attribution_summary(attrs))
    return out


# ---------------------------------------------------------------- targets

def targets(rows, ix, bat, streams, split_ids, gate_cache, variant):
    pooled = RowIndex(rows)
    out = {}
    for n in TEACHERS:
        th = bat['teacher_idx'][n]
        mir = hypothesis_index(mirror(vector(th)))
        S = [s for s in streams if s['teacher'] == n]
        parts = {}
        for part in ('within', 'mixed24'):
            lc, z = pooled(n, variant, 'local', 15, part), pooled(n, variant, 'zero', 15, part)
            per = {sid: pooled(n, variant, 'local', 15, part, splits=[sid]) for sid in split_ids}
            parts[part] = dict(exact=lc['exact'], n=lc['n'], accuracy=ratio(lc['exact'], lc['n']), zero_exact=z['exact'],
                               worst_split=min(ratio(p['exact'], p['n']) for p in per.values()))
        l2 = all(p['n'] > 0 and p['exact'] >= 0.8 * p['n'] and p['exact'] > p['zero_exact'] for p in parts.values())
        beh = [dict(split=s['split'], order=s['order'], **behaviour(ix, n, th, s['final_index'], vector(s['final_index']), gate_cache),
                    converged_to='teacher' if s['final_index'] == th else 'mirror' if s['final_index'] == mir else 'other')
               for s in S]
        rec = sum(b['recovered'] for b in beh)
        per_split = {sid: sum(b['recovered'] for b in beh if b['split'] == sid) for sid in split_ids}
        views = {}
        for view in ('own', 'b1'):
            pool = pool_timelines([x for s in S for x in timeline_summaries(s, view)])
            pool['per_split_final_retention'] = {sid: pool_timelines(
                [x for s in S if s['split'] == sid for x in timeline_summaries(s, view)])['final_retention'] for sid in split_ids}
            mins = [min_prefix_accuracy(s, view) for s in S]
            pool['min_prefix_seen_accuracy'] = dict(min=min(mins), per_stream=mins)
            views[view] = pool
        own = views['own']
        l3 = own['presented'] > 0 and own['ever_acquired'] > 0 and own['final_correct'] >= 0.95 * own['presented'] \
            and own['forgotten'] <= 0.05 * own['ever_acquired']
        curve = []
        for i, k in enumerate(CHECKPOINTS):
            cs = [s['cost'][i] for s in S]
            curve.append(dict(checkpoint=k, recovered=sum(behaviour(ix, n, th, x['local_index'], vector(x['local_index']), gate_cache)['recovered'] for x in cs),
                              **{f'mean_{m}': round(sum(x[m] for x in cs) / len(cs), 6) for m in
                                 ('candidate_losses', 'predictions', 'history_predictions', 'decisions', 'commits')}))
        lf = sum(bool(ev_fwd(ix, s['final_index'])) for s in S)
        t = dict(L1_recovery=dict(recovered=rec, per_split=per_split, worst_split=min(per_split.values()),
                                  convergence=dict(Counter(b['converged_to'] for b in beh)),
                                  distinct_final_vectors=len({s['final_index'] for s in S}), runs_detail=beh,
                                  passed=rec >= 23 and min(per_split.values()) >= 4),
                 L2_outputs=dict(partitions=parts, passed=l2),
                 L3_preservation=dict(own_map=own, b1_only=views['b1'], passed=l3),
                 recovery_by_cost=curve, local_forward_witness_runs=dict(runs=lf, of=len(S)))
        t['suffices'] = t['L1_recovery']['passed'] and l2 and l3
        if n not in POSITIVE:
            t['target_applies'] = False
            t['note'] = 'L1-L3 are declared for S1 and C2 only; reported, not scored'
        out[n] = t
    out['suffices_for_S1_and_C2'] = out['S1']['suffices'] and out['C2']['suffices']
    return out


# ---------------------------------------------------------------- main

def run_tests():
    names = ['tests.test_update_diagnosis', 'tests.test_local_supervision', 'tests.test_mechanism_learning']
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    with open('/dev/null', 'w') as sink:
        res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return dict(modules=names, run=res.testsRun, failures=len(res.failures), errors=len(res.errors),
                passed=res.wasSuccessful())


def main(argv=None):
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'reports'))
    ap.add_argument('--manifest-only', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = freeze_manifest(out)
    if args.manifest_only:
        print(m['sha256'], m['contract_sha256'])
        return
    splits = json.loads((ROOT / SPLITS).read_text())['splits']
    timing = {}
    t0 = time.perf_counter()
    bat = build_battery(args.workers)
    timing['battery'] = time.perf_counter() - t0
    v1s = json.loads((ROOT / f'reports/{V1}-battery-summary.json').read_text())
    hashes = [atom_hash(v) for v in bat['table']]
    gate = dict(battery=hashlib.sha256(''.join(hashes).encode()).hexdigest() == v1s['battery_sha256']
                and bat['schema_hash'] == v1s['atom_schema_sha256'] and hashes == v1s['hypothesis_hashes'])
    gate['prior_artifacts'] = all(sha(p) == h for p, h in m['prior_artifacts'].items())
    if not (gate['battery'] and gate['prior_artifacts']):
        sys.exit(f'battery or prior artifacts differ: {gate}')
    t0 = time.perf_counter()
    ix = Index(bat)
    pj = Projected(ix)
    lls_ceil = json.loads((ROOT / f'reports/{LLS}-ceiling.json').read_text())['ceilings']
    members = {}
    for n, th in bat['teacher_idx'].items():
        c = lls_ceil[n]['IB']
        mir = hypothesis_index(mirror(vector(th)))
        members[n] = [vector(th), vector(mir)]
        gate.setdefault('teacher_class_matches_frozen_IB', True)
        gate['teacher_class_matches_frozen_IB'] &= (c['class_size'] == 2 and c['mechanism_homogeneous']
                                                    and c['core_sign_mirror']['index'] == mir
                                                    and c['core_sign_mirror']['in_class'])
    queries = sorted({q for s in splits for q in s['train']})
    tables = Tables(ix, pj, bat, queries)
    adj = adjacency()
    seen = {0}
    stack = [0]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    gate['N1_graph_connected'] = len(seen) == N_VECTORS
    gate['N1_N2_enumeration'] = all(
        len(set(n1(vector(h)))) == 26 and len(set(n2(vector(h)))) == 240 and not set(n1(vector(h))) & set(n2(vector(h)))
        for h in (0, 1, 2, 1093, 2186, 728))
    zero_ok = True
    for n, th in bat['teacher_idx'].items():
        for h in (hypothesis_index(w) for w in members[n]):
            zero_ok &= all(tables.loss[(n, qi)][h] == 0 for qi in queries)
    gate['teacher_class_zero_loss'] = zero_ok
    gr, certs = graphs(tables, adj, bat, [(n, qi) for n in TEACHERS for qi in queries])
    gate['G_down_subset_G_le'] = all(c['down_subset_le'] for c in certs.values())
    timing['tables_and_graphs'] = time.perf_counter() - t0

    lls = json.loads((ROOT / f'reports/{LLS}-learning.json').read_text())
    lls_streams = {(s['teacher'], s['split'], s['order']): s for s in lls['streams'] if s['condition'] == 'IB'}
    lls_rows = {tuple(r[:7]): r[7] for r in lls['rows'] if r[1] == 'IB' and r[6] in ('local', 'zero', 'teacher')}
    lls_work = lls['work']['IB_local']
    del lls
    ev = Evaluator(ix, bat)
    gate_cache = {}
    split_ids = [s['seed'] for s in splits]
    results, rows_all = {}, []
    lock = Counter()
    for variant in VARIANTS:
        t0 = time.perf_counter()
        sim = make_simulator('IB')
        lock_sim = make_simulator('IB') if variant == 'B0' else None
        streams, rows = [], []
        work = Counter()
        for n in TEACHERS:
            th = bat['teacher_idx'][n]
            ix_members = {hypothesis_index(w) for w in members[n]}
            for split in splits:
                parts = dict(within=split['within_context'],
                             mixed24=m['streams'][str(split['seed'])]['mixed24'], mixed36=split['mixed_evidence'])
                for o, order in enumerate(split['orders']):
                    req0 = sim.requests
                    L, states, exps, lock_ok = run_variant_stream(variant, n, th, order, sim, tables, gr, members[n],
                                                                  lockstep_sim=lock_sim)
                    if variant == 'B0':
                        lock['streams'] += 1
                        lock['lockstep_mismatch'] += not lock_ok
                    tl = timelines(tables, n, order, states, ix, th)
                    cost = []
                    for k in CHECKPOINTS:
                        h = states[k]
                        cost.append(dict(checkpoint=k, local_index=h))
                        for part, Q in [('seen', order[:k])] + list(parts.items()):
                            base = [n, variant, split['seed'], o, k, part]
                            rows.append(base + ['local', ev.single(n, h, Q)])
                            rows.append(base + ['zero', ev.single(n, hypothesis_index((0,) * 7), Q)])
                            rows.append(base + ['teacher', ev.single(n, th, Q)])
                    # cumulative cost at checkpoints from per-experience decisions
                    per_exp = [dict(decisions=len(e['decisions']), commits=e['commits']) for e in exps]
                    cand = {'B0': 27, 'REPEAT': 27, 'REPLAY': 27, 'SCOPE': 267}[variant]
                    keys = [len(observation_keys(QUERIES[q], 'IB')) for q in order]
                    for c in cost:
                        k = c['checkpoint']
                        dec = sum(p['decisions'] for p in per_exp[:k])
                        c['decisions'] = dec
                        c['commits'] = sum(p['commits'] for p in per_exp[:k])
                        c['candidate_losses'] = sum(cand * per_exp[t]['decisions'] for t in range(k))
                        c['predictions'] = sum(cand * per_exp[t]['decisions'] * keys[t] for t in range(k))
                        c['history_predictions'] = (sum(cand * sum(keys[:t]) for t in range(k))
                                                    if variant == 'REPLAY' else 0)
                    if cost[-1]['predictions'] != L.evaluations or cost[-1]['history_predictions'] != L.history_predictions \
                            or cost[-1]['candidate_losses'] != L.candidate_losses:
                        lock['cost_accounting_mismatch'] += 1
                    for kname in ('evaluations', 'history_predictions', 'candidate_losses', 'decisions', 'commits',
                                  'coefficients_changed'):
                        work[kname] += getattr(L, kname)
                    work['stored_maps'] += len(L.history)
                    work['simulation_requests'] += sim.requests - req0
                    end = stream_end(tables, n, order, states[-1], members[n], ix_members)
                    s = dict(teacher=n, variant=variant, split=split['seed'], order=o, queries=order, states=states,
                             final_index=states[-1], final_w=list(vector(states[-1])), experiences=exps,
                             timelines=tl, cost=cost, end=end)
                    s['attribution'] = attribution(tl, exps)
                    if variant == 'B0':
                        ref = lls_streams[(n, split['seed'], o)]
                        lock['frozen_stream_checks'] += 1
                        lock['frozen_timeline_mismatch'] += ref['timelines'] != tl
                        lock['frozen_checkpoint_mismatch'] += [r['local_index'] for r in ref['cost']] != [states[k] for k in CHECKPOINTS]
                        lock['frozen_final_mismatch'] += ref['final_index'] != states[-1]
                    streams.append(s)
        work['simulation_unique'] = len(sim.memo)
        if variant == 'B0':
            mine = {(r[0], 'IB') + tuple(r[2:7]): r[7] for r in rows}
            lock['frozen_row_mismatch'] = sum(lls_rows.get(k) != v for k, v in mine.items()) + len(set(lls_rows) - set(mine))
            lock['frozen_work_mismatch'] = int(any(work[k] != lls_work[k] for k in ('evaluations', 'commits', 'coefficients_changed'))
                                               or work['candidate_losses'] != lls_work['candidate_losses'])
        # learner-facing service losses equal battery-derived losses for every evaluated candidate
        qidx = {x: i for i, x in enumerate(QUERIES)}
        bad = 0
        for (w, x, key), y in sim.memo.items():
            qi = qidx[x]
            h = hypothesis_index(w)
            a = atom_of(ix, qi, key)
            if key[0] == 'B1':
                enc = sum(1 << (3 * (p + 1) + (q + 1)) for p, q in y)
                bad += enc != ix.table[h][a]
            else:
                bad += sum(1 << (v + 1) for v in y) != pj.value(a, h)
        lock[f'{variant}_service_checks'] = len(sim.memo)
        lock[f'{variant}_service_mismatch'] = bad
        rows_all += rows
        timing[f'variant_{variant}'] = time.perf_counter() - t0
        results[variant] = dict(streams=streams, work=dict(work))
    t0 = time.perf_counter()
    tg = {v: targets(rows_all, ix, bat, results[v]['streams'], split_ids, gate_cache, v) for v in VARIANTS}
    sm = {v: summaries(results[v]['streams']) for v in VARIANTS}
    timing['targets_and_summaries'] = time.perf_counter() - t0
    # gates
    checks = Counter()
    for v in VARIANTS:
        for s in results[v]['streams']:
            for e in s['experiences']:
                for d in e['decisions']:
                    checks['decision_labels'] += 1
                    checks['label_outside_classes'] += d['cls'] not in CLASSES
                    checks['scope_E1a'] += v == 'SCOPE' and d['cls'] == 'E1a'
                if v == 'REPEAT':
                    checks['repeat_final_check_missing'] += e['decisions'][-1]['committed']
                if v == 'REPLAY':
                    checks['replay_eligibility_mismatch'] += (e['commits'] > 0) != (e['counterfactual_b0']['b0_choice'] is not None)
                if not e['in_teacher_class']:
                    checks['graph_states'] += 1
    gate.update(lockstep_and_frozen=all(lock[k] == 0 for k in lock if 'mismatch' in k),
                labels_partition=checks['label_outside_classes'] == 0, scope_no_E1a=checks['scope_E1a'] == 0,
                repeat_terminates_with_check=checks['repeat_final_check_missing'] == 0,
                replay_eligibility_matches_B0=checks['replay_eligibility_mismatch'] == 0)
    gate['passed'] = all(gate.values())
    t0 = time.perf_counter()
    tests = run_tests()
    timing['tests'] = time.perf_counter() - t0
    completion = completion_status(gate, tests)
    prov = dict(python=platform.python_version(), implementation=platform.python_implementation(),
                source_sha256={p: sha(p) for p in ('src/update_diagnosis.py', 'experiments/update_diagnosis.py',
                                                   'tests/test_update_diagnosis.py', CONTRACT)})
    b0 = results['B0']
    dump(out / f'{PREFIX}-diagnosis.json', dict(
        experiment=PREFIX, phase='A', manifest_sha256=m['sha256'], completion=completion, correctness_gate=gate,
        reproduction=dict(lock), checks=dict(checks), tests=tests, summaries=sm['B0'], graph_certificates=certs,
        streams=b0['streams'], work=b0['work'], provenance=prov))
    head = dict(experiment=PREFIX, phase='B', manifest_sha256=m['sha256'], completion=completion, correctness_gate=gate,
                targets=tg, summaries={v: sm[v] for v in VARIANTS if v != 'B0'},
                work={v: results[v]['work'] for v in VARIANTS},
                streams=[s for v in VARIANTS if v != 'B0' for s in results[v]['streams']],
                provenance=prov, row_columns=['teacher', 'variant', 'split', 'order', 'checkpoint', 'partition',
                                              'learner', 'counts'])
    write_rows_report(out / f'{PREFIX}-variants.json', head, rows_all)
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds={k: round(v, 3) for k, v in timing.items()},
        total_including_preprocessing=round(time.perf_counter() - started, 3), workers=args.workers,
        peak_rss_kib=dict(main=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          largest_child=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                          method='getrusage ru_maxrss: main process peak, and the largest single terminated '
                                 'child; not the sum over concurrent workers'),
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        out_dir=str(out), command=' '.join(sys.argv), git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain'))))
    print(completion, json.dumps(gate))


if __name__ == '__main__':
    main()
