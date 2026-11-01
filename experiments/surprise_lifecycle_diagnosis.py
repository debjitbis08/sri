"""Surprise lifecycle v1, post hoc (docs/surprise-lifecycle-v1-pilot.md): where does a rule get lost, and does group size wrongly protect traces?

Pilot seeds only: nothing here is evidence for any contrast. The evaluator's labels are used to read the mechanism, and never reach it. Stages for
each new rule: traces kept -> traces together in a group -> a coherent fit proposed -> the shadow predicts unseen items -> validation accepts -> the
update survives later interference. Every promotion and slot write is also classed as useful or harmful by what it did and what fed it.
"""
import json
import multiprocessing as mp
import sys
import time

import numpy as np

sys.argv = [sys.argv[0], 'diagnosis'] + sys.argv[1:]
from experiments import surprise_lifecycle as E
from src import surprise_lifecycle as sl

SEEDS = E.PILOT_SEEDS
GAPS = (128, 512, 2048) if not E.SMOKE else (512,)
NAMES = {v: k for k, v in sl.K.items()}
USEFUL_SLOT = ('binding', 'exception', 'rare')
START = {n: s for n, _, s in E.LIFE_GRID} | {'rho': 8, 'tau_g': .4}


def rule_accuracy(R, cues_by_rule, protos, ids):
    return [float((ids[E.decode(C @ R, protos)] == o).mean()) for C, o in cues_by_rule]


def old_rule_accuracy(R, known, protos, ids):
    return float(np.mean([(ids[E.decode(C @ R, protos)] == o).mean() for C, o in known]))


def trace(task):
    label, cfg, kw, seed, g = task
    P = E.pool(seed)
    t = E.tables(seed)
    keys = P['stream_degraded']
    old = E.old_knowledge(seed, t, keys)
    w = E.world(seed, t, g, .2)
    protos, ids = E.prototypes(P['audit_degraded'])
    sets = E.audit_sets(seed, t, P['audit_clean'], E.R_AUDIT)
    noise = E.noise_cues(seed, w, P['audit_clean'], E.R_AUDIT)
    a = sl.Life(old['R'], old['slot_c'], old['slot_o'], cfg, 64, name=label, seed_rng=E.nc.stream('sl_sketch', seed), **kw)
    rule_ids = {w['cause'][i]: None for i in np.flatnonzero(w['kind'] == sl.K['rule'])}
    rule_index = {c: j for j, c in enumerate(sorted(rule_ids))}
    n_rules = len(rule_index)
    who = {}
    cur = {'kind': None, 'cause': None}

    stage = {'events': [0] * n_rules, 'kept': [0] * n_rules, 'evicted_on_arrival': [0] * n_rules, 'max_live': [0] * n_rules, 'max_together': [0] * n_rules, 'best_purity_when_4': [0.] * n_rules,
             'fits_majority': [0] * n_rules, 'fits_pure': [0] * n_rules, 'best_shadow_accuracy': [0.] * n_rules, 'wins': [0] * n_rules, 'failures': [0] * n_rules, 'promoted': [0] * n_rules, 'accuracy_at_promotion': [None] * n_rules}
    fail_kinds = {}
    fit_kinds = {}
    promotions, slots = [], {}
    shadow_info = {}
    evict_by_kind, kept_by_kind = {}, {}
    lifetime = {}

    def kinds_of(idx):
        return [who.get(int(a.serial[i]), ('?', -1)) for i in idx]

    orig_write = a._write_slot

    def write(cue, outcome):
        sims = a.tc[a.alive] @ cue
        idx = np.flatnonzero(a.alive)[sims >= a.tau]
        ks = [k for k, _ in kinds_of(idx)] or ['?']
        kind = max(set(ks), key=ks.count)
        slots[kind] = slots.get(kind, 0) + 1
        return orig_write(cue, outcome)
    a._write_slot = write

    orig_promote = a._promote

    def promote(s):
        ks = kinds_of(s.fit)
        rules = [c for k, c in ks if k == 'rule']
        before_old = old_rule_accuracy(a.R, sets['known_rule'], protos, ids)
        before_rule = rule_accuracy(a.R, sets['rule'], protos, ids)
        out = orig_promote(s)
        after_old = old_rule_accuracy(a.R, sets['known_rule'], protos, ids)
        after_rule = rule_accuracy(a.R, sets['rule'], protos, ids)
        major = max(set(rules), key=rules.count) if rules else None
        row = {'t': a.t, 'fit_kinds': sorted(k for k, _ in ks), 'old_rule_change': after_old - before_old, 'rule_gain_best': max(x - y for x, y in zip(after_rule, before_rule)), 'wins': s.wins, 'failures': s.failures}
        promotions.append(row)
        if major is not None and rules.count(major) * 2 > len(ks):
            j = rule_index[major]
            stage['promoted'][j] += 1
            stage['accuracy_at_promotion'][j] = after_rule[j]
        return out
    a._promote = promote

    orig_drop = a._drop_trace

    def drop(i):
        ser = int(a.serial[i])
        k = who.get(ser, ('?', -1))[0]
        if a.age[i] == a.t and a.alive[i]:
            evict_by_kind[k] = evict_by_kind.get(k, 0) + 1
            if k == 'rule':
                stage['evicted_on_arrival'][rule_index[who[ser][1]]] += 1
        if k == 'rule':
            lifetime.setdefault('rule', []).append(a.t - int(a.age[i]))
        elif k == 'noise-once':
            lifetime.setdefault('noise-once', []).append(a.t - int(a.age[i]))
        return orig_drop(i)
    a._drop_trace = drop

    pure_log, pure_live = [], {}
    orig_retire = a._retire

    def retire(s, reason='retired'):
        if s.id in pure_live:
            info = pure_live.pop(s.id)
            pure_log.append({'exit': 'displaced' if reason == 'displaced' else 'damage' if s.failures < cfg['h_f'] else 'failures', 'lived': a.t - s.born, 'wins': s.wins, 'failures': s.failures, 'damage': s.damage, 'rhits': s.rhits, 'accuracy': info['accuracy'], 'eligible': info['eligible'], 'seen': info['seen']})
        return orig_retire(s, reason)
    a._retire = retire

    orig_promote2 = a._promote

    def promote2(s):
        if s.id in pure_live:
            info = pure_live.pop(s.id)
            pure_log.append({'exit': 'promoted', 'lived': a.t - s.born, 'wins': s.wins, 'failures': s.failures, 'damage': s.damage, 'rhits': s.rhits, 'accuracy': info['accuracy'], 'eligible': info['eligible'], 'seen': info['seen']})
        return orig_promote2(s)
    a._promote = promote2

    composition, known_sid = [], set()
    marks = set(np.linspace(0, w['L'] - 1, 21)[1:].astype(int))
    for n, (c, o, k, cause) in enumerate(E.events(w, keys)):
        before = {s.id: s.failures for s in a.shadows}
        wins_before = {s.id: s.wins for s in a.shadows}
        ser = a.next_serial
        kname = NAMES[k]
        who[ser] = (kname, cause)       # before the step, since a trace can be evicted within it
        a.step(c, o)
        if a.next_serial > ser:
            kept_by_kind[kname] = kept_by_kind.get(kname, 0) + 1
        if kname == 'rule':
            j = rule_index[cause]
            stage['events'][j] += 1
            stage['kept'][j] += int(a.next_serial > ser)
        for s in a.shadows:
            if s.failures > before.get(s.id, s.failures):
                fail_kinds[kname] = fail_kinds.get(kname, 0) + 1
            if s.id not in known_sid:
                known_sid.add(s.id)
                ks = kinds_of(s.fit)
                rules = [cc for kk, cc in ks if kk == 'rule']
                for kk, _ in ks:
                    fit_kinds[kk] = fit_kinds.get(kk, 0) + 1
                major = max(set(rules), key=rules.count) if rules else None
                if major is not None:
                    j = rule_index[major]
                    if rules.count(major) * 2 > len(ks):
                        stage['fits_majority'][j] += 1
                    if rules.count(major) == len(ks):
                        stage['fits_pure'][j] += 1
                        C, oo = sets['rule'][j]
                        p = C @ a.R + (C @ s.chat / s.norm2)[:, None] * s.ebar[None]
                        pure_live[s.id] = {'accuracy': float((ids[E.decode(p, protos)] == oo).mean()), 'eligible': {}, 'seen': {}}
                # the shadow's own prediction on this rule's unseen items, measured for every shadow against every rule
                for j2, (C, oo) in enumerate(sets['rule']):
                    p = C @ a.R + (C @ s.chat / s.norm2)[:, None] * s.ebar[None]
                    acc = float((ids[E.decode(p, protos)] == oo).mean())
                    stage['best_shadow_accuracy'][j2] = max(stage['best_shadow_accuracy'][j2], acc)
        for s in a.shadows:
            if s.id in pure_live:
                info = pure_live[s.id]
                group = 'rule' if kname == 'rule' else 'old' if kname.startswith('known') else 'other'
                info['seen'][group] = info['seen'].get(group, 0) + 1
                if float(c @ s.chat) / np.sqrt(s.norm2) >= cfg['tau_s']:
                    info['eligible'][group] = info['eligible'].get(group, 0) + 1
        if kname == 'rule':
            for s in a.shadows:
                ks = kinds_of(s.fit)
                rules = [cc for kk, cc in ks if kk == 'rule']
                if rules.count(cause) * 2 > len(ks):
                    stage['wins'][rule_index[cause]] += s.wins - wins_before.get(s.id, s.wins)
                    stage['failures'][rule_index[cause]] += s.failures - before.get(s.id, s.failures)
        if n in marks or kname == 'rule':
            live = np.flatnonzero(a.alive)
            serials = [who.get(int(a.serial[i]), ('?', -1)) for i in live]
            per = {}
            for i, (kk, cc) in zip(live, serials):
                if kk == 'rule':
                    per.setdefault(cc, []).append(int(a.res[i]))
            for cc, gs in per.items():
                j = rule_index[cc]
                stage['max_live'][j] = max(stage['max_live'][j], len(gs))
                top = max(set(gs), key=gs.count)
                n_top = gs.count(top)
                stage['max_together'][j] = max(stage['max_together'][j], n_top)
                if n_top >= 4 or n_top >= cfg['m']:
                    size = int((a.res[a.alive] == top).sum())
                    stage['best_purity_when_4'][j] = max(stage['best_purity_when_4'][j], n_top / size)
        if n in marks:
            live = np.flatnonzero(a.alive)
            kinds = [who.get(int(a.serial[i]), ('?', -1))[0] for i in live]
            sup = np.array([a._support(int(i)) for i in live]) if len(live) else np.zeros(0)
            sizes = np.bincount(a.res[live][a.res[live] >= 0]) if len(live) and (a.res[live] >= 0).any() else np.zeros(1, int)
            composition.append({'t': n, 'live': len(live), 'share_noise_once': kinds.count('noise-once') / max(len(live), 1), 'share_rule': kinds.count('rule') / max(len(live), 1), 'share_supported': float((sup > 0).mean()) if len(live) else 0., 'largest_residual_group': int(sizes.max()), 'slots': a.n_new})
    read = E.read_arm(a, sets, noise, protos, ids)
    useful = [p for p in promotions if p['rule_gain_best'] >= .10]
    harmful = [p for p in promotions if p['old_rule_change'] <= -.02]
    both = [p for p in promotions if p['rule_gain_best'] >= .10 and p['old_rule_change'] <= -.02]
    last = composition[-5:]
    return {'label': label, 'seed': seed, 'g': g, 'stage': stage, 'read': {k: read[k] for k in ('rule_generalization', 'binding_recall', 'rare_recall', 'noise_capture', 'old_knowledge', 'known_rule', 'known_binding', 'exception_family')},
            'promotions': {'n': len(promotions), 'useful': len(useful), 'harmful': len(harmful), 'both': len(both), 'neither': len(promotions) - len(useful) - len(harmful) + len(both), 'rows': promotions[:60]}, 'slots_by_kind': slots,
            'slots_useful': sum(slots.get(k, 0) for k in USEFUL_SLOT), 'slots_total': sum(slots.values()), 'kept_by_kind': kept_by_kind, 'evicted_on_arrival_by_kind': evict_by_kind, 'failures_by_event_kind': fail_kinds, 'fit_trace_kinds': fit_kinds,
            'pure_shadows': pure_log, 'lifetime_median': {k: float(np.median(v)) for k, v in lifetime.items()}, 'store_late': {k: float(np.mean([r[k] for r in last])) for k in last[0]} if last else None, 'log': a.log}


def oracle_old(seed, g):
    """Where does the oracle lose old knowledge? Its updates alone, with slots kept, on each old reading."""
    sel = json.load(open(f'{E.REPORT}-selection.json'))['selected']['life-oracle']
    res, _ = E.run('life-oracle', sel, seed, g, .2, 64, 'degraded')
    return {'seed': seed, 'g': g, **{k: res['end'][k] for k in ('known_rule', 'known_binding', 'exception_family', 'old_knowledge', 'rule_generalization', 'binding_recall')}}


def main():
    t0 = time.perf_counter()
    sel = json.load(open(f'{E.REPORT}-selection.json'))['selected']
    runs = [('life-selected', sel['life'], {}), ('life-start', START, {}), ('cuegroup-selected', sel['life-cuegroup'], {'group_by': 'cue'}), ('cuegroup-start', START, {'group_by': 'cue'}),
            ('life-validated-support', sel['life'], {'support': 'validated'}), ('life-start-validated-support', START, {'support': 'validated'}),
            ('cuegroup-start-validated-support', START, {'group_by': 'cue', 'support': 'validated'}), ('sketch-selected', sel['life-sketch'], {'sketch': True}),
            ('life-m2', {**START, 'm': 2}, {}), ('life-m2-validated-support', {**START, 'm': 2}, {'support': 'validated'}), ('cuegroup-m2-validated-support', {**START, 'm': 2}, {'group_by': 'cue', 'support': 'validated'})]
    tasks = [(a, c, kw, s, g) for a, c, kw in runs for s in SEEDS for g in GAPS]
    with mp.get_context('fork').Pool(E.WORKERS) as p:
        rows = p.map(trace, tasks)
        ora = p.starmap(oracle_old, [(s, g) for s in SEEDS for g in E.GAPS])
    E.save_json(f'{E.REPORT}-diagnosis.json', {'rows': rows, 'oracle': ora})
    E.save_json(f'{E.REPORT}-diagnosis-runtime.json', E.runtime({'wall_s': time.perf_counter() - t0, 'runs': len(tasks) + len(ora)}))
    print('label seed g | kept evict-on-arrival live together maj pure | bestshadow | wins fails promoted | rule bind old | prom n/useful/harm | slots useful/total | live noise-share supported largest | life rule/once | sketch')
    for r in rows:
        st = r['stage']
        sm = lambda k: sum(st[k])
        print(r['label'], r['seed'], r['g'], '|', sm('kept'), sm('evicted_on_arrival'), max(st['max_live']), max(st['max_together']), sm('fits_majority'), sm('fits_pure'), '|', round(max(st['best_shadow_accuracy']), 2), '|', sm('wins'), sm('failures'), sm('promoted'), '|',
              round(r['read']['rule_generalization'], 2), round(r['read']['binding_recall'], 2), round(r['read']['old_knowledge'], 2), '|', r['promotions']['n'], r['promotions']['useful'], r['promotions']['harmful'], '|', r['slots_useful'], r['slots_total'], '|',
              *(round(r['store_late'][k], 2) for k in ('live', 'share_noise_once', 'share_supported', 'largest_residual_group')), '|', r['lifetime_median'].get('rule'), r['lifetime_median'].get('noise-once'), '|', r['log']['sketch_matches'], r['kept_by_kind'].get('noise-once'))
    print('pure shadows: label seed g | n by exit | median lived, accuracy, wins, failures, damage/R-hits | eligible share of old, other, rule events')
    for r in rows:
        ps = r['pure_shadows']
        if not ps:
            continue
        ex = {}
        for x in ps:
            ex[x['exit']] = ex.get(x['exit'], 0) + 1
        share = {g2: round(sum(x['eligible'].get(g2, 0) for x in ps) / max(sum(x['seen'].get(g2, 0) for x in ps), 1), 2) for g2 in ('old', 'other', 'rule')}
        print(r['label'], r['seed'], r['g'], '|', ex, '|', int(np.median([x['lived'] for x in ps])), round(float(np.median([x['accuracy'] for x in ps])), 2), int(np.median([x['wins'] for x in ps])), int(np.median([x['failures'] for x in ps])), f"{int(np.median([x['damage'] for x in ps]))}/{int(np.median([x['rhits'] for x in ps]))}", '|', share)
    for o in ora:
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in o.items()})


if __name__ == '__main__':
    main()
