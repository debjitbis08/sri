"""Long-term memory v1 (docs/long-term-memory-v1-contract.md): a memory network combined with sparse48 in the forward pass, trained on the readers'
errors, unable to change the core. Host runs write a trajectory cache; memory variants replay from it offline.

Stages (argv[1]):
  encoders          frozen key encoders for the pilot seeds (GPU)
  hosts             sparse48 with the cache hook, sparse48 without it (gate 1, first pilot seed), and mixed, on the pilot seeds (GPU)
  gates             gates 1-5, from the cache; selection and confirmation refuse without a passing report tied to these inputs
  pilot             the sequential sweep on the pilot seeds, offline on CPU
  confirm-encoders  refuses before the contract is frozen: frozen key encoders for seeds 394-403, in their own process
  confirm-hosts     refuses before the contract is frozen: host runs for seeds 394-403
  confirm           refuses before the contract is frozen: candidate, six ablations and references
  analyse           the contract's decision rule

Usage: NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.long_term_memory hosts; .venv/bin/python -m experiments.long_term_memory pilot.
"""
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

from experiments import long_term_memory_host as H
from src import long_term_memory as ltm

X, nc, P, AP, DEV = H.X, H.nc, H.P, H.AP, H.DEV
SMOKE, STAGE, RUNS, REPORT, CONTRACT, PILOT_SEEDS, CONFIRM_SEEDS, N, BLOCKS, KEYS = H.SMOKE, H.STAGE, H.RUNS, H.REPORT, H.CONTRACT, H.PILOT_SEEDS, H.CONFIRM_SEEDS, H.N, H.BLOCKS, H.KEYS
save_json, save_npz, cache_dir = H.save_json, H.save_npz, H.cache_dir
WORKERS = int(os.environ.get('NSY_WORKERS', 11))
TRANSFER = ('chinese/withheld', 'devanagari_digits/withheld')
ELIGIBLE, NEAR = .03, .03
COUNT_IDENTITY_MIN = .80
STORE_CEILING = .80
START = dict(form='mlp', projections='learned', key='frozen', eta=.01, beta=.9, C=16, p=.05)
SETTINGS = [('eta', (.001, .003, .01, .03)), ('beta', (0., .9)), ('C', (1, 16, 64)), ('p', (.02, .05, .1))]


def runtime(extra):
    return H.runtime({'source_sha256': nc.sha256(__file__), 'mechanisms_sha256': nc.sha256(ltm.__file__), 'workers': WORKERS, **extra})


# ---------------------------------------------------------------- replay

_c = {}


def load_cache(seed, arm='sparse48', mode='full'):
    key = (seed, arm, mode)
    if key not in _c:
        d = cache_dir(arm, seed, mode)
        train = {b: dict(np.load(d / f'train-{b}.npz')) for b in BLOCKS} if mode == 'full' else None
        _c[key] = {'train': train, 'eval': {b: dict(np.load(d / f'eval-{b}.npz')) for b in BLOCKS}}
    return _c[key]


def eval_name(name, field):
    return f"{name.replace('/', '__')}___{field}"


def read_set(ev, name, cfg, mem):
    logits = torch.from_numpy(ev[eval_name(name, 'logits')])
    counts = torch.from_numpy(ev[eval_name(name, 'counts')]).long()
    truth = torch.from_numpy(ev[eval_name(name, 'truth')]).long()
    alone = ltm.accuracy(logits, counts, truth)
    if mem is None:
        return {'readers': alone, 'combined': alone, 'memory': None}
    q, c = torch.from_numpy(ev[eval_name(name, f"q_{cfg['key']}")]), torch.from_numpy(ev[eval_name(name, f"c_{cfg['value']}")])
    with torch.no_grad():
        term = mem.term(q, c)
    return {'readers': alone, 'combined': ltm.accuracy(logits + term, counts, truth), 'memory': ltm.accuracy(term, counts, truth),
            'changed': float(((logits + term).argmax(1) != logits.argmax(1)).float().mean())}


def match_names():
    return [f'match/{lang}/{g}' for lang in P.LANGS for g in ('paired', 'withheld')]


def readings(ev, cfg, mem):
    out = {t: read_set(ev, f'recall/{t}', cfg, mem) for t in ('A', 'B', 'unknown')}
    for name in match_names():
        if eval_name(name, 'logits') in ev:
            out[name] = read_set(ev, name, cfg, mem)
    out['answer_transfer'] = float(np.mean([out[f'match/{k}']['combined'] for k in TRANSFER]))
    out['answer_transfer_readers'] = float(np.mean([out[f'match/{k}']['readers'] for k in TRANSFER]))
    return out


def replay(cfg, seed):
    """Science only: timing is kept by the caller."""
    torch.set_num_threads(1)
    cache = load_cache(seed)
    d_k, d_v = KEYS[cfg['key']], KEYS[cfg['value']]
    mem = ltm.Memory(cfg['form'], d_k, d_v, cfg['projections'], cfg['eta'], cfg['beta'], cfg['C'], cfg['p'],
                     seed=int(nc.stream('ltm_memory', seed, cfg['form'], cfg['key'], cfg['value'], cfg['projections']).integers(1 << 30)))
    reads, losses, ticks = {}, {}, {}
    for b in BLOCKS:
        losses[str(b)] = ltm.train_block(mem, cache['train'][b], cfg['key'], cfg['value'])
        ticks[str(b)] = mem.t
        reads[str(b)] = readings(cache['eval'][b], cfg, mem)
    return {'reads': reads, 'losses': losses, 's': float(mem.s.detach()), 'ticks': ticks, 'floats': ltm.param_floats(cfg['form'], d_k, d_v)}


def replay_key(cfg, seed):
    return f"{hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]}-seed{seed}-N{N}"


def cache_digest(seed):
    info = json.load(open(H.info_path('sparse48', seed, 'full')))
    return hashlib.sha256(json.dumps({'files': info['files'], 'hashes': info['hashes'], 'provenance': info['provenance']}, sort_keys=True).encode()).hexdigest()


def replay_provenance(cfg, seed):
    return {'memory': nc.sha256(ltm.__file__), 'replay': nc.sha256(__file__), 'contract': nc.sha256(CONTRACT), 'cache': cache_digest(seed), 'cfg': cfg, 'seed': seed, 'torch': torch.__version__}


class Replay:
    """A finished replay is reused only if its code, contract and cache are unchanged. Timing is written apart from science."""

    def __call__(self, task):
        cfg, seed = task
        key = replay_key(cfg, seed)
        path = RUNS / 'replays' / f'{key}.json.gz'
        prov = replay_provenance(cfg, seed)
        if path.exists():
            try:
                rec = json.loads(gzip.decompress(path.read_bytes()))
                if rec['provenance'] == H.DUR.roundtrip(prov):
                    return key, rec['science'], False
            except (OSError, ValueError, EOFError, KeyError):
                pass
        t0 = time.process_time()
        science = H.DUR.roundtrip(replay(cfg, seed))
        H.DUR.atomic_write(path, gzip.compress(json.dumps({'provenance': prov, 'science': science}, sort_keys=True).encode()))
        save_json(RUNS / 'replay-timing' / f'{key}.json', {'cpu_s': time.process_time() - t0})
        return key, science, True


COST = {'executed': 0, 'reused': 0, 'cpu_s': 0.}


def run_replays(tasks):
    with mp.get_context('fork').Pool(min(WORKERS, max(len(tasks), 1))) as pool:
        out = {}
        for key, science, fresh in pool.imap_unordered(Replay(), tasks):
            out[key] = science
            COST['executed' if fresh else 'reused'] += 1
            if fresh:
                COST['cpu_s'] += json.load(open(RUNS / 'replay-timing' / f'{key}.json'))['cpu_s']
        return out


def flat_read(seed, b):
    """`flat-store`: each pair's frozen key and count, written once; A from block 4, B from block 5."""
    z = np.load(cache_dir('sparse48', seed, 'full') / 'flat.npz')
    ev = load_cache(seed)['eval'][b]
    keys, vals = [], []
    if b >= 4:
        keys.append(z['A_keys']); vals.append(z['A_counts'])
    if b >= 5:
        keys.append(z['B_keys']); vals.append(z['B_counts'])
    if not keys:
        return {'A': None, 'B': None}
    K, V = np.vstack(keys), np.concatenate(vals)
    out = {}
    for t in ('A', 'B'):
        q = ev[eval_name(f'recall/{t}', 'q_frozen')]
        out[t] = float((V[(q @ K.T).argmax(1)] == ev[eval_name(f'recall/{t}', 'truth')]).mean())
    return out


def mixed_reads(seed):
    """The `mixed` reference: its own host and readers, no memory term."""
    r = readings(load_cache(seed, 'mixed', 'light')['eval'][6], None, None)
    return {'A': r['A']['readers'], 'B': r['B']['readers'], 'answer_transfer': r['answer_transfer_readers']}


# ---------------------------------------------------------------- gates

def gate_inputs(seeds):
    return {'memory': nc.sha256(ltm.__file__), 'caches': {str(s): cache_digest(s) for s in seeds}}


def gate_seeds(seeds, name, plain_required):
    t0 = time.perf_counter()
    out = {}
    expected = H.D.STEPS * 3 + 3 * H.L
    for seed in seeds:
        missing = [t for t in (('sparse48', seed, 'full'), ('mixed', seed, 'light')) if not H.host_valid(*t)]
        if missing:
            raise SystemExit(f'host runs missing or stale: {missing}')
        info = json.load(open(H.info_path('sparse48', seed, 'full')))
        r = {'items_per_update': info['items'] / max(info['updates'], 1)}
        cached = sum(len(load_cache(seed)['train'][b]['n']) for b in BLOCKS)
        r['clock_covers_every_update'] = cached == expected
        pp = H.info_path('sparse48', seed, 'plain')
        if pp.exists():
            r['gate1_identical'] = json.load(open(pp))['hashes'] == info['hashes']
        ev = load_cache(seed)['eval'][6]
        name_a = 'recall/A'
        q, cc = ev[eval_name(name_a, 'q_frozen')], ev[eval_name(name_a, 'c_frozen')]
        counts, truth = ev[eval_name(name_a, 'counts')], ev[eval_name(name_a, 'truth')]
        flat, labels = cc.reshape(-1, cc.shape[-1]), counts.reshape(-1)
        half = len(flat) // 2
        protos = sl_unit(np.stack([flat[:half][labels[:half] == k].mean(0) for k in range(1, 10)]))
        r['gate2_count_identity'] = float(((flat[half:] @ protos.T).argmax(1) + 1 == labels[half:]).mean())
        E = AP.EVAL_RENDERS
        sym = np.repeat(np.arange(len(q) // E), E)
        pm = sl_unit(np.stack([q[sym == k].mean(0) for k in range(len(q) // E)]))
        r['gate2_symbol_identity_prototype_includes_query'] = float(((q @ pm.T).argmax(1) == sym).mean())
        mem = ltm.Memory('mlp', 32, 32, seed=seed)
        lg = torch.from_numpy(ev[eval_name(name_a, 'logits')])
        r['gate3_silent'] = bool((mem.combined(torch.from_numpy(q), torch.from_numpy(cc), lg) == lg).all())
        r['gate4_flat'] = {str(b): flat_read(seed, b) for b in (4, 5, 6)}
        ltm.train_block(mem, load_cache(seed)['train'][4], 'frozen', 'frozen', limit=50)
        h = mem.state_hash()
        readings(ev, {'key': 'frozen', 'value': 'frozen'}, mem)
        r['gate5_audit_unchanged'] = h == mem.state_hash()
        rng = nc.stream('ltm_shuffle', seed)
        r['gate5_shuffled'] = float((counts[np.arange(len(q)), np.argmax(ev[eval_name(name_a, 'logits')], 1)] == rng.permutation(truth)).mean())
        base = readings(ev, None, None)
        r['sparse48'] = {'A': base['A']['readers'], 'B': base['B']['readers'], 'answer_transfer': base['answer_transfer_readers'],
                         'by_block_A': {str(b): read_set(load_cache(seed)['eval'][b], 'recall/A', None, None)['readers'] for b in BLOCKS},
                         'by_block_B': {str(b): read_set(load_cache(seed)['eval'][b], 'recall/B', None, None)['readers'] for b in BLOCKS}}
        r['mixed'] = mixed_reads(seed)
        checks = [r['clock_covers_every_update'], r['gate3_silent'], r['gate5_audit_unchanged'], abs(r['gate5_shuffled'] - 1 / 9) <= .06, r['gate4_flat']['6']['A'] >= STORE_CEILING]
        if plain_required and seed == seeds[0]:
            checks.append(r.get('gate1_identical') is True)
        r['passed'] = bool(all(checks))
        out[str(seed)] = r
    count = float(np.mean([out[str(s)]['gate2_count_identity'] for s in seeds]))
    report = {'seeds': out, 'count_identity': count, 'frozen_value_source': 'frozen' if count >= COUNT_IDENTITY_MIN else 'slow', 'passed': all(x['passed'] for x in out.values()), 'inputs': gate_inputs(seeds)}
    save_json(f'{REPORT}-{name}.json', report)
    save_json(f'{REPORT}-{name}-runtime.json', runtime({'wall_s': time.perf_counter() - t0}))
    print(json.dumps({k: {x: y for x, y in v.items()} for k, v in out.items()}, indent=1, default=float))
    print('passed', report['passed'], 'frozen values from', report['frozen_value_source'])


def sl_unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def require_gates(seeds, name):
    """Selection and confirmation refuse without a passing report for exactly these inputs. Returns the report."""
    path = f'{REPORT}-{name}.json'
    if not os.path.exists(path):
        raise SystemExit(f'no gate report at {path}: run the gates first')
    rep = json.load(open(path))
    if not rep['passed']:
        raise SystemExit(f'gates failed in {path}: {[k for k, v in rep["seeds"].items() if not v["passed"]]}')
    if rep['inputs'] != gate_inputs(seeds):
        raise SystemExit(f'{path} was made from other inputs: rerun the gates')
    return rep


# ---------------------------------------------------------------- selection

def with_value(cfg, frozen_values):
    """A key source and its value source: `frozen` keys take the source gate 2 decided."""
    return {**cfg, 'value': frozen_values if cfg['key'] == 'frozen' else cfg['key']}


def score(cfg, seeds, results):
    rows = [results[replay_key(cfg, s)]['reads']['6'] for s in seeds]
    a = [r['A']['combined'] for r in rows]
    b = [r['B']['combined'] for r in rows]
    t = float(np.mean([r['answer_transfer'] for r in rows]))
    t0 = float(np.mean([readings(load_cache(s)['eval'][6], None, None)['answer_transfer_readers'] for s in seeds]))
    return {'retained_A': float(np.mean(a)), 'retained_B': float(np.mean(b)), 'answer_transfer': t, 'sparse48_transfer': t0, 'objective': float(np.mean(a + b)), 'eligible': t >= t0 - ELIGIBLE}


def better(a, b):
    """Eligible first, then the objective; if neither is eligible, the smaller shortfall."""
    if a['eligible'] != b['eligible']:
        return a['eligible']
    return a['objective'] > b['objective'] if a['eligible'] else a['answer_transfer'] > b['answer_transfer']


def best_of(cands, seeds, results):
    cur = None
    for c in cands:
        if cur is None or better(score(c, seeds, results), score(cur, seeds, results)):
            cur = c
    return cur


def pilot():
    t0 = time.perf_counter()
    seeds = PILOT_SEEDS
    gate = require_gates(seeds, 'gates')
    fv = gate['frozen_value_source']
    results, log = {}, []

    def sweep(name, cands):
        results.update(run_replays([(c, s) for c in cands for s in seeds]))
        best = best_of(cands, seeds, results)
        log.append({'stage': name, 'candidates': [{'cfg': c, 'score': score(c, seeds, results)} for c in cands], 'best': best})
        print(name, {str(c[name]): round(score(c, seeds, results)['objective'], 3) for c in cands}, flush=True)
        return best
    cur = with_value(START, fv)
    cur = sweep('form', [{**cur, 'form': f} for f in ltm.FORMS])
    cur = sweep('projections', [{**cur, 'projections': p} for p in ('learned', 'fixed')])
    cands = [with_value({**cur, 'key': k}, fv) for k in ('live', 'slow', 'frozen')]
    best = sweep('key', cands)
    pref = next((c for c in cands[:2] if score(c, seeds, results)['eligible'] and score(c, seeds, results)['objective'] >= score(best, seeds, results)['objective'] - NEAR), None)
    cur = pref or best
    log.append({'stage': 'key-preference', 'chosen': cur})
    for name, values in SETTINGS:
        if name == 'p' and cur['form'] != 'k-sparse':
            continue
        cur = sweep(name, [{**cur, name: v} for v in values])
    print('candidate', cur, score(cur, seeds, results))
    save_json(f'{REPORT}-selection.json', {'candidate': cur, 'score': score(cur, seeds, results), 'frozen_value_source': fv, 'log': log, 'final': {str(s): results[replay_key(cur, s)] for s in seeds},
                                           'sparse48': {str(s): {str(b): {k: read_set(load_cache(s)['eval'][b], f'recall/{k}', None, None)['readers'] for k in 'AB'} for b in BLOCKS} for s in seeds},
                                           'mixed': {str(s): mixed_reads(s) for s in seeds}, 'flat': {str(s): {str(b): flat_read(s, b) for b in (4, 5, 6)} for s in seeds}, 'inputs': gate['inputs']})
    save_json(f'{REPORT}-pilot-runtime.json', runtime({'wall_s': time.perf_counter() - t0, **COST}))


# ---------------------------------------------------------------- confirmation

def frozen():
    return 'Status: **frozen' in CONTRACT.read_text()


def refuse_unless_frozen():
    if not frozen():
        raise SystemExit('the contract is not frozen: seeds 394-403 stay unbuilt, and confirmation refuses to run')


def ablations(c, fv):
    out = [{**c, 'projections': 'fixed' if c['projections'] == 'learned' else 'learned'}]
    out += [with_value({**c, 'key': k}, fv) for k in KEYS if k != c['key']]
    out += [{**c, 'form': f} for f in ltm.FORMS if f != c['form']]
    out.append({**c, 'C': 16 if c['C'] == 1 else 1})
    return out


def confirm_encoders():
    refuse_unless_frozen()
    H.fit_encoders(CONFIRM_SEEDS)


def confirm_hosts():
    refuse_unless_frozen()
    H.run_hosts([('sparse48', s, 'full') for s in CONFIRM_SEEDS] + [('mixed', s, 'light') for s in CONFIRM_SEEDS])


def confirm_gates():
    refuse_unless_frozen()
    gate_seeds(CONFIRM_SEEDS, 'confirm-gates', False)


def confirm():
    refuse_unless_frozen()
    t0 = time.perf_counter()
    require_gates(PILOT_SEEDS, 'gates')
    gate = require_gates(CONFIRM_SEEDS, 'confirm-gates')
    sel = json.load(open(f'{REPORT}-selection.json'))
    if sel['frozen_value_source'] != gate['frozen_value_source']:
        print('gate 2 decided differently on the confirmation seeds; the pilot\'s decision stands, and this is reported', flush=True)
    c, fv = sel['candidate'], sel['frozen_value_source']
    results = run_replays([(x, s) for x in [c] + ablations(c, fv) for s in CONFIRM_SEEDS])
    H.DUR.atomic_write(f'{REPORT}-confirm.json.gz', gzip.compress(json.dumps(results, sort_keys=True, default=float).encode()))
    save_json(f'{REPORT}-confirm-runtime.json', runtime({'wall_s': time.perf_counter() - t0, **COST}))


def boot(d, rng):
    d = np.asarray(d)
    m = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def analyse():
    res = json.loads(gzip.decompress(open(f'{REPORT}-confirm.json.gz', 'rb').read()))
    sel = json.load(open(f'{REPORT}-selection.json'))
    c, fv = sel['candidate'], sel['frozen_value_source']
    rng = nc.stream('ltm_bootstrap')

    def r(cfg, s, field):
        x = res[replay_key(cfg, s)]['reads']['6']
        return x['answer_transfer'] if field == 'answer_transfer' else x[field]['combined']
    seeds = CONFIRM_SEEDS
    base = {s: readings(load_cache(s)['eval'][6], None, None) for s in seeds}
    gA = boot([r(c, s, 'A') - base[s]['A']['readers'] for s in seeds], rng)
    gB = boot([r(c, s, 'B') - base[s]['B']['readers'] for s in seeds], rng)
    tr = boot([r(c, s, 'answer_transfer') - base[s]['answer_transfer_readers'] for s in seeds], rng)
    near = boot([r(c, s, 'A') - flat_read(s, 6)['A'] for s in seeds], rng)
    retains = gA[0] >= .15 and gA[1] > 0 and gB[0] >= .15 and gB[1] > 0
    verdict = 'holds' if tr[1] >= -.05 else 'lost' if tr[0] <= -.05 and tr[2] < 0 else 'inconclusive'
    if retains and verdict == 'holds':
        outcome = 'long-term-works' if near[1] >= -.15 else 'retains-store-gap'
    elif retains and verdict == 'lost':
        outcome = 'retains-at-transfer-cost'
    elif retains:
        outcome = 'retains-transfer-inconclusive'
    else:
        outcome = 'no-retention-shown'

    def three(d, margin=.05):
        return {'difference': d[0], 'interval': [d[1], d[2]], 'reading': 'better' if d[0] >= margin and d[1] > 0 else 'worse' if d[0] <= -margin and d[2] < 0 else 'equivalent' if d[1] >= -margin and d[2] <= margin else 'inconclusive'}
    secondary = {json.dumps(x, sort_keys=True): {f: three(boot([r(c, s, f) - r(x, s, f) for s in seeds], rng)) for f in ('A', 'B', 'answer_transfer')} for x in ablations(c, fv)}
    mixed = {s: mixed_reads(s) for s in seeds}
    versus_mixed = {f: three(boot([r(c, s, f) - mixed[s][f] for s in seeds], rng)) for f in ('A', 'B', 'answer_transfer')}
    save_json(f'{REPORT}-analysis.json', {'outcome': outcome, 'candidate': c, 'retained_A_gain': gA, 'retained_B_gain': gB, 'answer_transfer_difference': tr, 'transfer_reading': verdict, 'near_store': near, 'secondary': secondary, 'versus_mixed': versus_mixed})
    print('outcome', outcome, gA, gB, tr, near)


if __name__ == '__main__':
    {'encoders': lambda: H.fit_encoders(PILOT_SEEDS), 'hosts': H.hosts, 'gates': lambda: gate_seeds(PILOT_SEEDS, 'gates', True), 'pilot': pilot, 'confirm-encoders': confirm_encoders, 'confirm-hosts': confirm_hosts,
     'confirm-gates': confirm_gates, 'confirm': confirm, 'analyse': analyse}[STAGE]()
