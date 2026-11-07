"""Frozen contract: docs/rule-critic-binding-v1.md. Direct-supervision binding diagnostic."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import contextlib
import io
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import rule_critic_diagnostic as D
from experiments import rule_guided_search as G
from experiments import durable as DUR

ROOT, nc = G.ROOT, G.nc
SMOKE = bool(os.environ.get('EV_SMOKE'))
NAME = 'rule-critic-binding-v1' + ('-smoke' if SMOKE else '')
CONTRACT = ROOT / 'docs/rule-critic-binding-v1.md'
RUNS, REPORT = ROOT / 'runs' / NAME, ROOT / 'reports' / NAME
ARMS = ('original', 'canonical', 'matching')


def canonical_rules(rs):
    rules = []
    for h, hv, body, form in rs['rules']:
        names = {}
        def var(v):
            if v not in names:
                names[v] = len(names)
            return names[v]
        head = tuple(var(v) for v in hv)
        body = [(r, tuple(var(v) for v in vs)) for r, vs in body]
        rules.append((h, head, body, form))
    return dict(rs, rules=rules)


class CanonicalCritic(G.Critic):
    def tokens(self, rs):
        return super().tokens(canonical_rules(rs))


class MatchingCritic(nn.Module):
    """Relative inputs only. No reference to R/orient/side metadata or symbolic validity."""
    table = G.Critic.table

    def __init__(self, rd):
        super().__init__()
        self.S = rd.S
        codes = rd.net.R_L.detach().clone()
        self.register_buffer('allcodes', F.normalize(codes, dim=1))
        stepcodes = self.allcodes[torch.as_tensor(rd.vocab)]
        self.register_buffer('codes', torch.cat([stepcodes, stepcodes, torch.zeros(2, G.SYM)]))
        V = len(rd.vocab)
        flags = torch.zeros(self.S + 2, 3)
        flags[V:self.S, 0] = 1
        flags[self.S, 1] = 1
        flags[self.S + 1, 2] = 1
        self.register_buffer('flags', flags)
        self.atom = nn.Sequential(nn.Linear(13, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.rule = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(68, 64), nn.ReLU(), nn.Linear(64, 1))

    def emb_all(self):
        return self.codes

    def tokens(self, rs):
        rs = canonical_rules(rs)
        codes, feats, groups = [], [], []
        for j, (h, hv, body, _) in enumerate(rs['rules']):
            for r, (a, b), role in [(h, hv, 0)] + [(r, v, 1 + i) for i, (r, v) in enumerate(body)]:
                base = r < rs['nb']
                codes.append(self.allcodes[rs['glob'][r]] if base else torch.zeros(G.SYM))
                feats.append(torch.cat([F.one_hot(torch.tensor(role), 3), F.one_hot(torch.tensor(a), 3),
                                        F.one_hot(torch.tensor(b), 3), torch.tensor([r == rs['ask'], base])]).float())
                groups.append(j)
        # Precompute shared comparisons; no hard threshold or equality decision.
        sims = self.codes @ torch.stack(codes).T
        return sims, torch.stack(feats), torch.tensor(groups), len(rs['rules'])

    def score(self, toks, E, last, step):
        sims, feat, groups, J = toks
        n, A = len(last), len(feat)
        x = torch.cat([sims[last, :, None], sims[step, :, None], feat[None].expand(n, A, 11)], 2)
        a = self.atom(x)
        r = torch.zeros(n, J, 64).index_add(1, groups, a)
        ctx = self.rule(r).sum(1)
        flags = torch.stack([self.flags[last, 0], self.flags[step, 0],
                             self.flags[last, 1], self.flags[step, 2]], 1)
        return torch.sigmoid(self.out(torch.cat([ctx, flags], 1))[:, 0])


class SymbolicReference:
    """Evaluation-only exact R* language from diagnostic metadata."""
    def __init__(self, rd):
        self.rd = rd

    def table(self, rs):
        S, s = self.rd.S, D.star(self.rd, rs)
        C = np.zeros((S + 1, S + 1), dtype=np.float32)
        C[S, s] = C[s, s] = C[s, S] = 1
        return C


def oracle_graph(rd):
    out = {}
    for r in rd.vocab:
        M = np.zeros((rd.N, rd.N), dtype=bool)
        for a, b in rd.W.true[r]:
            M[a, b] = a != b
        out[r] = M
    out[rd.kind] = G.RA.reduce_kind(out[rd.kind], out[rd.sub])
    return out


def nearest_reads(critic, rd, sets):
    codes = F.normalize(rd.net.R_L.detach()[torch.as_tensor(rd.vocab)], dim=1).numpy()
    cos = codes @ codes.T
    np.fill_diagonal(cos, -np.inf)
    nearest = cos.argmax(1)
    out = {str(k): [0, 0] for k in range(2, 13)}
    margins = []
    for rs in sets:
        C = critic.table(rs)
        s = D.star(rd, rs)
        V = len(rd.vocab)
        other = int(nearest[s % V]) + (s // V) * V
        margins.append(float(1 - cos[s % V, other % V]))
        for k in range(2, 13):
            for i in range(k):
                p = [s] * k
                p[i] = other
                out[str(k)][0] += int(G.program_accepts(C, p, rd.S))
                out[str(k)][1] += 1
    return {'accepted': {k: a / n for k, (a, n) in out.items()}, 'counts': out,
            'minimum_cosine_gap': min(margins), 'mean_cosine_gap': float(np.mean(margins))}


def make_data(rd, orc, seed):
    tr, held, combos = D.split(rd, orc)
    rng = np.random.default_rng([seed, 21])
    rsets = []
    for _ in range(D.N_RS):
        R = int(tr[rng.integers(len(tr))])
        allowed = [c for c in D.COMBOS if c != combos[R]]
        o, side = allowed[rng.integers(len(allowed))]
        rsets.append(D.make_rs(rd, R, o, side, rng))
    return rsets, D.eval_sets(rd, tr, held, combos, seed), {
        'train_relations': tr, 'held_relations': held, 'held_combinations': combos}


def verify():
    """Mechanical tests, not a quality gate for model/hyperparameter selection."""
    torch.set_num_threads(1)
    rd = G.Readers(450)
    rng = np.random.default_rng(77)
    for o, side in D.COMBOS:
        rs = D.make_rs(rd, rd.vocab[0], o, side, rng)
        sw = {0: 2, 1: 0, 2: 1}
        renamed = dict(rs, rules=[(h, tuple(sw[v] for v in hv),
                                   [(r, tuple(sw[v] for v in vs)) for r, vs in body], f)
                                  for h, hv, body, f in rs['rules']])
        assert canonical_rules(rs)['rules'] == canonical_rules(renamed)['rules']
        c = MatchingCritic(rd)
        C = c.table(rs)
        assert np.array_equal(C, c.table(renamed))
        assert np.allclose(C, c.table(dict(rs, rules=list(reversed(rs['rules'])))), atol=1e-6)
        assert np.array_equal(C, c.table(dict(rs, sym=G.unitv(rng, 2))))
        poisoned = dict(rs, R=-999, orient='invalid', side='invalid')
        assert np.array_equal(C, c.table(poisoned))
        s = D.star(rd, rs)
        P = [(s, -1), (s, s)]
        lp = D.prog_logp(c, rs, c.emb_all(), P)
        manual = [C[rd.S, s] * C[s, rd.S], C[rd.S, s] * C[s, s] * C[s, rd.S]]
        assert np.allclose(lp.detach().exp(), manual)
        lp.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in c.parameters())
    _, ev, _ = make_data(rd, oracle_graph(rd), 450)
    ref = SymbolicReference(rd)
    for sets in ev.values():
        reads = D.program_reads(ref, rd, sets, 450)
        assert all(v == 1 for v in reads['valid_accepted'].values())
        assert all(v in (None, 0) for d in reads['negative_accepted'].values() for v in d.values())
        assert all(v == 0 for v in reads['invariance_changed'].values())
        assert all(v == 0 for v in nearest_reads(ref, rd, sets)['accepted'].values())
    print('Verification passed: normalization, symmetry, metadata isolation, scoring/gradients, exact reference.', flush=True)


def run(arm, seed):
    torch.set_num_threads(1)
    ph = {}
    @contextlib.contextmanager
    def phase(name):
        w, c = time.perf_counter(), time.process_time()
        yield
        ph[name] = {'wall_seconds': time.perf_counter() - w, 'cpu_seconds': time.process_time() - c}
    with phase('construction'):
        rd = G.Readers(seed)
        orc = oracle_graph(rd)
        rsets, ev, split = make_data(rd, orc, seed)
        torch.manual_seed(seed)
        critic = (G.Critic(rd, 'rules') if arm == 'original' else
                  CanonicalCritic(rd, 'rules') if arm == 'canonical' else MatchingCritic(rd))
    checks = {'training': 0, 'evaluation': 0}
    mode = ['training']
    original_score = critic.score
    def score(*args):
        checks[mode[0]] += len(args[2])
        return original_score(*args)
    critic.score = score
    audit, exposures = set(), [0]
    rsidx = {id(rs): i for i, rs in enumerate(rsets)}
    old_prog = D.prog_logp
    def prog(*args):
        rs, programs = args[1], args[3]
        for p in programs:
            audit.add((rsidx[id(rs)], tuple(int(x) for x in p)))
        exposures[0] += len(programs)
        return old_prog(*args)
    D.prog_logp = prog
    ckpts, ckcost = {}, []
    def on_ckpt(step):
        mode[0] = 'evaluation'
        with phase(f'checkpoint_{step}'):
            with torch.no_grad():
                ckpts[step] = {k: D.program_reads(critic, rd, v, seed, full=False) for k, v in ev.items()}
        ckcost.append(ph[f'checkpoint_{step}'])
        mode[0] = 'training'
        print(f'{arm} seed {seed}: checkpoint {step}', flush=True)
    with phase('training_including_checkpoints'):
        log = D.train(critic, 'direct', rsets, [], rd, seed, on_ckpt)
    D.prog_logp = old_prog
    ph['training'] = {k: ph['training_including_checkpoints'][k] - sum(c[k] for c in ckcost)
                      for k in ('cpu_seconds', 'wall_seconds')}
    del ph['training_including_checkpoints']
    mode[0] = 'evaluation'
    with phase('final_evaluation'):
        with torch.no_grad():
            final = {k: D.program_reads(critic, rd, v, seed) for k, v in ev.items()}
            nearest = {k: nearest_reads(critic, rd, v) for k, v in ev.items()}
    with phase('symbolic_reference'):
        ref = SymbolicReference(rd)
        reference = {k: D.program_reads(ref, rd, v, seed) for k, v in ev.items()}
        ref_nearest = {k: nearest_reads(ref, rd, v) for k, v in ev.items()}
        for x in reference.values():
            assert all(v == 1 for v in x['valid_accepted'].values())
            assert all(v in (0, None) for d in x['negative_accepted'].values() for v in d.values())
            assert all(v == 0 for v in x['invariance_changed'].values())
        assert all(v == 0 for x in ref_nearest.values() for v in x['accepted'].values())
    with phase('save'):
        RUNS.mkdir(parents=True, exist_ok=True)
        ck = RUNS / f'{arm}-seed{seed}.pt'
        torch.save(critic.state_dict(), ck)
    params = sum(p.numel() for p in critic.parameters())
    buffers = sum(b.numel() * b.element_size() for b in critic.buffers())
    res = {'arm': arm, 'seed': seed, 'split': split, 'log': log, 'checkpoints': ckpts, 'final': final,
           'nearest': nearest, 'symbolic_reference': reference, 'symbolic_nearest': ref_nearest,
           'checkpoint': {'file': str(ck.relative_to(ROOT)), 'sha256': nc.sha256(ck)},
           'reader_sha256': rd.ck_sha}
    rt = {'phases': ph, 'optimizer_steps': D.STEPS, 'training_rule_sets': len(rsets),
          'independent_training_examples': len(audit), 'training_program_exposures': exposures[0],
          'local_checks_scored': checks, 'parameters': params, 'parameter_bytes': params * 4,
          'gradient_bytes': params * 4, 'adam_moment_bytes_upper_bound': params * 8,
          'adam_step_counter_bytes_upper_bound': len(list(critic.parameters())) * 4,
          'buffer_bytes': buffers, 'checkpoint_bytes': ck.stat().st_size,
          'reader_parameter_bytes': sum(p.numel() * p.element_size() for p in rd.model.parameters()),
          'oracle_graph_bytes': sum(x.nbytes for x in orc.values()),
          'rule_set_json_bytes': len(json.dumps(rsets, default=lambda x: x.tolist()).encode()),
          'audit_serialized_bytes': len(json.dumps(sorted(audit)).encode()),
          'python_container_heap_bytes': None, 'python_container_heap_reason': 'Included in RSS; not separately profiled',
          'raw_training_examples_retained_on_disk': False, 'fits': 1, 'tuning_fits': 0,
          'candidate_searches': 0, 'inference_benchmark': None,
          'inference_benchmark_reason': 'Whole evaluation timing only; no latency or scaling claim'}
    return res, rt


def task(t):
    w, c = time.perf_counter(), time.process_time()
    res, rt = run(*t)
    rt.update(wall_seconds=time.perf_counter() - w, cpu_seconds=time.process_time() - c,
              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(f'Finished {t[0]} seed {t[1]}', flush=True)
    return f'{t[0]}/seed{t[1]}', res, rt


def summarize(results):
    out = {}
    for arm in ARMS:
        rs = [r for r in results.values() if r['arm'] == arm]
        def stats(xs):
            return {'mean': float(np.mean(xs)), 'min': min(xs), 'max': max(xs), 'seeds': xs}
        item = {}
        for split in ('held_relations', 'held_instances'):
            valid = {str(k): stats([r['final'][split]['valid_accepted'][str(k)] for r in rs]) for k in D.LENS}
            neg = {kind: stats([float(np.mean([r['final'][split]['negative_accepted'][kind][str(k)]
                                              for k in range(2, 13)])) for r in rs]) for kind in D.NEG_KINDS}
            nearest = stats([float(np.mean(list(r['nearest'][split]['accepted'].values()))) for r in rs])
            inv = {k: stats([r['final'][split]['invariance_changed'][k] for r in rs])
                   for k in rs[0]['final'][split]['invariance_changed']}
            item[split] = {'valid': valid, 'negatives_mean_over_lengths_2_12': neg,
                           'nearest_mean_over_lengths_2_12': nearest, 'invariance': inv}
        f = item['held_relations']
        item['passes_mean_gate'] = (all(f['valid'][str(k)]['mean'] >= .9 for k in range(3, 13)) and
            all(f['negatives_mean_over_lengths_2_12'][k]['mean'] <= .1 for k in D.NEG_KINDS[:4]) and
            f['nearest_mean_over_lengths_2_12']['mean'] <= .1 and all(x['mean'] <= .02 for x in f['invariance'].values()))
        out[arm] = item
    return out


def main():
    w = time.perf_counter()
    workers = int(os.environ.get('EV_WORKERS', 3))
    tasks = [(a, s) for a in ARMS for s in D.SEEDS]
    source_files = [Path(__file__), Path(D.__file__), Path(G.__file__)]
    hashes = {str(p.relative_to(ROOT)): nc.sha256(p) for p in source_files}
    with __import__('multiprocessing').get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        out = pool.map(DUR.durable(task, RUNS / 'done', '{0}-seed{1}'), tasks, chunksize=1)
    results = {k: r for k, r, _ in out}
    # Roundtrip gives uniform JSON string keys for in-memory and resumed runs.
    results = DUR.roundtrip(results)
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        np.show_config()
        blas = buf.getvalue()
    rt = G.V2.runtime({'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
        'source_hashes': hashes, 'workers': workers, 'wall_seconds': time.perf_counter() - w,
        'aggregate_worker_cpu_seconds': sum(r['cpu_seconds'] for _, _, r in out),
        'tasks': {k: r for k, _, r in out}, 'blas_configuration': blas,
        'torch_configuration': torch.__config__.show(), 'seed_policy': '450–453; split 2029/2030; paired data RNG',
        'dirty_including_untracked': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip()),
        'cpu_model': next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                           if s.startswith('model name')), None),
        'memory_limit': {'address_space_rlimit_bytes': resource.getrlimit(resource.RLIMIT_AS)},
        'whole_job_peak_memory_bytes': None, 'whole_job_peak_memory_reason': 'Only per-process ru_maxrss measured',
        'selection_policy': 'All arms and final checkpoint reported; no tuning or checkpoint selection'})
    G.V2.save_json(f'{REPORT}.json', {'status': NAME, 'results': results})
    G.V2.save_json(f'{REPORT}-runtime.json', rt)
    summary = summarize(results)
    G.V2.save_json(f'{REPORT}-summary.json', summary)
    print(json.dumps({a: {'passes': x['passes_mean_gate'], 'valid': x['held_relations']['valid']['3'],
                          'nearest': x['held_relations']['nearest_mean_over_lengths_2_12']}
                      for a, x in summary.items()}, indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        w, c = time.perf_counter(), time.process_time()
        verify()
        G.V2.save_json(ROOT / 'reports/rule-critic-binding-v1-verification-runtime.json', {
            'wall_seconds': time.perf_counter() - w, 'cpu_seconds': time.process_time() - c,
            'source_sha256': nc.sha256(__file__), 'contract_sha256': nc.sha256(CONTRACT),
            'scope': 'Mechanical verification, no training'})
    else:
        main()
