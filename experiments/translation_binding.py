"""Translation memory v1, binding stage (docs/translation-memory-v1-draft.md, "Binding revision"): a bounded one-step
source-to-target memory on the frozen keys, cooperating with the changing host through learned routing.

Replays the host stage's saved states (the host never reads the memory). Each batch, in order:
  evaluation during acquisition (saved mid-batch states) and immediately before consolidation: current associations absent
  gate examples in the pre-write state, then each buffered, unwritten entry written once, then gate examples in the post-write
  state; the gate alone trains on both, with recorded labels; the sentence rule alone trains on the batch's sentences
  evaluation after the write
The final point (after the context-only block) is evaluation only. Every evaluation is state-preserving and mixes current and
older words; no router or gate is told a word's batch, age or write status.

Write source (a condition): `label` (the recorded training label, primary) or `teacher` (the host's answer at consolidation).
Memories: `onestep` (the system), `assoc` (the joint recurrent network at the same sizes, read at 16 steps, a comparison),
`flat-sparse` (slots on the same separated patterns), `flat-store` (slots on the dense frozen keys). All use the same keys, buffered
renderings, write events and write source. Each has its own gate, router and sentence rule at the same budget.

Stages (argv[1]):
  select  seed 299: the two size gates under label writes, then the selected size under teacher writes; seed 297: confirmation;
          router thresholds on seed 299 batches 2-3, pre-write and post-write -> reports/<NAME>-selection.json
  run     both sources, both arms, the host stage's seeds, at the selected size -> reports/<NAME><TM_PASS_TAG>-results.json.gz

Usage: TM_MID=250,1000 TM_BRANCHES=main NFI_TAG=-mid TM_STABLE_KEYS=1 .venv/bin/python -m experiments.translation_binding select
"""
import gzip
import json
import multiprocessing as mp
import os
import resource
import sys
import time

import numpy as np
import torch

from experiments import durable as DUR
from experiments import generic_relations as G
from experiments import limited_memory as LM
from experiments import number_systems as X
from experiments import translation_memory as TM
from experiments import translation_stable_keys as TK
from experiments import translation_store as TS

nc = X.nc
SMOKE = X.SMOKE
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'select'
GRID = tuple((m, k) for m in (256, 512, 1024) for k in (round(0.02 * m), round(0.04 * m)))
EXACT_GATE, FLAT_TOLERANCE = (0., 1.) if SMOKE else (0.95, 0.02)
SEL_SEED, CONF_SEED = 299, 297
GATE_STEPS, GATE_LR = 300, 0.05
SENT_TRAIN, SENT_TEST, SENT_STEPS, SENT_LR = 200, 100, 200, 0.1
CORRUPTION = (0., .2, .4, .6)
FREE_STARTS = 200
RECURRENT_STEPS = max(TS.CHECKPOINTS)
MEMORIES = ('onestep', 'assoc', 'flat-sparse', 'flat-store')
SOURCES = ('label', 'teacher')
OLD_FINAL = range(4)  # the primary old-word cohort after all writes: batches 1-4, which have left the buffer
NAME = 'translation-memory-v1-binding' + ('-smoke' if SMOKE else '')
PREFIX = X.ROOT / 'reports' / NAME
SELECTION = f'{PREFIX}-selection.json'
WORKERS = X.WORKERS
HOST = {}


# ---------------------------------------------------------------- host states and keys

def host_results():
    if not HOST:
        with gzip.open(X.ROOT / 'reports' / f'{TM.NAME}-results.json.gz', 'rt') as f:
            HOST.update(json.load(f)['results'])
    return HOST


class Saved:
    def __init__(self, arm, seed):
        rec = host_results()[f'{arm}/seed{seed}']
        self.arm, self.seed = arm, seed
        self.points = torch.load(X.ROOT / rec['checkpoint']['file'], weights_only=False)['main']
        self.meta = rec['branches']['main']['points']
        self.mat = TM.material(seed)

    def model(self, state):
        LM.CURRENT.update(arm=self.arm, seed=self.seed, step=0, r=None)
        torch.manual_seed(self.seed)
        enc, model = X.make_model(self.seed)
        model.load_state_dict(state['sri']); enc['image'].load_state_dict(state['image']); enc['sequence'].load_state_dict(state['sequence'])
        enc['image'].eval()
        return enc, model


def key_encoder(seed):
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    fit = torch.load(TK.RUNS / 'fits' / f"{c['arch']}-key{c['key']}-seed{seed}.pt", weights_only=False)
    model = TK.build(c['arch'], c['key'], seed)
    model.load_state_dict(fit['checkpoints'][c['steps']]['state'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, c['key'], c['bytes']


class HostView:
    """The host at one saved state: its answer distribution for images (over all target words)."""

    @torch.no_grad()
    def __init__(self, saved, state):
        self.enc, self.model = saved.model(state)
        self.code = G.codes_fn(self.enc, self.model)
        self.targets = self.code(saved.mat['tgt_train'][:, 0], 'image')

    @torch.no_grad()
    def probs(self, x):
        q = self.code(np.ascontiguousarray(x.reshape(-1, 28, 28), dtype=np.float32), 'image')
        return torch.softmax(G.pair(self.model, q, self.targets[None].expand(len(q), -1, -1), 1), 1)


# ---------------------------------------------------------------- memories

class Memory:
    """One memory with its own gate, sentence rule and routing inputs."""

    def __init__(self, name, seed, cfg, width):
        m, k = cfg
        self.name, self.k = name, k
        self.sep = TS.Separation(width, m, k, nc.stream('tm_separation', seed, m, k))
        self.cb = TS.codebook(TM.N_TGT, m, k, nc.stream('tm_codebook', seed, m, k))
        self.gate, self.rule = TS.Gate(), TS.SentenceRule()
        rng = nc.stream('tm_assoc', seed, m, k)
        self.mem = {'onestep': lambda: TS.OneStep(m, m, k, k, rng), 'assoc': lambda: TS.Associative(m, m, k, k, rng),
                    'flat-sparse': lambda: TS.FlatSparse(TM.N_WORDS, TM.N_TGT, k), 'flat-store': lambda: TS.FlatStore(TM.N_WORDS, TM.N_TGT)}[name]()

    def pattern(self, keys):
        return self.sep(keys, torch.zeros(keys.shape[1]), 1.)

    def write(self, keys, targets):
        if self.name in ('onestep', 'assoc'):
            self.mem.write(self.pattern(keys), self.cb[targets])
        elif self.name == 'flat-sparse':
            self.mem.write(self.pattern(keys), targets.tolist())
        else:
            self.mem.write(keys, targets.tolist())

    def read(self, keys, training=False):
        """-> scores (n, T) in [0, 1], reinstated target part (n, m_t), converged (n,), info."""
        n = len(keys)
        if self.name == 'onestep':
            s = self.pattern(keys)
            tp = self.mem.read(s)
            return TS.target_scores(tp, self.cb), tp, torch.ones(n), {'active_connections': self.mem.active_connections(s), 'dense_ops': self.mem.dense_ops()}
        if self.name == 'assoc':
            out, info = self.mem.retrieve(self.pattern(keys))
            tp = out[RECURRENT_STEPS]
            info['checkpoints'] = {t: TS.target_scores(v, self.cb) for t, v in out.items()}
            return TS.target_scores(tp, self.cb), tp, info['converged'].float(), info
        sc, win = self.mem.read(self.pattern(keys) if self.name == 'flat-sparse' else keys)
        if training:
            self.mem.count_use(win)
        return sc, self.cb[sc.argmax(1)], torch.ones(n), {}

    def digest(self):
        m = self.mem
        parts = [m.W] if hasattr(m, 'W') else [torch.stack(m.keys) if m.keys else torch.zeros(1), torch.tensor(m.values + m.used)]
        return TS.digest(*parts, *self.gate.parameters(), *self.rule.parameters())


# ---------------------------------------------------------------- one replay

def triples(rng, words, n, exclude=()):
    out, seen = [], set(map(tuple, exclude))
    while len(out) < n:
        t = tuple(int(w) for w in rng.choice(words, 3, replace=False))
        if t not in seen:
            out.append(t)
    return np.array(out)


def replay(arm, seed, cfg, source, names=MEMORIES, full=True, thresholds=None):
    """`full` adds the mid-batch evaluations, gates, sentences and extras; without it only the post-write evaluations and the
    pre-write ones (the size and router selections need nothing else)."""
    saved = Saved(arm, seed)
    mat = saved.mat
    labels = torch.tensor(mat['tgt'])
    key_model, width, key_bytes = key_encoder(seed)
    key_digest = TS.digest(*key_model.parameters())
    keys_of = lambda x: TK.encode(key_model, x).cpu()
    E = keys_of(mat['src_eval'])
    eval_digest = TS.digest(E)
    M = {n: Memory(n, seed, cfg, width) for n in names}
    written, exact, train_sents = {}, {}, []
    srng = nc.stream('tb_sentence_draws', seed)
    out = {'evaluations': [], 'writes': [], 'gate_loss': {n: [] for n in names}}

    def evaluate(view, batch, moment, step):
        before = {n: m.digest() for n, m in M.items()}
        ev = evaluation(M, view, E, mat, labels, batch, written, full, thresholds, train_sents, nc.stream('tb_sentence_test', seed, batch, moment))
        ok = all(M[n].digest() == before[n] for n in M) and TS.digest(*key_model.parameters()) == key_digest and TS.digest(keys_of(mat['src_eval'])) == eval_digest
        out['evaluations'].append({'batch': batch + 1, 'moment': moment, 'step': step, 'state_preserved_and_keys_identical': ok, **ev})

    for b in range(TM.BATCHES):
        pt = saved.points[b]
        if full:
            for q in pt['mids']:
                evaluate(HostView(saved, q['host']), b, f"offset_{q['offset']}", q['step'])
        view = HostView(saved, pt['host'])
        evaluate(view, b, 'pre_write', pt['step'])
        buf = saved.meta[b]['buffer']
        bimg = np.stack([mat['src_train'][e['word'], e['render']] for e in buf])
        bk, ph = keys_of(bimg), view.probs(bimg).cpu()
        bw = torch.tensor([e['word'] for e in buf])
        examples = {}
        if full:  # gate examples before writing: this batch's associations are absent from memory
            for n, m in M.items():
                sc, _, conv, _ = m.read(bk, training=True)
                examples[n] = [(TS.features(ph, sc, conv), TS.p_memory(sc))]
        new = [i for i, e in enumerate(buf) if e['word'] not in written]
        ws = bw[new]
        host_answer = ph[new].argmax(1)
        target = labels[ws] if source == 'label' else host_answer
        for w, a in zip(ws.tolist(), target.tolist()):
            written[w] = a
        for m in M.values():
            m.write(bk[new], target)
        if names and ('onestep' in M or 'assoc' in M):
            pat = next(M[n] for n in ('onestep', 'assoc') if n in M).pattern(bk[new])
            for w, sp, a in zip(ws.tolist(), pat, target.tolist()):
                exact[w] = (sp, a)
        out['writes'].append({'batch': b + 1, 'words': ws.tolist(), 'written': target.tolist(), 'written_correct': float((target == labels[ws]).float().mean()),
                              'host_answer_correct': float((host_answer == labels[ws]).float().mean())})
        if full:
            for n, m in M.items():  # gate examples after writing, then the gate alone trains on both states, same budget
                sc, _, conv, _ = m.read(bk, training=True)
                examples[n].append((TS.features(ph, sc, conv), TS.p_memory(sc)))
                f = torch.cat([examples[n][0][0], examples[n][1][0]])
                pm = torch.cat([examples[n][0][1], examples[n][1][1]])
                out['gate_loss'][n].append(m.gate.fit(f, pm, ph.repeat(2, 1), labels[bw].repeat(2), GATE_STEPS, GATE_LR))
            tr = triples(srng, TM.batch_words(b), SENT_TRAIN)
            train_sents.extend(tr.tolist())
            rj = srng.integers(TM.R_TRAIN, size=tr.shape)
            imgs = mat['src_train'][tr, rj]
            xk, xh = keys_of(imgs), view.probs(imgs).cpu()
            for m in M.values():
                sc, tp, conv, _ = m.read(xk, training=True)
                with torch.no_grad():
                    w = m.gate(TS.features(xh, sc, conv))
                r = TS.routed_patterns(w, tp, xh, m.cb).reshape(len(tr), 3, -1)
                m.rule.fit(r, labels[torch.from_numpy(tr)].flip(1), m.cb, SENT_STEPS, SENT_LR)
        evaluate(view, b, 'post_write', pt['step'])
        out['evaluations'][-1]['exact_recovery'] = exact_recovery(M, exact)
    last = saved.points[TM.BATCHES]
    evaluate(HostView(saved, last['host']), TM.BATCHES - 1, 'final', last['step'])
    if full:
        out['extras'] = extras(M, exact, seed)
    out['written'] = {str(w): a for w, a in written.items()}
    out['bytes'] = {'memory': {n: float(m.mem.nbytes()) for n, m in M.items()}, 'key_encoder': key_bytes,
                    'buffer': TM.CAPACITY * (28 * 28 * 4 + 2 + 2 + 4 + 1)}  # rendering, label, teacher answer, entry step, written flag
    return out


def exact_recovery(M, exact):
    """Recovery of the written answer from every exact written pattern so far."""
    if not exact:
        return {}
    words = sorted(exact)
    S = torch.stack([exact[w][0] for w in words])
    y = torch.tensor([exact[w][1] for w in words])
    r = {}
    if 'onestep' in M:
        r['onestep'] = float((TS.target_scores(M['onestep'].mem.read(S), M['onestep'].cb).argmax(1) == y).float().mean())
    if 'assoc' in M:
        res, _ = M['assoc'].mem.retrieve(S)
        r['assoc_16'] = float((TS.target_scores(res[RECURRENT_STEPS], M['assoc'].cb).argmax(1) == y).float().mean())
    return r


@torch.no_grad()
def evaluation(M, view, E, mat, labels, batch, written, full, thresholds, train_sents, rng):
    """All words presented so far (current and older together), independent evaluation renderings. Per batch: host, memory alone
    (true translation, and fidelity to the written answer where one exists), gate, router."""
    R = TM.R_EVAL
    words = list(range(TM.WORDS * (batch + 1)))
    idx = torch.tensor([w * R + r for w in words for r in range(R)])
    y = labels.repeat_interleave(R)[idx]
    bof = torch.tensor([w // TM.WORDS for w in words]).repeat_interleave(R)
    ph = view.probs(mat['src_eval'].reshape(-1, 28, 28)[idx.numpy()]).cpu()
    per = lambda v: {str(b + 1): float(v[bof == b].float().mean()) for b in range(batch + 1)}
    res = {'host': per(ph.argmax(1) == y), 'memories': {}}
    wr = torch.tensor([written.get(w, -1) for w in words]).repeat_interleave(R)
    for n, m in M.items():
        sc, tp, conv, info = m.read(E[idx])
        ans = sc.argmax(1)
        r = {'memory': per(ans == y), 'fidelity': {k: v for k, v in per(ans == wr).items() if any(w // TM.WORDS == int(k) - 1 for w in written)}}
        r['router'] = {str(th): per(TS.route(th, ph, sc) == y) for th in TS.ROUTER_THRESHOLDS}
        if 'checkpoints' in info:
            r['checkpoints'] = {str(t): per(s.argmax(1) == y) for t, s in info['checkpoints'].items()}
            r['converged'] = float(info['converged'].float().mean())
            r['cycle'] = float(info['cycle'].float().mean())
            r['active_connections_mean'] = float(info['active_connections'].mean())
            r['dense_ops'] = int(info['dense_ops'])
        elif 'active_connections' in info:
            r['active_connections_mean'] = float(info['active_connections'].mean())
            r['dense_ops'] = int(info['dense_ops'])
        if full:
            pick, w = m.gate.answer(TS.features(ph, sc, conv), TS.p_memory(sc), ph)
            r['gate'], r['gate_weight'] = per(pick == y), per(w)
            if thresholds is not None:
                r['router_selected'] = r['router'][str(thresholds[n])]
            old = [x for x in words if x // TM.WORDS < batch]
            if len(old) >= 3:
                r['sentences'] = sentence_test(m, view, E, mat, labels, old, train_sents, rng)
        res['memories'][n] = r
    return res


@torch.no_grad()
def sentence_test(m, view, E, mat, labels, old, train_sents, rng):
    te = triples(rng, np.array(old), SENT_TEST, exclude=train_sents)
    rj = rng.integers(TM.R_EVAL, size=te.shape)
    xh = view.probs(mat['src_eval'][te, rj]).cpu()
    sc, tp, conv, _ = m.read(E[torch.from_numpy(te.reshape(-1) * TM.R_EVAL + rj.reshape(-1))])
    single, w = m.gate.answer(TS.features(xh, sc, conv), TS.p_memory(sc), xh)
    r = TS.routed_patterns(w, tp, xh, m.cb).reshape(len(te), 3, -1)
    ok = m.rule.logits(r, m.cb).argmax(2) == labels[torch.from_numpy(te)].flip(1)
    return {'position': [float(ok[:, j].float().mean()) for j in range(3)], 'whole': float(ok.all(1).float().mean()),
            'single_word': float((single == labels[torch.from_numpy(te)].reshape(-1)).float().mean()), 'mixing': torch.softmax(m.rule.theta, 1).tolist()}


@torch.no_grad()
def extras(M, exact, seed):
    """Corrupted exact patterns (recovery of the written answer) and, for the recurrent comparison, free recall."""
    words = sorted(exact)
    S = torch.stack([exact[w][0] for w in words])
    y = torch.tensor([exact[w][1] for w in words])
    rng = nc.stream('tb_extras', seed)
    k = int(S[0].sum())
    out = {}
    for frac in CORRUPTION:
        noisy, j = S.clone(), int(round(frac * k))
        for i in range(len(noisy)):
            on, off = noisy[i].nonzero()[:, 0], (noisy[i] == 0).nonzero()[:, 0]
            noisy[i, on[torch.from_numpy(rng.choice(len(on), j, replace=False))]] = 0
            noisy[i, off[torch.from_numpy(rng.choice(len(off), j, replace=False))]] = 1
        r = {'onestep': float((TS.target_scores(M['onestep'].mem.read(noisy), M['onestep'].cb).argmax(1) == y).float().mean())}
        res, _ = M['assoc'].mem.retrieve(noisy)
        r['assoc'] = {str(t): float((TS.target_scores(v, M['assoc'].cb).argmax(1) == y).float().mean()) for t, v in res.items()}
        r['onestep_equals_joint_step_1'] = torch.equal(M['onestep'].mem.read(noisy), res[1])
        sc, _ = M['flat-sparse'].mem.read(noisy)
        r['flat-sparse'] = float((sc.argmax(1) == y).float().mean())
        out[f'corrupt_{frac}'] = r
    v = M['assoc']
    res, info = v.mem.retrieve(TS.codebook(FREE_STARTS, S.shape[1], k, rng))
    best = TS.target_scores(res[RECURRENT_STEPS], v.cb).max(1)
    hit = (best.values >= 0.8) & torch.isin(best.indices, y)
    out['free_recall'] = {'starts': FREE_STARTS, 'distinct_stored_regenerated': int(best.indices[hit].unique().numel()), 'stored_pairs': len(words),
                          'spurious_fraction': float((~hit).float().mean()), 'converged': float(info['converged'].float().mean())}
    return out


# ---------------------------------------------------------------- stages

def after_all_writes(res):
    """The post-write evaluation of batch 6: after all 120 writes."""
    return next(e for e in res['evaluations'] if e['batch'] == TM.BATCHES and e['moment'] == 'post_write')


def size_row(res):
    e = after_all_writes(res)
    row = {'exact_recovery': e['exact_recovery'], 'written_correct': float(np.mean([w['written_correct'] for w in res['writes']]))}
    for n, r in e['memories'].items():
        row[n] = {'all_words': float(np.mean(list(r['memory'].values()))), 'old_words': float(np.mean([r['memory'][str(b + 1)] for b in OLD_FINAL]))}
    return row


def pool(rows):
    """Mean over arms of every number in the size rows."""
    def avg(vals):
        return {k: avg([v[k] for v in vals]) for k in vals[0]} if isinstance(vals[0], dict) else float(np.mean(vals))
    return avg(rows)


def passes(row):
    return {'exact': row['exact_recovery']['onestep'] >= EXACT_GATE,
            'independent': row['onestep']['all_words'] >= row['flat-store']['all_words'] - FLAT_TOLERANCE}


def starmap(jobs):
    with mp.get_context('fork').Pool(min(WORKERS, len(jobs))) as pool_:
        return pool_.starmap(replay, jobs)


def select():
    t0 = time.perf_counter()
    arms = TM.ARMS
    timing = {}
    t = time.perf_counter()
    jobs = [(arm, SEL_SEED, cfg, 'label', MEMORIES, False, None) for cfg in GRID for arm in arms]
    rows = {}
    for job, res in zip(jobs, starmap(jobs)):
        rows.setdefault(str(job[2]), {})[job[0]] = size_row(res)
    table = {c: {'per_arm': v, 'pooled': pool(list(v.values()))} for c, v in rows.items()}
    for c in table:
        table[c]['passes'] = passes(table[c]['pooled'])
    timing['grid_seconds'] = time.perf_counter() - t
    ok = [cfg for cfg in GRID if all(table[str(cfg)]['passes'].values())]
    chosen = ok[0] if ok else None
    out = {'status': 'pilot selection; the tolerance is an engineering selection rule, not a demonstrated statistical equivalence',
           'rule': {'exact_gate': EXACT_GATE, 'flat_store_tolerance': FLAT_TOLERANCE, 'write_source': 'label', 'order': 'fewest units, then fewest active'},
           'seeds': {'selection': SEL_SEED, 'confirmation': CONF_SEED}, 'grid': table, 'passing': ok, 'chosen': chosen, 'teacher_check': None, 'confirmation': None, 'router': None}
    if chosen is not None:
        t = time.perf_counter()
        jobs = [(arm, SEL_SEED, chosen, 'teacher', MEMORIES, False, None) for arm in arms] + [(arm, CONF_SEED, chosen, s, MEMORIES, False, None) for arm in arms for s in SOURCES]
        results = starmap(jobs)
        got = {(j[0], j[1], j[3]): size_row(r) for j, r in zip(jobs, results)}
        out['teacher_check'] = {arm: got[(arm, SEL_SEED, 'teacher')] for arm in arms}
        label_conf = pool([got[(arm, CONF_SEED, 'label')] for arm in arms])
        out['confirmation'] = {'label': label_conf, 'passes': passes(label_conf), 'teacher': {arm: got[(arm, CONF_SEED, 'teacher')] for arm in arms}}
        timing['check_and_confirm_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        jobs = [(arm, SEL_SEED, chosen, s, MEMORIES, False, None) for arm in arms for s in SOURCES]
        results = starmap(jobs)
        out['router'] = {s: router_choice([r for j, r in zip(jobs, results) if j[3] == s]) for s in SOURCES}
        timing['router_seconds'] = time.perf_counter() - t
    DUR.atomic_write(SELECTION, (json.dumps(out, indent=1, sort_keys=True, default=float) + '\n').encode())
    DUR.atomic_write(f'{PREFIX}-selection-runtime.json', (json.dumps({'scope': 'nondeterministic timing of the selection', 'wall_seconds': time.perf_counter() - t0, **timing, 'workers': WORKERS,
                     'device': 'cpu', 'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'revision': X.git('rev-parse', 'HEAD'),
                     'dirty': bool(X.git('status', '--porcelain'))}, indent=1, sort_keys=True, default=float) + '\n').encode())
    print(json.dumps({'passing': ok, 'chosen': chosen, 'confirmation': out['confirmation'] and out['confirmation']['passes'], 'router': out['router'] and {s: out['router'][s]['chosen'] for s in SOURCES}}))


def router_choice(results):
    """Per memory: the threshold with the best mean over batches 2-3, pre-write and post-write, of (current + previous words) / 2;
    ties go to the earlier option. The pre-write states are what give a threshold a job."""
    scores = {n: {} for n in MEMORIES}
    for n in MEMORIES:
        for th in TS.ROUTER_THRESHOLDS:
            vals = []
            for r in results:
                for e in r['evaluations']:
                    if e['batch'] in (2, 3) and e['moment'] in ('pre_write', 'post_write'):
                        acc = e['memories'][n]['router'][str(th)]
                        cur = acc[str(e['batch'])]
                        vals.append((cur + np.mean([a for b, a in acc.items() if int(b) < e['batch']])) / 2)
            scores[n][str(th)] = float(np.mean(vals))
    return {'scores': scores, 'chosen': {n: max(TS.ROUTER_THRESHOLDS, key=lambda th: (scores[n][str(th)], -TS.ROUTER_THRESHOLDS.index(th))) for n in MEMORIES}}


def run_task(task):
    arm, seed, source = task
    t0, c0 = time.perf_counter(), time.process_time()
    sel = json.loads(open(SELECTION).read())
    cfg = tuple(sel['chosen'])
    if os.environ.get('TM_MEMORY_OVERRIDE'):
        cfg = tuple(int(v) for v in os.environ['TM_MEMORY_OVERRIDE'].split(','))
    res = replay(arm, seed, cfg, source, MEMORIES, True, sel['router'][source]['chosen'])
    res['config'] = {'memory': cfg, 'source': source}
    return f'{arm}/seed{seed}/{source}', res, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                               'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run_stage():
    t0 = time.perf_counter()
    tag = os.environ.get('TM_PASS_TAG', '')
    seeds = sorted({int(k.split('seed')[1]) for k in host_results()})
    tasks = [(arm, seed, s) for seed in seeds for arm in TM.ARMS for s in SOURCES]
    run = DUR.durable(run_task, X.ROOT / 'runs' / (NAME + tag) / 'done', '{0}-seed{1}-{2}')
    done = []
    with mp.get_context('fork').Pool(min(WORKERS, len(tasks))) as pool_:
        for item in pool_.imap_unordered(run, tasks):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(tasks)}) at {time.perf_counter() - t0:.0f} s', flush=True)
    with gzip.open(f'{PREFIX}{tag}-results.json.gz', 'wt') as f:
        f.write(json.dumps({'spec': str(TM.SPEC.relative_to(X.ROOT)), 'host_stage': TM.NAME, 'selection': json.loads(open(SELECTION).read()),
                            'results': {k: r for k, r, _ in sorted(done, key=lambda d: d[0])}}, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    DUR.atomic_write(f'{PREFIX}{tag}-runtime.json', (json.dumps({'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz', 'runner_sha256': nc.sha256(__file__),
                     'store_sha256': nc.sha256(TS.__file__), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')), 'command': ' '.join([sys.executable, *sys.argv]),
                     'workers': WORKERS, 'device': 'cpu', 'tasks': worker, 'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()), 'wall_seconds': time.perf_counter() - t0,
                     'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}, indent=1, sort_keys=True, default=float) + '\n').encode())


if __name__ == '__main__':
    torch.set_num_threads(1)
    LM.install()
    {'select': select, 'run': run_stage}[STAGE]()
