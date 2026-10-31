"""Translation memory v1, memory stage (docs/translation-memory-v1-draft.md): replays the host stage's saved states
through the phases the host does not take part in, for every memory variant.

At each consolidation point p = 1..6 (the end of batch p's acquisition; the host is frozen from here to the next batch):
  2. consolidation  map refit on the anchors (p = 1 sets the reference instead); every buffer entry not yet written is
                    encoded fresh, mapped, and written once with the host's current answer as its target
  3. gate           each variant's gate trained alone on the buffer entries and their recorded labels
  4. sentences      each variant's sentence rule trained alone on sentences of batch p's words
  5. evaluation     state-preserving (digest of every memory, map, gate and rule before and after)
Point 7 (after the final context-only block) is evaluation only.

Variants: `assoc` (recurrent read at the selected cap), `assoc-1step` (same weights, one-step read), `assoc-nomap`
(recurrent, raw codes everywhere), `flat-sparse` (slot per write keyed by the separated pattern), `flat-store` (slot per
write keyed by the mapped dense code). Each variant has its own gate, router and sentence rule.

Stages (argv[1]):
  select  the pilot selections on seeds 299 (selection) and 298 (confirmation): map regularization, memory grid and step
          cap, router thresholds -> reports/translation-memory-v1-selection.json
  main    the memory pass on the host stage's seeds with the frozen selection -> reports/<NAME>-results.json.gz

Runs on the CPU (the host stage's states are loaded into CPU models). NSY_SMOKE=1 reads the smoke host stage.
Usage: .venv/bin/python -m experiments.translation_memory_pass select
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
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'main'
HOST_NAME = TM.NAME
LAMBDAS = (0.001, 0.01, 0.1, 1.)
GRID = tuple((m, k) for m in (256, 512, 1024) for k in (round(0.02 * m), round(0.04 * m)))
STORAGE_GATE = float(os.environ.get('TM_STORAGE_GATE', '0.95')) if SMOKE else 0.95  # the override exists only for smoke tests
CAP_TOLERANCE = 0.01
FOLDS = 4
GATE_STEPS, GATE_LR = 300, 0.05
SENT_TRAIN, SENT_TEST, SENT_STEPS, SENT_LR = 200, 100, 200, 0.1
CORRUPTION = (0., .2, .4, .6)
FREE_STARTS = 200
STABLE = os.environ.get('TM_STABLE_KEYS') == '1'  # keys from the separate frozen encoder: no anchors, no map, no centring
VARIANTS = ('assoc', 'assoc-1step', 'flat-sparse', 'flat-store') if STABLE else ('assoc', 'assoc-1step', 'assoc-nomap', 'flat-sparse', 'flat-store')
VERSION = '-stable-keys' if STABLE else ''
SELECTION = X.ROOT / 'reports' / ('translation-memory-v1-selection' + VERSION + ('-smoke' if SMOKE else '') + '.json')
NAME = 'translation-memory-v1-memory' + VERSION + ('-smoke' if SMOKE else '') + os.environ.get('TM_PASS_TAG', '')
WORKERS = X.WORKERS
D_CODE = TM.KEY_DIM if TM.IDENTITY else 16  # key width: the identity key head, or the host's code (inner width of the wide standard network)


class FixedFrame:
    """The stable-key branch's coordinates: keys are used as they are."""

    def __init__(self, d):
        self.mu, self.s2, self.A = torch.zeros(d), 1., torch.zeros(1)

    def __call__(self, x):
        return x.float()


def stable_key_encoder(seed):
    """The frozen key encoder for a seed, at the setting selected on validation symbols."""
    c = json.loads(open(f'{TK.REPORT}-selection.json').read())['chosen']
    fit = torch.load(TK.RUNS / 'fits' / f"{c['arch']}-key{c['key']}-seed{seed}.pt", weights_only=False)
    model = TK.build(c['arch'], c['key'], seed)
    model.load_state_dict(fit['checkpoints'][c['steps']]['state'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, c['key']


# ---------------------------------------------------------------- host access

HOST = {}


def host_results():
    if not HOST:
        with gzip.open(X.ROOT / 'reports' / f'{HOST_NAME}-results.json.gz', 'rt') as f:
            HOST.update(json.load(f)['results'])
    return HOST


class Host:
    """The saved host states of one arm, seed and branch, loaded on demand into a CPU model."""

    def __init__(self, arm, seed, branch):
        rec = host_results()[f'{arm}/seed{seed}']
        self.arm, self.seed, self.branch = arm, seed, branch
        self.points = rec['branches'][branch]['points']
        self.events = rec['branches'][branch]['events']
        self.exposures = rec['branches'][branch]['exposures']
        self.last_presented = rec['branches'][branch]['last_presented']
        self.states = {p['point']: p for p in torch.load(X.ROOT / rec['checkpoint']['file'], weights_only=False)[branch]}
        self.mat = TM.material(seed)

    def load(self, point):
        LM.CURRENT.update(arm=self.arm, seed=self.seed, step=0, r=None)
        torch.manual_seed(self.seed)
        enc, model = X.make_model(self.seed)
        s = self.states[point]['host']
        model.load_state_dict(s['sri']); enc['image'].load_state_dict(s['image']); enc['sequence'].load_state_dict(s['sequence'])
        enc['image'].eval()
        return enc, model


@torch.no_grad()
def encode(enc, model, x):
    return G.codes_fn(enc, model)(np.ascontiguousarray(x.reshape(-1, 28, 28), dtype=np.float32), 'image').double()


@torch.no_grad()
def encode_keys(enc, model, x):
    """The memory's key representation: the identity key head when the host stage trained one, else the host's code."""
    if not TM.IDENTITY:
        return encode(enc, model, x)
    return TM.keys_of(enc, torch.from_numpy(np.ascontiguousarray(x.reshape(-1, 28, 28), dtype=np.float32))).double()


@torch.no_grad()
def logits_against_targets(model, q, tcodes):
    return G.pair(model, q.float(), tcodes.float()[None].expand(len(q), -1, -1), 1)


# ---------------------------------------------------------------- one replay

class Variant:
    """One memory variant with its own gate, sentence rule and usage bookkeeping."""

    def __init__(self, name, mem, sep, cb, cap, mapped):
        self.name, self.mem, self.sep, self.cb, self.cap, self.mapped = name, mem, sep, cb, cap, mapped
        self.gate, self.rule = TS.Gate(), TS.SentenceRule()

    def keyspace(self, x, ctx):
        """Codes (n, d) -> (sparse pattern, dense key) in this variant's coordinates."""
        xr = ctx['map'](x) if self.mapped else x.float()
        z = (xr.float() - ctx['map'].mu.float()) / float(np.sqrt(ctx['map'].s2))
        return self.sep(xr, ctx['map'].mu, ctx['map'].s2), z

    def write(self, x, targets, ctx):
        s, z = self.keyspace(x, ctx)
        if self.name in ('assoc', 'assoc-nomap'):
            self.mem.write(s, self.cb[targets])
        elif self.name == 'flat-sparse':
            self.mem.write(s, targets.tolist())
        elif self.name == 'flat-store':
            self.mem.write(z, targets.tolist())

    def read(self, x, ctx, training=False):
        """-> scores (n, T), reinstated target part (n, m_t), converged (n,), info."""
        s, z = self.keyspace(x, ctx)
        n = len(x)
        if self.name.startswith('assoc'):
            if self.name == 'assoc-1step':
                tp = self.mem.one_step(s)
                return TS.target_scores(tp, self.cb), tp, torch.ones(n), {}
            out, info = self.mem.retrieve(s)
            tp = out[self.cap]
            info['checkpoint_scores'] = {t: TS.target_scores(v, self.cb) for t, v in out.items()}
            conv = (info['converged'] & (info['steps'] <= self.cap)).float()
            return TS.target_scores(tp, self.cb), tp, conv, info
        sc, win = self.mem.read(s if self.name == 'flat-sparse' else z)
        if training:
            self.mem.count_use(win)
        tp = self.cb[sc.argmax(1)] if self.mem.keys else torch.zeros(n, self.cb.shape[1])
        return sc, tp, torch.ones(n), {}

    def digest(self):
        m = self.mem
        if hasattr(m, 'W'):
            parts = [m.W]
        else:
            parts = [torch.stack(m.keys) if m.keys else torch.zeros(1), torch.tensor(m.values + m.used)]
        return TS.digest(*parts, *[p for p in self.gate.parameters()], *[p for p in self.rule.parameters()])


def build_variants(seed, cfg, cap, names, d=D_CODE):
    m, k = cfg
    sep = TS.Separation(d, m, k, nc.stream('tm_separation', seed, m, k))
    cb = TS.codebook(TM.N_TGT, m, k, nc.stream('tm_codebook', seed, m, k))
    out = {}
    shared = None
    for name in names:
        if name in ('assoc', 'assoc-1step'):
            shared = shared or TS.Associative(m, m, k, k, nc.stream('tm_assoc', seed, m, k))
            mem = shared
        elif name == 'assoc-nomap':
            mem = TS.Associative(m, m, k, k, nc.stream('tm_assoc', seed, m, k))
        elif name == 'flat-sparse':
            mem = TS.FlatSparse(TM.N_WORDS, TM.N_TGT, k)
        else:
            mem = TS.FlatStore(TM.N_WORDS, TM.N_TGT)
        out[name] = Variant(name, mem, sep, cb, cap, name != 'assoc-nomap')
    return out


def sentences(rng, words, n, exclude=()):
    """n ordered triples of distinct words, none equal to a triple in `exclude`."""
    out, seen = [], set(map(tuple, exclude))
    while len(out) < n:
        t = tuple(int(w) for w in rng.choice(words, 3, replace=False))
        if t not in seen:
            out.append(t)
    return np.array(out)


def replay(arm, seed, branch, cfg, lam, cap, names, full=True, thresholds=None):
    """The memory phases over the saved host trajectory. `full` adds gates, sentences and the final-point extras."""
    host = Host(arm, seed, branch)
    mat = host.mat
    labels = torch.tensor(mat['tgt'])
    key_model, key_digest, width = None, None, D_CODE
    if STABLE:
        key_model, width = stable_key_encoder(seed)
        key_digest = TS.digest(*key_model.parameters())
        ctx = {'map': FixedFrame(width)}
    else:
        ctx = {'map': TS.ReferenceMap(D_CODE, lam)}
    V = build_variants(seed, cfg, cap, names, width)
    written, teacher, write_point = {}, {}, {}
    ref_code_at_write, exact = {}, {}
    sym = np.repeat(np.arange(TM.ANCHOR_SYMBOLS), TM.ANCHOR_RENDERS)
    train_sents = []
    srng = nc.stream('tm_sentence_draws', seed)
    out = {'points': []}
    for point in range(1, TM.BATCHES + 2):
        enc, model = host.load(point)
        pinfo = host.points[point - 1]
        rec = {'point': point, 'step': pinfo['step']}
        keyfn = (lambda x: TK.encode(key_model, x).double()) if STABLE else (lambda x, e=enc, m=model: encode_keys(e, m, x))
        ctx['keyfn'] = keyfn
        tcodes = encode(enc, model, mat['tgt_train'][:, 0])
        E = encode(enc, model, mat['src_eval'])  # the host's codes, for its answers (120 * R_EVAL, d)
        Ek = keyfn(mat['src_eval'])  # the memory's keys for the same renderings
        p_host_E = torch.softmax(logits_against_targets(model, E, tcodes), 1)
        consolidating = point <= TM.BATCHES
        if STABLE:
            rec['key_encoder_unchanged'] = TS.digest(*key_model.parameters()) == key_digest
            rec['eval_keys_digest'] = TS.digest(Ek)
        else:
            A_now = keyfn(mat['anchor_imgs'])
            if point == 1:
                ctx['map'].set_reference(A_now)
            elif consolidating:
                ctx['map'].fit(A_now)
            rec['anchor_residual'] = ctx['map'].residual(A_now)
        buf = pinfo['buffer']
        bimgs = np.stack([mat['src_train'][e['word'], e['render']] for e in buf]) if buf else None
        bx, bk = (encode(enc, model, bimgs), keyfn(bimgs)) if buf else (None, None)
        bwords = torch.tensor([e['word'] for e in buf])
        if consolidating:
            # 2. consolidation: each unwritten entry once, teacher = the host's current answer
            new = [i for i, e in enumerate(buf) if e['word'] not in written]
            if new:
                xn = bk[new]
                ans = logits_against_targets(model, bx[new], tcodes).argmax(1)
                for i, a in zip(new, ans.tolist()):
                    w = buf[i]['word']
                    written[w], teacher[w], write_point[w] = True, a, point
                for v in V.values():
                    if v.name != 'assoc-1step':
                        v.write(xn, ans, ctx)
                ws = [buf[i]['word'] for i in new]
                if 'assoc' in V:  # measurement copy of the exact written cues (not used by the system)
                    s_new, _ = V['assoc'].keyspace(xn, ctx)
                    for w, sp, a in zip(ws, s_new, ans.tolist()):
                        exact[w] = (sp, a)
                rec['written'] = ws
                rec['teacher_acc'] = float((ans == labels[ws]).float().mean())
                for w in ws:  # measurement only: the word's mean evaluation code in reference coordinates at write
                    ref_code_at_write[w] = ctx['map'](Ek.reshape(TM.N_WORDS, TM.R_EVAL, -1)[w]).mean(0)
                # immediate accuracy: exact written cues (recovery of the written target) and evaluation renderings
                rec['immediate'] = {}
                Ew = Ek.reshape(TM.N_WORDS, TM.R_EVAL, -1)[ws].reshape(-1, Ek.shape[1])
                for v in V.values():
                    sc_exact, _, _, _ = v.read(xn, ctx)
                    sc_ev, _, _, _ = v.read(Ew, ctx)
                    rec['immediate'][v.name] = {'exact_cue_recovery': float((sc_exact.argmax(1) == ans).float().mean()),
                                                'eval_acc': float((sc_ev.argmax(1) == labels[ws].repeat_interleave(TM.R_EVAL)).float().mean())}
            if full:
                # 3. gate: on every buffer entry with its recorded label
                ph = torch.softmax(logits_against_targets(model, bx, tcodes), 1)
                rec['gate_loss'] = {}
                for v in V.values():
                    sc, _, conv, _ = v.read(bk, ctx, training=True)
                    rec['gate_loss'][v.name] = v.gate.fit(TS.features(ph, sc, conv), TS.p_memory(sc), ph, labels[bwords], GATE_STEPS, GATE_LR)
                # 4. sentences of this batch's words (training renderings), read through the memory just written
                bw = TM.batch_words(point - 1)
                tr = sentences(srng, bw, SENT_TRAIN)
                train_sents.extend(tr.tolist())
                rj = srng.integers(TM.R_TRAIN, size=tr.shape)
                x = encode(enc, model, mat['src_train'][tr, rj])
                xk = keyfn(mat['src_train'][tr, rj])
                ph = torch.softmax(logits_against_targets(model, x, tcodes), 1)
                rec['sentence_loss'] = {}
                for v in V.values():
                    sc, tp, conv, _ = v.read(xk, ctx, training=True)
                    with torch.no_grad():
                        w = v.gate(TS.features(ph, sc, conv))
                    r = TS.routed_patterns(w, tp, ph, v.cb).reshape(len(tr), 3, -1)
                    rec['sentence_loss'][v.name] = v.rule.fit(r, labels[torch.from_numpy(tr)].flip(1), v.cb, SENT_STEPS, SENT_LR)
        # 5. evaluation (state-preserving)
        before = {k: v.digest() for k, v in V.items()}
        map_before = TS.digest(ctx['map'].A)
        rec['eval'] = evaluate(V, ctx, Ek, p_host_E, labels, written, point, full, thresholds, enc, model, mat, tcodes,
                               train_sents, nc.stream('tm_sentence_test', seed, point))
        if 'assoc' in V and exact:  # recovery of the written target from every exact cue so far, per checkpoint
            ws_all = sorted(exact)
            res_x, _ = V['assoc'].mem.retrieve(torch.stack([exact[w][0] for w in ws_all]))
            tg = torch.tensor([exact[w][1] for w in ws_all])
            rec['exact_all'] = {str(t): float((TS.target_scores(tp, V['assoc'].cb).argmax(1) == tg).float().mean()) for t, tp in res_x.items()}
        rec['state_preserved'] = all(V[k].digest() == before[k] for k in V) and TS.digest(ctx['map'].A) == map_before
        Er = Ek.reshape(TM.N_WORDS, TM.R_EVAL, -1)
        rec['drift'] = {str(w): float((ctx['map'](Er[w]).mean(0) - ref_code_at_write[w]).norm() / np.sqrt(ctx['map'].s2)) for w in written}
        if full and point == TM.BATCHES + 1:
            rec['extras'] = extras(V, exact, seed)
        out['points'].append(rec)
    out['teacher'] = {str(w): a for w, a in teacher.items()}
    out['write_point'] = {str(w): p for w, p in write_point.items()}
    out['bytes'] = {k: float(v.mem.nbytes()) for k, v in V.items()}
    return out


def evaluate(V, ctx, E, p_host, labels, written, point, full, thresholds, enc, model, mat, tcodes, train_sents, rng):
    """Per batch of written words: host, memory-alone (and checkpoints), gate and router accuracy; sentences of old words."""
    R = TM.R_EVAL
    words = sorted(written)
    if not words:
        return {}
    idx = torch.tensor([w * R + r for w in words for r in range(R)])
    y = labels.repeat_interleave(R)[idx]
    batch_of = torch.tensor([w // TM.WORDS for w in words]).repeat_interleave(R)
    ph = p_host[idx]
    host_ok = ph.argmax(1) == y
    per_batch = lambda ok: {str(b + 1): float(ok[batch_of == b].float().mean()) for b in sorted(set(batch_of.tolist()))}
    res = {'host': per_batch(host_ok), 'variants': {}}
    for v in V.values():
        sc, tp, conv, info = v.read(E[idx], ctx)
        r = {'memory': per_batch(sc.argmax(1) == y)}
        if 'checkpoint_scores' in info:
            r['checkpoints'] = {str(t): per_batch(s.argmax(1) == y) for t, s in info['checkpoint_scores'].items()}
            r['converged'] = per_batch(info['converged'])
            r['cycle'] = per_batch(info['cycle'])
            r['steps_mean'] = float(info['steps'].float().mean())
            r['active_connections_mean'] = float(info['active_connections'].mean())
            r['dense_ops_per_read'] = int(info['dense_ops'])
        r['router'] = {str(th): per_batch(TS.route(th, ph, sc) == y) for th in TS.ROUTER_THRESHOLDS}
        if full:
            f = TS.features(ph, sc, conv)
            ans, w = v.gate.answer(f, TS.p_memory(sc), ph)
            r['gate'] = per_batch(ans == y)
            r['gate_weight'] = per_batch(w)
            if thresholds is not None:
                r['router_selected'] = r['router'][str(thresholds[v.name])]
            old = [w_ for w_ in words if w_ // TM.WORDS < point - 1] if point <= TM.BATCHES else words
            if len(old) >= 3:
                r['sentences'] = sentence_test(v, ctx, old, enc, model, mat, tcodes, labels, train_sents, rng)
        res['variants'][v.name] = r
    return res


@torch.no_grad()
def sentence_test(v, ctx, old, enc, model, mat, tcodes, labels, train_sents, rng):
    """Unseen combinations of old words, evaluation renderings: word accuracy by position, whole-sentence accuracy, and the
    single-word (gate) accuracy on the same word tokens."""
    te = sentences(rng, np.array(old), SENT_TEST, exclude=train_sents)
    rj = rng.integers(TM.R_EVAL, size=te.shape)
    x = encode(enc, model, mat['src_eval'][te, rj])
    ph = torch.softmax(logits_against_targets(model, x, tcodes), 1)
    sc, tp, conv, _ = v.read(ctx['keyfn'](mat['src_eval'][te, rj]), ctx)
    f = TS.features(ph, sc, conv)
    single, w = v.gate.answer(f, TS.p_memory(sc), ph)
    r = TS.routed_patterns(w, tp, ph, v.cb).reshape(len(te), 3, -1)
    pred = v.rule.logits(r, v.cb).argmax(2)
    truth = labels[torch.from_numpy(te)].flip(1)
    ok = pred == truth
    return {'position': [float(ok[:, j].float().mean()) for j in range(3)], 'whole': float(ok.all(1).float().mean()),
            'single_word': float((single == labels[torch.from_numpy(te)].reshape(-1)).float().mean()),
            'mixing': torch.softmax(v.rule.theta, 1).tolist()}


@torch.no_grad()
def extras(V, exact, seed):
    """Final point: corrupted exact cues (a fraction of the written pattern's active units moved to random inactive units;
    recovery of the written target) and free recall from random sparse starts."""
    out = {}
    if not exact:
        return out
    words = sorted(exact)
    S = torch.stack([exact[w][0] for w in words])
    tg = torch.tensor([exact[w][1] for w in words])
    m, k = S.shape[1], int(S[0].sum())
    rng = nc.stream('tm_extras', seed)
    for frac in CORRUPTION:
        noisy = S.clone()
        j = int(round(frac * k))
        for i in range(len(noisy)):
            on, off = noisy[i].nonzero()[:, 0], (noisy[i] == 0).nonzero()[:, 0]
            noisy[i, on[torch.from_numpy(rng.choice(len(on), j, replace=False))]] = 0
            noisy[i, off[torch.from_numpy(rng.choice(len(off), j, replace=False))]] = 1
        r = {}
        if 'assoc' in V:
            res, info = V['assoc'].mem.retrieve(noisy)
            r['assoc'] = {str(t): float((TS.target_scores(tp, V['assoc'].cb).argmax(1) == tg).float().mean()) for t, tp in res.items()}
            r['assoc_converged'] = float(info['converged'].float().mean())
        if 'flat-sparse' in V:
            sc, _ = V['flat-sparse'].mem.read(noisy)
            r['flat-sparse'] = float((sc.argmax(1) == tg).float().mean())
        out[f'corrupt_{frac}'] = r
    if 'assoc' in V:
        v = V['assoc']
        starts = TS.codebook(FREE_STARTS, m, k, rng)
        res, info = v.mem.retrieve(starts)
        sc = TS.target_scores(res[max(TS.CHECKPOINTS)], v.cb)
        best = sc.max(1)
        hit = (best.values >= 0.8) & torch.isin(best.indices, tg)
        out['free_recall'] = {'starts': FREE_STARTS, 'distinct_stored_regenerated': int(best.indices[hit].unique().numel()),
                              'stored_pairs': len(words), 'spurious_fraction': float((~hit).float().mean()),
                              'converged': float(info['converged'].float().mean())}
    return out


# ---------------------------------------------------------------- stages

def pooled(per_batch):
    return float(np.mean(list(per_batch.values())))


def select():
    """Pilot selections (proposed settings; docs/translation-memory-v1-draft.md, "Pilot and selection")."""
    t0 = time.perf_counter()
    arms = TM.ARMS
    sel_seed, conf_seed = (299, 298)
    timing = {}
    # 1. map regularization: held-out anchor error, symbol-level folds, consolidation points 2-6
    t = time.perf_counter()
    errs = {} if STABLE else {str(l): [] for l in LAMBDAS}
    sym = np.repeat(np.arange(TM.ANCHOR_SYMBOLS), TM.ANCHOR_RENDERS)
    for seed in () if STABLE else (sel_seed, conf_seed):
        perm = nc.stream('tm_folds', seed).permutation(TM.ANCHOR_SYMBOLS)
        folds = [set(perm[i::FOLDS].tolist()) for i in range(FOLDS)]
        for arm in arms:
            host = Host(arm, seed, 'main')
            ref = None
            for point in range(1, TM.BATCHES + 1):
                enc, model = host.load(point)
                A = encode(enc, model, host.mat['anchor_imgs'])
                if point == 1:
                    ref, s2 = A, float(A.pow(2).mean())
                    continue
                for l in LAMBDAS:
                    e = TS.heldout_anchor_error(A, ref, sym, l, s2, folds)
                    errs[str(l)].append({'seed': seed, 'arm': arm, 'point': point, 'error': e})
    mean_err = {l: float(np.mean([r['error'] for r in v if r['seed'] == sel_seed])) for l, v in errs.items()}
    lam = float(min(mean_err, key=mean_err.get)) if mean_err else None
    confirm_err = {l: float(np.mean([r['error'] for r in v if r['seed'] == conf_seed])) for l, v in errs.items()}
    timing['lambda_seconds'] = time.perf_counter() - t
    # 2. memory grid on the selection seed: storage gate on exact cues after all writes (16-step read), then the cap
    t = time.perf_counter()
    grid = {}
    with mp.get_context('fork').Pool(min(WORKERS, len(GRID) * len(arms))) as pool:
        jobs = [(arm, sel_seed, 'main', cfg, lam, 16, ('assoc',), False, None) for cfg in GRID for arm in arms]
        for (arm, _, _, cfg, *_), res in zip(jobs, pool.starmap(replay, jobs)):
            grid.setdefault(str(cfg), {})[arm] = summarize_grid(res)
    timing['grid_seconds'] = time.perf_counter() - t
    passing = [cfg for cfg in GRID if np.mean([grid[str(cfg)][a]['exact_recovery_16'] for a in arms]) >= STORAGE_GATE]
    chosen = passing[0] if passing else None
    cap = None
    if chosen is not None:
        cap = choose_cap([grid[str(chosen)][a]['eval_checkpoints'] for a in arms])
    # 3. confirmation on a separate pilot seed
    t = time.perf_counter()
    confirm = None
    if chosen is not None:
        with mp.get_context('fork').Pool(min(WORKERS, len(arms))) as pool:
            jobs = [(arm, conf_seed, 'main', chosen, lam, 16, ('assoc',), False, None) for arm in arms]
            cres = {arm: summarize_grid(r) for arm, r in zip(arms, pool.starmap(replay, jobs))}
        confirm = {'per_arm': cres, 'passes_storage_gate': bool(np.mean([cres[a]['exact_recovery_16'] for a in arms]) >= STORAGE_GATE),
                   'cap_rule': choose_cap([cres[a]['eval_checkpoints'] for a in arms])}
    timing['confirm_seconds'] = time.perf_counter() - t
    # 4. router thresholds on pilot batches 2-3 (balanced current and previous words), each retrieval method separately
    t = time.perf_counter()
    thresholds, router_scores = None, None
    if chosen is not None:
        with mp.get_context('fork').Pool(min(WORKERS, len(arms))) as pool:
            jobs = [(arm, sel_seed, 'main', chosen, lam, cap, VARIANTS, False, None) for arm in arms]
            rres = pool.starmap(replay, jobs)
        router_scores = {v: {} for v in VARIANTS}
        for v in VARIANTS:
            for th in TS.ROUTER_THRESHOLDS:
                vals = []
                for r in rres:
                    for p in r['points'][1:3]:
                        acc = p['eval']['variants'][v]['router'][str(th)]
                        cur = acc[str(p['point'])]
                        prev = np.mean([a for b, a in acc.items() if int(b) < p['point']])
                        vals.append((cur + prev) / 2)
                router_scores[v][str(th)] = float(np.mean(vals))
        thresholds = {v: max(TS.ROUTER_THRESHOLDS, key=lambda th: (router_scores[v][str(th)], -TS.ROUTER_THRESHOLDS.index(th))) for v in VARIANTS}
    timing['router_seconds'] = time.perf_counter() - t
    out = {'status': 'pilot selection (proposed settings, not evidence that the configuration works)',
           'seeds': {'selection': sel_seed, 'confirmation': conf_seed}, 'arms': arms,
           'lambda': {'grid': LAMBDAS, 'mean_heldout_error_selection_seed': mean_err, 'mean_heldout_error_confirmation_seed': confirm_err,
                      'chosen': lam, 'per_point': errs},
           'memory_grid': {'storage_gate': STORAGE_GATE, 'results': grid, 'passing': [list(c) for c in passing],
                           'chosen': list(chosen) if chosen else None, 'cap': cap, 'cap_tolerance': CAP_TOLERANCE},
           'confirmation': confirm, 'router': {'scores': router_scores, 'chosen': thresholds}}
    SELECTION.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    rt = {'scope': 'nondeterministic timing of the pilot selection', 'wall_seconds': time.perf_counter() - t0, **timing,
          'workers': WORKERS, 'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
          'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain'))}
    SELECTION.with_name(SELECTION.stem + '-runtime.json').write_text(json.dumps(rt, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'lambda': lam, 'passing': passing, 'chosen': chosen, 'cap': cap, 'thresholds': thresholds,
                      'confirmation': confirm and {k: confirm[k] for k in ('passes_storage_gate', 'cap_rule')}}, default=str))


def summarize_grid(res):
    """Storage gate and cap inputs from a replay: exact-cue recovery after all writes (last consolidation point) and
    memory-alone recall on independent renderings per checkpoint at that point, pooled over batches."""
    last = res['points'][TM.BATCHES - 1]
    ev = last['eval']['variants']['assoc']
    return {'exact_recovery': last['exact_all'], 'exact_recovery_16': last['exact_all']['16'],
            'eval_checkpoints': {t: pooled(v) for t, v in ev['checkpoints'].items()},
            'converged': pooled(ev['converged']), 'cycle': pooled(ev['cycle']), 'steps_mean': ev['steps_mean'],
            'teacher_acc': [p.get('teacher_acc') for p in res['points'][:TM.BATCHES]], 'bytes': res['bytes']['assoc']}


def choose_cap(per_arm_checkpoints):
    acc = {int(t): float(np.mean([c[t] for c in per_arm_checkpoints])) for t in per_arm_checkpoints[0]}
    top = acc[max(acc)]
    return min(t for t, a in acc.items() if a >= top - CAP_TOLERANCE)


def run_main_task(task):
    arm, seed, branch = task
    t0, c0 = time.perf_counter(), time.process_time()
    sel = json.loads(SELECTION.read_text())
    cfg, cap = tuple(sel['memory_grid']['chosen']), sel['memory_grid']['cap']
    th = {k: (v if v == 'host' else float(v)) for k, v in sel['router']['chosen'].items()}
    res = replay(arm, seed, branch, cfg, sel['lambda']['chosen'], cap, VARIANTS, True, th)
    res['config'] = {'memory': cfg, 'cap': cap, 'lambda': sel['lambda']['chosen'], 'thresholds': th}
    return f'{arm}/seed{seed}/{branch}', res, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                             'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main_stage():
    t0 = time.perf_counter()
    seeds = sorted({int(k.split('seed')[1]) for k in host_results()})
    tasks = [(arm, seed, branch) for seed in seeds for arm in TM.ARMS for branch in TM.BRANCHES]
    run = DUR.durable(run_main_task, X.ROOT / 'runs' / NAME / 'done', '{0}-seed{1}-{2}')
    done = []
    with mp.get_context('fork').Pool(min(WORKERS, len(tasks))) as pool:
        for item in pool.imap_unordered(run, tasks):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(tasks)}) at {time.perf_counter() - t0:.0f} s', flush=True)
    prefix = X.ROOT / 'reports' / NAME
    res = {'spec': str(TM.SPEC.relative_to(X.ROOT)), 'host_stage': HOST_NAME, 'selection': json.loads(SELECTION.read_text()),
           'results': {k: r for k, r, _ in sorted(done, key=lambda d: d[0])}}
    with gzip.open(f'{prefix}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    (X.ROOT / 'reports' / f'{NAME}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'runner_sha256': nc.sha256(__file__), 'store_sha256': nc.sha256(TS.__file__),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'workers': WORKERS, 'device': 'cpu', 'tasks': worker,
        'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()), 'wall_seconds': time.perf_counter() - t0,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}, indent=1, sort_keys=True, default=float) + '\n')


def setup():
    torch.set_num_threads(1)
    LM.install()
    TM.install_key_head()


if __name__ == '__main__':
    setup()
    {'select': select, 'main': main_stage}[STAGE]()
