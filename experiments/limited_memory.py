"""Limited, decaying memory as pressure to consolidate (docs/limited-memory-proposal.md).

First reader layer of g and P: 48 structure units (RigL, one code dimension per operand) and 16 memory units (dense;
incoming weights and biases decay toward their start, outgoing weights toward zero), or the arms' variants. Blocks 1-3
use the D-varied curriculum without rehearsal; a delay of context episodes follows, then relearning of English with
rehearsal draws from the buffer (the buffer is sampled only in retraining). Checkpoints, four-way silencing, the
single-coordinate probe and the reader-disruption test are recorded at the end of each block, after the delay and after
relearning.

Stages (argv[2]):
  main     8 arms x seeds 300-309 (the selection run)
  fork-confirm  the same fork from the confirm stage's `mixed` parents, seeds 310-319
  fork     decay against no decay from the end-of-block-3 checkpoint of `mixed`, seeds 300-309
  confirm  sparse48, mixed, mixed-reset and NLM_SELECTED, seeds 310-319

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.limited_memory pilot main
       (rerun the same command to resume after an interruption: experiments/durable.py)
"""
import contextlib
import copy
import io
import os
import resource
import sys
import time

import numpy as np
import torch
from torch import nn

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import generic_relations as G
from experiments import transition_pilot as TP
from experiments import sequence_pilot as SP
from experiments import fan_in_pilot as FI
from experiments import fan_in_pilot_analysis as AN
from experiments import fan_in_count_coordinate as FCC
from experiments import device as DEV
from experiments import durable as DUR

nc, ns = X.nc, X.ns
STAGE = sys.argv[2] if len(sys.argv) > 2 else 'main'
SMOKE = X.SMOKE
STRUCT, MEM, CODE = 48, 16, FI.CODE
ARMS = ('dense', 'sparse64', 'sparse48', 'mixed', 'mixed-h2000', 'mixed-h500', 'mixed-h125', 'mixed-reset')
HALF_LIFE = {'mixed-h2000': 2000, 'mixed-h500': 500, 'mixed-h125': 125}
FORK = {'fork-hinf': None, 'fork-h2000': 2000, 'fork-h500': 500, 'fork-h125': 125}
SEEDS = {'main': tuple(range(300, 310)), 'fork': tuple(range(300, 310)), 'confirm': tuple(range(310, 320)),
         'fork-confirm': tuple(range(310, 320))}[STAGE]
DELAY, RELEARN, FORK_STEPS, EVERY = (20, 8, 20, 10) if SMOKE else (2000, 200, 2000, 500)
DISRUPT_P, DISRUPT_DRAWS = .25, 10
NAME = 'limited-memory-v1-' + STAGE + ('-smoke' if SMOKE else '') + os.environ.get('NFI_TAG', '')
FORK_LIKE = STAGE in ('fork', 'fork-confirm')
PARENT_NAME = 'limited-memory-v1-' + ('confirm' if STAGE == 'fork-confirm' else 'main') + ('-smoke' if SMOKE else '') + os.environ.get('NFI_TAG', '')
RUNS = X.ROOT / 'runs' / NAME
CURRENT = {}
FLAGS = {'full': (), 'mem_silenced': ('mem',), 'struct_silenced': ('struct',), 'both_silenced': ('mem', 'struct')}


def arms_for_stage():
    if STAGE == 'main':
        return ARMS
    if FORK_LIKE:
        return tuple(FORK)
    sel = os.environ.get('NLM_SELECTED')
    if sel not in HALF_LIFE:
        raise SystemExit('confirm needs NLM_SELECTED set to one of ' + ', '.join(HALF_LIFE))
    return ('sparse48', 'mixed', 'mixed-reset', sel)


# ---------------------------------------------------------------- model

class SplitLinear(nn.Module):
    """First reader layer as 48 structure units (a RigL-masked layer) and 16 memory units (dense), concatenated in that
    order; mem is None for sparse48, whose 16 missing units output zero. `silence` holds 'struct' and/or 'mem'."""

    def __init__(self, struct, mem):
        super().__init__()
        self.struct, self.mem, self.silence = struct, mem, set()

    def forward(self, x):
        s = x.new_zeros(len(x), STRUCT) if 'struct' in self.silence else self.struct(x)
        m = x.new_zeros(len(x), MEM) if self.mem is None or 'mem' in self.silence else self.mem(x)
        return torch.cat([s, m], 1)


def readers(model):
    return (('g', model.g), ('P', model.P))


def make_model(seed, arm):
    enc, model = SP.make_model(seed)  # same draws as the sequence pilot: every arm starts from the same weights
    for name, seq in readers(model):
        base = seq[0]
        ra, rb = nc.stream('lm_mask', seed, name, 'a'), nc.stream('lm_mask', seed, name, 'b')
        if arm == 'dense':
            continue
        if arm == 'sparse64':
            FI.reader_mask(base, 16, ra, rb)
            base.register_buffer('swaps', torch.zeros((), dtype=torch.long))
            continue
        tmp = copy.deepcopy(base)
        FI.reader_mask(tmp, 16, ra, rb)  # 64 rows drawn as in sparse64; the first 48 are the structure units
        n_in = base.weight.shape[1]
        struct = nn.Linear(n_in, STRUCT)
        struct.weight.data, struct.bias.data = tmp.weight.data[:STRUCT].clone(), tmp.bias.data[:STRUCT].clone()
        struct.__class__ = FI.MaskedLinear
        struct.register_buffer('mask', tmp.mask[:STRUCT].clone())
        struct.register_buffer('swaps', torch.zeros((), dtype=torch.long))
        mem = None
        if arm != 'sparse48':
            mem = nn.Linear(n_in, MEM)
            mem.weight.data, mem.bias.data = base.weight.data[STRUCT:].clone(), base.bias.data[STRUCT:].clone()  # standard init, no rescaling
            mem.__class__ = FI.MaskedLinear
            mem.register_buffer('mask', torch.ones_like(mem.weight))
            mem.register_buffer('swaps', torch.zeros((), dtype=torch.long))
            mem.register_buffer('w0', mem.weight.data.clone())
            mem.register_buffer('b0', mem.bias.data.clone())
        seq[0] = SplitLinear(struct, mem)
    return enc, model


def first_layers(model):
    return {f'{n}/first': seq[0] for n, seq in readers(model)}


def structure_layers(model):
    """Layers RigL acts on: the 48 structure rows, or all 64 rows of sparse64."""
    return {k: (v.struct if isinstance(v, SplitLinear) else v) for k, v in first_layers(model).items()
            if isinstance(v, SplitLinear) or hasattr(v, 'mask')}


def memory_layers(model):
    return {k: v.mem for k, v in first_layers(model).items() if isinstance(v, SplitLinear) and v.mem is not None}


@torch.no_grad()
def decay_step(model, r):
    """w <- w0 + (w - w0) r for the memory units' incoming weights and biases; outgoing columns <- r * w."""
    for name, seq in readers(model):
        mem = seq[0].mem
        mem.weight.copy_(mem.w0 + (mem.weight - mem.w0) * r)
        mem.bias.copy_(mem.b0 + (mem.bias - mem.b0) * r)
        seq[2].weight[:, STRUCT:] *= r


def train_update(enc, model, opt, material, rng, compare):
    """TP.train_update, then RigL on the structure rows, the memory units' reset, and decay (arm-specific)."""
    arm = CURRENT['arm']
    CURRENT['step'] += 1
    step, end = CURRENT['step'], FI.END_FRACTION * 3 * D.STEPS
    rigl = structure_layers(model) if arm not in ('dense',) and step % FI.DT == 0 and step <= end else {}
    for lin in rigl.values():
        lin.collect, lin.probe = True, torch.zeros_like(lin.weight, requires_grad=True)
    out = TP.train_update(enc, model, opt, material, rng, compare)
    alpha = FI.ALPHA * (1 + np.cos(np.pi * step / end)) / 2
    for name, lin in rigl.items():
        lin.collect = False
        FI.rigl_step(lin, FI.rigl_groups(name, lin), opt, alpha, 'zero', None, None)
        del lin.probe
    if arm == 'mixed-reset' and step % FI.DT == 0 and step <= end:
        for name, lin in memory_layers(model).items():
            FI.reset_step(lin, opt, nc.stream('lm_reset', CURRENT['seed'], name, step), FI.REWIRE_P[name])
    if CURRENT.get('r') is not None:
        decay_step(model, CURRENT['r'])
    return out


# ---------------------------------------------------------------- measures

@contextlib.contextmanager
def silenced(model, parts):
    layers = [v for v in first_layers(model).values() if isinstance(v, SplitLinear)]
    for L in layers:
        L.silence = set(parts)
    try:
        yield
    finally:
        for L in layers:
            L.silence = set()


@torch.no_grad()
def matching(enc, model, withheld):
    """Isolated matching per script (as generic_extras 'iso'): accuracy over all, even-indexed and odd-indexed evaluation
    items, for paired and withheld counts."""
    enc['image'].eval(); enc['sequence'].eval()
    code, E = G.codes_fn(enc, model), P.EVAL_PER
    hits = {(lang, grp): [] for lang in P.LANGS for grp in ('paired', 'withheld')}
    for lang in P.LANGS:
        for n in ns.NUMBERS:
            q = code(X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]], 'image')
            cc = code(np.stack([P.EVAL[(lang, n, i)][0] for i in range(E)]).reshape(-1, 28, 28), 'image').reshape(E, 9, -1)
            counts = np.stack([P.EVAL[(lang, n, i)][1] for i in range(E)])
            pick = counts[np.arange(E), G.pair(model, q, cc, 1).argmax(1).numpy()]
            hits[lang, 'withheld' if n in withheld.get(lang, ()) else 'paired'].append(pick == n)
    enc['image'].train(); enc['sequence'].train()
    return {f'{lang}/{grp}': {'all': float(np.concatenate(h).mean()), 'even': float(np.concatenate([x[::2] for x in h]).mean()),
                              'odd': float(np.concatenate([x[1::2] for x in h]).mean())}
            for (lang, grp), h in hits.items() if h}


def transfer8(enc, model, withheld, match=None):
    """The eight transfer measures of the proposal (cross-sense and comparison from the senses evaluation, matching on
    withheld counts)."""
    m = match or matching(enc, model, withheld)
    s = SP.evaluate_senses(enc, model, withheld, False)
    cs, cmp = s['cross_sense'], s['comparison']
    return {'chinese_withheld_matching': m['chinese/withheld']['all'], 'devanagari_withheld_matching': m['devanagari_digits/withheld']['all'],
            'taps_to_chinese_withheld': cs['taps_to_form/chinese/withheld'], 'taps_to_devanagari_withheld': cs['taps_to_form/devanagari_digits/withheld'],
            'chinese_to_taps_withheld': cs['form_to_taps/chinese/withheld'], 'comparison_chinese': cmp['forms/chinese'],
            'comparison_devanagari': cmp['forms/devanagari_digits'], 'comparison_english': cmp['forms/english']}, m


def four_way(enc, model, withheld):
    """Matching and the eight transfer measures with nothing, the memory part, the structure part, or both silenced.
    Arms without a memory/structure split report only the full network."""
    split = any(isinstance(v, SplitLinear) for v in first_layers(model).values())
    out = {}
    for name, parts in FLAGS.items():
        if parts and not split:
            continue
        with silenced(model, parts):
            t8, m = transfer8(enc, model, withheld)
        out[name] = {'matching': m, 'transfer8': t8}
    return out


def coordinate_stats(enc, model):
    cp = AN.coordinate_probe(enc, model)
    align, coord = FCC.direction(enc, model)
    return {'probe': cp, 'linear_axis_alignment': align, 'linear_axis_coordinate': coord}


def reader_parts(model):
    """(layer, active code-input mask) for every first-layer piece."""
    out = []
    for lin in first_layers(model).values():
        for part in ((lin.struct, lin.mem) if isinstance(lin, SplitLinear) else (lin,)):
            if part is not None:
                m = part.mask if hasattr(part, 'mask') else torch.ones_like(part.weight)
                out.append((part, m[:, :2 * CODE] > 0))
    return out


@torch.no_grad()
def disruption(enc, model, withheld, seed, point):
    """Transfer measures with each active code-input weight of the readers' first layer zeroed with probability 0.25;
    ten draws from the same checkpoint (weights restored between draws), averaged within the seed."""
    parts = reader_parts(model)
    saved = [p.weight.clone() for p, _ in parts]
    draws = []
    for d in range(DISRUPT_DRAWS):
        rng = nc.stream('lm_disrupt', seed, point, d)
        for (p, act), w in zip(parts, saved):
            drop = torch.as_tensor(rng.random(tuple(act.shape)) < DISRUPT_P).to(act.device) & act
            p.weight[:, :2 * CODE][drop] = 0.
        draws.append(transfer8(enc, model, withheld)[0])
        for (p, _), w in zip(parts, saved):
            p.weight.copy_(w)
    return {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}


@torch.no_grad()
def memory_activity(enc, model):
    """Per memory unit: share of sampled code pairs (collections and forms) on which the unit is active, and its
    outgoing weight norm. Empty for arms without a memory part."""
    code = G.codes_fn(enc, model)
    colls = torch.stack([code(P.DOTS['succ'][n], 'image') for n in ns.NUMBERS]).reshape(-1, CODE)
    forms = torch.cat([torch.stack([code(X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]], 'image') for n in ns.NUMBERS]).reshape(-1, CODE) for L in P.LANGS])
    allc = torch.cat([colls, forms])
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(allc), size=(2, 2000))
    a, b = allc[i], allc[j]
    out = {}
    for name, seq in readers(model):
        if not isinstance(seq[0], SplitLinear) or seq[0].mem is None:
            continue
        x = torch.cat([a, b, torch.zeros(len(a), 8)], 1) if name == 'g' else torch.cat([a, b], 1)
        act = (seq[0].mem(x) > 0).float().mean(0)
        out[name] = {'active_share': act.tolist(), 'outgoing_norm': seq[2].weight[:, STRUCT:].norm(dim=0).tolist()}
    return out


def norms(model):
    out = {}
    for name, seq in readers(model):
        out[name] = {'second': float(seq[2].weight.norm()), 'third': float(seq[4].weight.norm()),
                     'memory_columns': float(seq[2].weight[:, STRUCT:].norm())}
    return out


def snapshot(enc, model, withheld, seed, point, disrupt):
    """Everything recorded at a checkpoint."""
    ev = P.evaluate(enc, model, withheld, full=True)
    ev['senses'] = D.evaluate_senses(enc, model, withheld, full=True)
    out = {'evaluate': ev, 'four_way': four_way(enc, model, withheld), **coordinate_stats(enc, model),
           'memory_activity': memory_activity(enc, model), 'norms': norms(model)}
    if disrupt:
        out['disruption'] = disruption(enc, model, withheld, seed, point)
    return out


# ---------------------------------------------------------------- runs

def save_atomic(obj, path):
    buf = io.BytesIO()
    torch.save(obj, buf)
    DUR.atomic_write(path, buf.getvalue())


def weights(enc, model, withheld, arm, seed):
    return {'image': enc['image'].state_dict(), 'sequence': enc['sequence'].state_dict(), 'sri': model.state_dict(),
            'arm': arm, 'seed': seed, 'withheld': {k: list(v) for k, v in withheld.items()}}


def rehearsal_batch(eps, buffer, reh_rng):
    """The pilots' rehearsal: N_REHEARSE episodes drawn from the buffer replace the last of the batch (retraining only)."""
    earlier = list(buffer)
    old = [buffer[earlier[int(reh_rng.integers(len(earlier)))]][int(reh_rng.integers(D.BUFFER))] for _ in range(D.N_REHEARSE)]
    return eps[:D.BATCH_EP - D.N_REHEARSE] + old


def setup(arm, seed):
    CURRENT.update(arm=arm, seed=seed, step=0, r=2. ** (-1. / HALF_LIFE[arm]) if arm in HALF_LIFE else None)
    torch.manual_seed(seed)
    enc, model = X.make_model(seed)
    params = [*enc['image'].parameters(), *enc['sequence'].parameters(), *model.parameters()]
    return enc, model, torch.optim.Adam(params, P.LR)


def run_main(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    withheld = D.withheld_counts(seed)
    enc, model, opt = setup(arm, seed)
    rng = nc.stream('partA_train', seed, 'A-order')
    reh_rng, buffer = nc.stream('partD_rehearsal', seed), {}
    losses, series, snaps, cpu = [], [], {}, {'train': 0., 'eval': 0.}
    ck = RUNS / 'checkpoints'
    updates = 0

    def timed(f, *a):
        e0 = time.process_time()
        r = f(*a)
        cpu['eval'] += time.process_time() - e0
        return r

    def periodic(phase):
        t8, m = timed(transfer8, enc, model, withheld)
        series.append({'update': updates, 'phase': phase, 'transfer8': t8, 'matching': m, 'norms': norms(model)})

    for b, lang in enumerate(P.LANGS):
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        batches = [P.episodes(lang, withheld[lang], srng, D.BATCH_EP) for _ in range(D.STEPS)]
        e0 = time.process_time()
        for step, eps in enumerate(batches):  # no rehearsal in the blocks
            ls = train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
            updates += 1
            if step % 100 == 99:
                losses.append([b, step + 1, *ls])
            if updates % EVERY == 0:
                cpu['train'] += time.process_time() - e0
                periodic(f'block{b + 1}')
                e0 = time.process_time()
        cpu['train'] += time.process_time() - e0
        pool = [ep for batch in batches for ep in batch]  # the buffer is filled as before but not sampled until retraining
        buffer[lang] = [pool[i] for i in nc.stream('partD_buffer', seed, lang).choice(len(pool), D.BUFFER, replace=len(pool) < D.BUFFER)]
        snaps[f'block{b + 1}'] = timed(snapshot, enc, model, withheld, seed, f'block{b + 1}', b == len(P.LANGS) - 1)
        if b in (0, len(P.LANGS) - 1):
            save_atomic(weights(enc, model, withheld, arm, seed), ck / f'{arm}-seed{seed}-block{b + 1}.pt')
    if arm == 'mixed':  # the fork parent: model, optimizer, buffer
        save_atomic({**weights(enc, model, withheld, arm, seed), 'opt': opt.state_dict(), 'buffer': buffer, 'step': CURRENT['step']},
                    RUNS / 'parents' / f'{arm}-seed{seed}.pt')
    if arm in HALF_LIFE:  # decay-only diagnostic: the end-of-block-3 network, nothing trained, decay applied for the delay
        frozen = copy.deepcopy(model)
        for _ in range(DELAY):
            decay_step(frozen, CURRENT['r'])
        snaps['decay_only'] = {'four_way': timed(four_way, enc, frozen, withheld)}
    drng, crng = nc.stream('lm_delay', seed), nc.stream('lm_delay_context', seed)
    e0 = time.process_time()
    for step in range(DELAY):  # context episodes only: no forms, no rehearsal
        train_update(enc, model, opt, SP.batch_material([], D.context_episodes(drng, D.N_CONTEXT), 'taps'), rng, True)
        updates += 1
        if updates % EVERY == 0:
            cpu['train'] += time.process_time() - e0
            periodic('delay')
            e0 = time.process_time()
    cpu['train'] += time.process_time() - e0
    snaps['delay'] = timed(snapshot, enc, model, withheld, seed, 'delay', True)
    save_atomic(weights(enc, model, withheld, arm, seed), ck / f'{arm}-seed{seed}-delay.pt')
    relearning = {}
    points = {round(RELEARN * f) for f in (0, .125, .25, .5, 1)}
    srng, crng = nc.stream('lm_relearn_schedule', seed), nc.stream('lm_relearn_context', seed)
    e0 = time.process_time()
    for t in range(RELEARN + 1):
        if t in points:
            cpu['train'] += time.process_time() - e0
            relearning[str(t)] = {'matching': timed(matching, enc, model, withheld), 'memory_activity': timed(memory_activity, enc, model)}
            e0 = time.process_time()
        if t == RELEARN:
            break
        eps = rehearsal_batch(P.episodes(P.LANGS[0], withheld[P.LANGS[0]], srng, D.BATCH_EP), buffer, reh_rng)
        train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
        updates += 1
    cpu['train'] += time.process_time() - e0
    snaps['relearn'] = timed(snapshot, enc, model, withheld, seed, 'relearn', True)
    path = ck / f'{arm}-seed{seed}-relearn.pt'
    save_atomic(weights(enc, model, withheld, arm, seed), path)
    return f'{arm}/seed{seed}', {'withheld': {k: list(v) for k, v in withheld.items()}, 'loss_every_100': losses, 'series': series,
                                 'snapshots': snaps, 'relearning': relearning,
                                 'checkpoint': {'file': str(path.relative_to(X.ROOT)), 'sha256': nc.sha256(path)}}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, **{f'{k}_cpu_seconds': v for k, v in cpu.items()},
        'updates': updates, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run_fork(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    path = X.ROOT / 'runs' / PARENT_NAME / 'parents' / f'mixed-seed{seed}.pt'
    if not path.exists():
        raise SystemExit(f'fork parent missing: {path} (run the parent stage first)')
    ck = torch.load(path, weights_only=False)
    withheld = {k: tuple(v) for k, v in ck['withheld'].items()}
    CURRENT.update(arm='mixed', seed=seed, step=ck['step'], r=None if FORK[arm] is None else 2. ** (-1. / FORK[arm]))
    torch.manual_seed(seed)
    enc, model = X.make_model(seed)
    model.load_state_dict(ck['sri']); enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    opt = torch.optim.Adam([*enc['image'].parameters(), *enc['sequence'].parameters(), *model.parameters()], P.LR)
    opt.load_state_dict(ck['opt'])
    buffer = ck['buffer']
    srng, crng, rng, reh_rng = (nc.stream(n, seed) for n in ('lm_fork_schedule', 'lm_fork_context', 'lm_fork_train', 'lm_fork_rehearsal'))
    lang = P.LANGS[-1]
    track = []

    def record(update):
        row = {'update': update}
        for name in ('full', 'mem_silenced', 'struct_silenced'):
            with silenced(model, FLAGS[name]):
                row[name] = matching(enc, model, withheld)
        track.append(row)

    record(0)
    for step in range(FORK_STEPS):  # retraining: block-3 episodes plus rehearsal draws, identical in every branch
        eps = rehearsal_batch(P.episodes(lang, withheld[lang], srng, D.BATCH_EP), buffer, reh_rng)
        train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
        if (step + 1) % EVERY == 0:
            record(step + 1)
    final = {'four_way': four_way(enc, model, withheld), **coordinate_stats(enc, model), 'norms': norms(model)}
    return f'{arm}/seed{seed}', {'track': track, 'final': final, 'half_life': FORK[arm]}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'updates': FORK_STEPS,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run(task):
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    key, res, rt = (run_fork if task[0] in FORK else run_main)(task)
    if DEV.ACTIVE:
        rt['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    return key, res, rt


def install():
    SB.install()
    X.make_model = lambda seed: DEV.build(lambda: make_model(seed, CURRENT['arm']))
    TP.EXTRA_LOSS = SP.sequence_loss
    P.evaluate, D.evaluate_senses, D.train_update, D.batch_material = G.evaluate, SP.evaluate_senses, train_update, SP.batch_material


def main():
    install()
    arms = arms_for_stage()
    D.ARMS = {a: (False, 'taps') for a in arms}
    D.SEEDS = SEEDS
    only = os.environ.get('NFI_ONLY')  # 'arm:seed,...': run only these (timing, checks)
    if only:
        picked = [t.split(':') for t in only.split(',')]
        D.ARMS = {a: D.ARMS[a] for a in dict.fromkeys(a for a, _ in picked)}
        D.SEEDS = sorted({int(s) for _, s in picked})
    D.NAME = NAME
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, RUNS
    D.run = DUR.durable(run, RUNS / 'done', '{0}-seed{1}')
    D.main()
    path = X.ROOT / 'reports' / f'{NAME}-runtime.json'
    rt = __import__('json').loads(path.read_text())
    rt.update({'threads_per_worker': FI.THREADS, 'stage': STAGE, **DEV.info()})
    path.write_text(__import__('json').dumps(rt, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
