"""Number concept network v1: digit succession, one-anchor dice transfer and controls.

Implements docs/number-concept-network-v1-spec.md. Modes:

  pilot         seed 100 (stage-1 donor seed 101), 500 digits per class, 200 dice per count,
                5 epochs/stage, validation pools only; random-layout results are not computed.
  confirmation  seeds 0-4, full pools, 20 epochs/stage, held-out test sets plus diagnostic
                validation metrics. Refuses to run until docs/number-concept-network-v1-contract.md
                exists (the frozen contract).

Stage 1 trains the digit modality and S per (seed, variant); stage 2 trains five dice arms
(U, A, SH, AO, SW) per stage-1 checkpoint. Deterministic scientific results, the split/sample
manifest and nondeterministic runtime data are written to separate files. Checkpoints go to
runs/ (gitignored); their SHA-256 values are recorded in the results.

Environment: NCN_PREFIX (output prefix), NCN_WORKERS, NCN_MNIST (IDX directory),
NCN_SMOKE=1 (tiny pools and one epoch, for pipeline smoke tests only).
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import contextlib
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import number_concept as nc  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

MODE = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
SMOKE = os.environ.get('NCN_SMOKE') == '1'
SPEC = ROOT / 'docs/number-concept-network-v1-spec.md'
CONTRACT = ROOT / 'docs/number-concept-network-v1-contract.md'
NAME = f'number-concept-network-v1-{MODE}' + ('-smoke' if SMOKE else '')
PREFIX = Path(os.environ.get('NCN_PREFIX', str(ROOT / 'reports' / NAME)))
RUNS = ROOT / 'runs' / PREFIX.name
MNIST_DIR = Path(os.environ.get('NCN_MNIST', str(ROOT / 'data/mnist')))
WORKERS = int(os.environ.get('NCN_WORKERS', str(max(1, (os.cpu_count() or 1) - 1))))
TIMING_WARMUP, TIMING_REPS = 10, 100

CONFIGS = {
    'pilot': dict(seeds=(100,), donors={100: 101}, digits_per_class=500, dice_per_count=200, epochs=5,
                  eval_splits=('validation',), random_layout=False),
    'confirmation': dict(seeds=(0, 1, 2, 3, 4), donors={s: (s + 1) % 5 for s in range(5)}, digits_per_class=None,
                         dice_per_count=None, epochs=20, eval_splits=('validation', 'test'), random_layout=True),
}
CFG = dict(CONFIGS[MODE])
if SMOKE:
    CFG.update(digits_per_class=60, dice_per_count=40, epochs=1)
HEADLINE_SPLIT = 'test' if MODE == 'confirmation' else 'validation'
SMOKE_POOL = {'standard': {'train': 60, 'validation': 30, 'test': 30}, 'random': {'validation': 10, 'test': 10}}
DATA = {}


# ---------------------------------------------------------------- data

def prepare():
    raw, checks = nc.load_mnist(MNIST_DIR)
    train_all, val = nc.mnist_split(raw['train_labels'])
    lab = raw['train_labels']
    train = train_all
    if CFG['digits_per_class']:
        train = train_all[nc.class_subset(lab[train_all], nc.DIGITS, CFG['digits_per_class'], 'mnist')]
    images = lambda a: torch.from_numpy(a.astype(np.float32) / 255)
    digit = {'train': (images(raw['train_images'][train]), lab[train]),
             'validation': (images(raw['train_images'][val]), lab[val])}
    if 'test' in CFG['eval_splits']:
        digit['test'] = (images(raw['test_images']), raw['test_labels'])
    pools, placement = {}, {}
    for layout, splits in nc.DICE_POOLS.items():
        for split in splits:
            n = SMOKE_POOL[layout][split] if SMOKE else None
            pool = nc.build_dice(layout, split, per_count=n)
            pools[(layout, split)] = pool
            placement[f'{layout}/{split}'] = nc.placement_summary(pool)
    dtrain = pools[('standard', 'train')]
    dice_idx = np.arange(len(dtrain['ids']))
    if CFG['dice_per_count']:
        dice_idx = nc.class_subset(dtrain['labels'], nc.FACES, CFG['dice_per_count'], 'dice')
    dice = {'train': (torch.from_numpy(dtrain['images'][dice_idx]), dtrain['labels'][dice_idx], dtrain['ids'][dice_idx])}
    for split in ('validation', 'test'):
        p = pools[('standard', split)]
        dice[split] = (torch.from_numpy(p['images']), p['labels'], p['ids'])
        p = pools[('random', split)]
        dice[f'random_{split}'] = (torch.from_numpy(p['images']), p['labels'], p['ids'])
    anchors = {}
    for seed in CFG['seeds']:
        ones = np.flatnonzero(dice['train'][1] == 1)
        anchors[seed] = int(ones[nc.stream('anchor', seed).integers(len(ones))])
    DATA.update(digit=digit, dice=dice, anchors=anchors)
    manifest = {
        'mnist_sha256': checks, 'mnist_validation_indices': val.tolist(), 'mnist_train_indices_used': train.tolist(),
        'dice_generator': nc.GENERATOR_VERSION, 'dice_train_ids_used': dice['train'][2].tolist(),
        'anchors': {str(s): {'position': i, 'image_id': int(dice['train'][2][i])} for s, i in anchors.items()},
        'dice_pools': {f'{l}/{s}': {**{k: p[k].tolist() for k in ('ids', 'labels', 'radius', 'brightness',
                                                                  'quarter', 'attempts')},
                                    'centers': [c[:n].tolist() for c, n in zip(p['centers'], p['labels'])]}
                       for (l, s), p in pools.items()},
    }
    hashes = {f'{l}/{s}': hashlib.sha256(p['images'].tobytes()).hexdigest() for (l, s), p in pools.items()}
    isolation = {
        'mnist_train_validation_disjoint': not set(train.tolist()) & set(val.tolist()),
        'mnist_validation_600_per_digit': bool((np.bincount(lab[val], minlength=10) == 600).all()),
        'mnist_test_loaded_only_in_confirmation': ('test' in digit) == (MODE == 'confirmation'),
        'dice_split_streams_distinct': len({hashes[k] for k in hashes}) == len(hashes),
        'anchor_in_dice_training_pool': all(dice['train'][1][i] == 1 for i in anchors.values()),
    }
    data_bytes = {'mnist_train_float32': int(digit['train'][0].numel() * 4),
                  'dice_pools_float32': int(sum(p['images'].nbytes for p in pools.values()))}
    return {'mnist_sha256': checks, 'dice_image_sha256': hashes, 'placement': placement, 'isolation': isolation,
            'sizes': {k: int(len(v[1])) for k, v in {**digit, **dice}.items()}, 'data_bytes': data_bytes}, manifest


# ---------------------------------------------------------------- workers

def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def ckpt_path(*parts):
    return RUNS / ('-'.join(map(str, parts)) + '.pt')


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return {'file': rel(path), 'sha256': nc.sha256(path), 'bytes': path.stat().st_size}


def worker_runtime(cpu0, wall0):
    return {'cpu_seconds': time.process_time() - cpu0, 'wall_seconds': time.perf_counter() - wall0,
            'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def stage1_task(args):
    seed, variant = args
    cpu0, wall0 = time.process_time(), time.perf_counter()
    images, labels = DATA['digit']['train']
    modality, successor = nc.init_digit(seed), nc.init_successor(seed, variant)
    init = {'digit': nc.state_hash(modality), 'successor': nc.state_hash(successor)}
    sampler = nc.Sampler(labels, nc.DIGITS, range(9), ('digit', seed))
    steps = nc.steps_for(len(labels), CFG['epochs'])
    res = {'seed': seed, 'variant': variant, 'init_hashes': init, 'steps': steps}
    try:
        log, counts = nc.train(modality, successor, images, labels, sampler, steps, train_successor=True,
                               pairs='correct', lam_succ=1.)
    except nc.NumericalFailure as e:
        res.update(failed=True, reason=str(e), log=e.log)
        return (seed, variant), res, {**worker_runtime(cpu0, wall0), 'counts': {}}
    res['training'] = log
    if variant == 'rotation':
        res['angles'] = successor.alpha.detach().tolist()
    res['eval'] = {}
    probe_cpu = time.process_time()
    for split in CFG['eval_splits']:
        ev, cents = nc.evaluate_digits(modality, successor, images, labels, *DATA['digit'][split])
        res['eval'][split] = ev
    counts['encoded_images_eval'] = sum(len(labels) + len(DATA['digit'][s][1]) for s in CFG['eval_splits'])
    counts['probe_fits'] = 2 * len(CFG['eval_splits'])
    eval_cpu = time.process_time() - probe_cpu
    res['checkpoint'] = save(ckpt_path('stage1', seed, variant),
                             {'digit': modality.state_dict(), 'successor': successor.state_dict(),
                              'centroids': cents, 'variant': variant, 'seed': seed})
    res['final_hashes'] = {'digit': nc.state_hash(modality), 'successor': nc.state_hash(successor)}
    rt = {**worker_runtime(cpu0, wall0), 'evaluation_cpu_seconds': eval_cpu, 'counts': counts}
    return (seed, variant), res, rt


def load_stage1(seed, variant):
    path = ckpt_path('stage1', seed, variant)
    if not path.exists():
        return None
    ck = torch.load(path, weights_only=False)
    modality, successor = nc.Modality(nc.stream('load')), nc.make_successor(variant, nc.stream('load'))
    modality.load_state_dict(ck['digit'])
    successor.load_state_dict(ck['successor'])
    for p in [*modality.parameters(), *successor.parameters()]:
        p.requires_grad_(False)
    return modality, successor, ck['centroids'], nc.sha256(path)


def train_dice(seed, successor, c1, arm):
    """Stage-2 boundary: only dice data, the frozen S and (anchored arms) the digit-1 centroid."""
    pairs, lam_succ, lam_anchor, _ = nc.ARMS[arm]
    images, labels, _ = DATA['dice']['train']
    anchor = images[DATA['anchors'][seed]]
    dice = nc.init_dice(seed)
    init = nc.state_hash(dice)
    sampler = nc.Sampler(labels, nc.FACES, range(1, 6), ('dice', seed))
    steps = nc.steps_for(len(labels), CFG['epochs'])
    log, counts = nc.train(dice, successor, images, labels, sampler, steps, train_successor=False, pairs=pairs,
                           lam_succ=lam_succ, anchor=anchor if lam_anchor else None, c1=c1, lam_anchor=lam_anchor)
    return dice, init, log, counts


def dice_task(args):
    seed, variant, arm = args
    cpu0, wall0 = time.process_time(), time.perf_counter()
    res = {'seed': seed, 'variant': variant, 'arm': arm}
    recipient = load_stage1(seed, variant)
    if recipient is None:
        res.update(unavailable=True, reason='recipient stage-1 fit failed or missing')
        return (seed, variant, arm), res, {**worker_runtime(cpu0, wall0), 'counts': {}}
    digit, own_s, cents, rhash = recipient
    res['recipient_checkpoint_sha256'] = rhash
    successor = own_s
    if nc.ARMS[arm][3]:
        donor_seed = CFG['donors'][seed]
        donor = load_stage1(donor_seed, variant)
        if donor is None:
            res.update(unavailable=True, reason=f'donor stage-1 fit (seed {donor_seed}) failed or missing')
            return (seed, variant, arm), res, {**worker_runtime(cpu0, wall0), 'counts': {}}
        successor = donor[1]
        res.update(donor_seed=donor_seed, donor_checkpoint_sha256=donor[3],
                   donor_differs_from_recipient=donor[3] != rhash)
    try:
        dice, init, log, counts = train_dice(seed, successor, cents[1].clone(), arm)
    except nc.NumericalFailure as e:
        res.update(failed=True, reason=str(e), log=e.log)
        return (seed, variant, arm), res, {**worker_runtime(cpu0, wall0), 'counts': {}}
    res.update(init_hash=init, training=log, steps=log['steps'])
    eval_cpu = time.process_time()
    images, labels, _ = DATA['dice']['train']
    pools = {}
    for split in CFG['eval_splits']:
        pools[split] = (DATA['dice'][split][0], DATA['dice'][split][1], True)
        if CFG['random_layout']:
            pools[f'random_{split}'] = (DATA['dice'][f'random_{split}'][0], DATA['dice'][f'random_{split}'][1], False)
    res['eval'] = nc.evaluate_dice(dice, successor, cents, images, labels, images[DATA['anchors'][seed]], pools)
    if nc.ARMS[arm][3]:
        res['eval']['operator_discrepancy'] = {
            split: nc.operator_discrepancy(successor, own_s, nc.encode(digit.encoder, DATA['digit'][split][0]))
            for split in CFG['eval_splits']}
    counts['encoded_images_eval'] = len(labels) + sum(len(p[1]) for p in pools.values())
    res['checkpoint'] = save(ckpt_path('dice', seed, variant, arm), {'dice': dice.state_dict(), 'arm': arm})
    rt = {**worker_runtime(cpu0, wall0), 'evaluation_cpu_seconds': time.process_time() - eval_cpu, 'counts': counts}
    return (seed, variant, arm), res, rt


# ---------------------------------------------------------------- after training

def grids(stage1, dice):
    """Fixed qualitative grids per (seed, variant); diagnostics, not translation metrics."""
    out = {}
    split = CFG['eval_splits'][0]
    rng = nc.stream('grid')
    dimg, dlab, _ = DATA['dice'][split]
    gimg, glab = DATA['digit'][split]
    di = [int(rng.choice(np.flatnonzero(dlab == c))) for c in nc.FACES]
    gi = [int(rng.choice(np.flatnonzero(glab == c))) for c in nc.FACES]
    sig = torch.sigmoid
    for (seed, variant), r in stage1.items():
        if r.get('failed') or seed not in CFG['seeds']:
            continue
        digit, _, _, _ = load_stage1(seed, variant)
        arms = [a for a in nc.ARMS if (seed, variant, a) in dice and 'checkpoint' in dice[(seed, variant, a)]]
        with torch.no_grad():
            xd, xg = dimg[di], gimg[gi]
            zg = digit.encoder(xg)
            rows = [list(xd.numpy()), list(xg.numpy()), list(sig(digit.decoder(zg)).numpy())]
            labels = ['dice input', 'digit input', 'digit reconstruction']
            for arm in arms:
                m = nc.Modality(nc.stream('load'))
                m.load_state_dict(torch.load(ckpt_path('dice', seed, variant, arm), weights_only=False)['dice'])
                zd = m.encoder(xd)
                rows += [list(sig(m.decoder(zd)).numpy()), list(sig(digit.decoder(zd)).numpy()),
                         list(sig(m.decoder(zg)).numpy())]
                labels += [f'{arm} dice reconstruction', f'{arm} dice->digit', f'{arm} digit->dice']
        path = Path(f'{PREFIX}-grids') / f'seed{seed}-{variant}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        nc.write_png(path, nc.tile(rows))
        out[f'{seed}/{variant}'] = {'file': rel(path), 'rows': labels,
                                    'columns': list(nc.FACES), 'split': split, 'dice_positions': di,
                                    'digit_positions': gi, 'sha256': nc.sha256(path)}
    return out


def timing(stage1, dice):
    """Deployment timing on the first seed's A arm: encoding+readout and cross-decoding."""
    out = {}
    seed = CFG['seeds'][0]
    for variant in nc.VARIANTS:
        if (seed, variant, 'A') not in dice or 'checkpoint' not in dice[(seed, variant, 'A')]:
            out[variant] = None
            continue
        digit, _, cents, _ = load_stage1(seed, variant)
        m = nc.Modality(nc.stream('load'))
        m.load_state_dict(torch.load(ckpt_path('dice', seed, variant, 'A'), weights_only=False)['dice'])
        imgs = DATA['dice'][CFG['eval_splits'][0]][0]
        fns = {'encode_readout': lambda x: nc.predict(m.encoder(x), cents),
               'cross_decode': lambda x: torch.sigmoid(digit.decoder(m.encoder(x)))}
        out[variant] = {}
        for name, fn in fns.items():
            for bs in (1, 256):
                x = imgs[:bs].clone()
                with torch.no_grad():
                    for _ in range(TIMING_WARMUP):
                        fn(x)
                    t = []
                    for _ in range(TIMING_REPS):
                        t0 = time.perf_counter()
                        fn(x)
                        t.append(time.perf_counter() - t0)
                q = np.percentile(t, [25, 50, 75])
                out[variant][f'{name}_batch{bs}'] = {'median_seconds': q[1], 'iqr_seconds': q[2] - q[0],
                                                     'repetitions': TIMING_REPS, 'warmup': TIMING_WARMUP}
    out['scope'] = ('CPU, one thread; includes allocation; no host/device transfer or compilation; '
                    'encode_readout includes cosine readout, cross_decode includes sigmoid')
    return out


def memory():
    digit, dice = nc.Modality(nc.stream('m')), nc.Modality(nc.stream('m'))
    out = {}
    for variant in nc.VARIANTS:
        s = nc.make_successor(variant, nc.stream('m'))
        p = {'digit_modality': nc.param_bytes(digit), 'dice_modality': nc.param_bytes(dice), 'successor': nc.param_bytes(s),
             'digit_centroids': 10 * nc.D * 8}
        out[variant] = {
            'parameter_bytes': p,
            'stage1_gradient_bytes': p['digit_modality'] + p['successor'],
            'stage1_adam_moment_bytes': 2 * (p['digit_modality'] + p['successor']),
            'stage2_gradient_bytes': p['dice_modality'], 'stage2_adam_moment_bytes': 2 * p['dice_modality'],
            'inference_full_bytes': sum(p.values()),
            'inference_recognition_only_bytes': nc.param_bytes(dice.encoder) + p['digit_centroids'],
            'probe_feature_cache_bytes_per_example': 16 * 4,
        }
    out['note'] = 'complex64 8 bytes/scalar, float32 4 bytes/scalar; analytical counts, not allocator measurements'
    return out


def summary(stage1, dice):
    """Per-variant decision summary. In the pilot this is a pipeline diagnostic, not a headline."""
    out = {'split': HEADLINE_SPLIT, 'mode': MODE,
           'status': 'headline' if MODE == 'confirmation' and not SMOKE else 'diagnostic only; not a headline test'}
    for variant in nc.VARIANTS:
        v = {'seeds': {}}
        for seed in CFG['seeds']:
            s1 = stage1.get((seed, variant), {})
            row = {'stage1_failed': bool(s1.get('failed'))}
            if not s1.get('failed'):
                ev = s1['eval'][HEADLINE_SPLIT]
                row.update(centroid_accuracy=ev['centroid_accuracy'], probe_accuracy=ev['probe_full']['accuracy'],
                           gates=ev['gates'], digit_composition_k2=ev['composition']['2']['macro_accuracy'])
            for arm in nc.ARMS:
                r = dice.get((seed, variant, arm), {})
                ok = 'eval' in r
                row[arm] = r['eval']['pools'][HEADLINE_SPLIT]['primary_macro_2_6'] if ok else None
                if arm == 'A' and ok:
                    row['dice_composition_k2'] = r['eval']['pools'][HEADLINE_SPLIT]['composition']['2']['macro_accuracy']
            v['seeds'][str(seed)] = row
        rows = list(v['seeds'].values())
        a = [r['A'] for r in rows]
        complete = all(x is not None for x in a)
        v['gates_all_seeds'] = all(not r['stage1_failed'] and all(r['gates'].values()) for r in rows)
        v['A_mean'] = float(np.mean(a)) if complete else None
        v['A_min'] = float(np.min(a)) if complete else None
        v['paired_differences'] = {}
        for ctrl in ('SH', 'AO', 'SW'):
            c = [r[ctrl] for r in rows]
            if complete and all(x is not None for x in c):
                d = np.array(a) - np.array(c)
                v['paired_differences'][ctrl] = {'differences': d.tolist(), 'mean': float(d.mean()),
                                                 'sd': float(d.std(ddof=1)) if len(d) > 1 else None}
            else:
                v['paired_differences'][ctrl] = None
        pd = v['paired_differences']
        v['primary_pass'] = bool(complete and v['gates_all_seeds'] and v['A_mean'] >= .9 and v['A_min'] >= .8
                                 and all(pd[c] is not None and pd[c]['mean'] >= .2 for c in pd))
        k2 = [(r.get('digit_composition_k2'), r.get('dice_composition_k2')) for r in rows]
        v['composition_pass'] = bool(all(x is not None and y is not None for x, y in k2)
                                     and np.mean([x for x, _ in k2]) >= .9 and np.mean([y for _, y in k2]) >= .9)
        out[variant] = v
    return out


# ---------------------------------------------------------------- main

def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def write(path, text):
    data = text.encode()
    info = {'sha256_uncompressed': hashlib.sha256(data).hexdigest(), 'bytes_uncompressed': len(data)}
    if len(data) > 1_000_000:
        with open(str(path) + '.gz', 'wb') as f:
            with gzip.GzipFile(filename='', mode='wb', fileobj=f, compresslevel=9, mtime=0) as g:
                g.write(data)
        info['file'] = rel(str(path) + '.gz')
    else:
        path.write_bytes(data)
        info['file'] = rel(path)
    return info


def main():
    if MODE not in CONFIGS:
        sys.exit(f'unknown mode {MODE!r}; use pilot or confirmation')
    if MODE == 'confirmation' and not SMOKE and not CONTRACT.exists():
        sys.exit(f'{CONTRACT.relative_to(ROOT)} does not exist: freeze the contract before confirmation runs')
    wall0, cpu0 = time.perf_counter(), time.process_time()
    phases = {}

    @contextlib.contextmanager
    def phase(name):
        w, c = time.perf_counter(), time.process_time()
        yield
        phases[name] = {'wall_seconds': time.perf_counter() - w, 'parent_cpu_seconds': time.process_time() - c}

    res = {'spec': str(SPEC.relative_to(ROOT)), 'mode': MODE, 'smoke': SMOKE,
           'config': {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in CFG.items()}}
    with phase('acceptance_checks'):
        res['acceptance_checks'] = nc.acceptance_checks()
    with phase('data_preparation'):
        res['data'], manifest = prepare()
    stage1_jobs = [(s, v) for s in sorted({*CFG['seeds'], *CFG['donors'].values()}) for v in nc.VARIANTS]
    dice_jobs = [(s, v, a) for s in CFG['seeds'] for v in nc.VARIANTS for a in nc.ARMS]
    worker = {}
    with mp.get_context('fork').Pool(min(WORKERS, len(dice_jobs))) as pool:
        with phase('stage1'):
            stage1 = {}
            for key, r, rt in pool.imap_unordered(stage1_task, stage1_jobs):
                stage1[key] = r
                worker['stage1/{}/{}'.format(*key)] = rt
        with phase('stage2'):
            dice = {}
            for key, r, rt in pool.imap_unordered(dice_task, dice_jobs):
                dice[key] = r
                worker['dice/{}/{}/{}'.format(*key)] = rt
    res['stage1'] = {f'{s}/{v}': stage1[(s, v)] for s, v in stage1_jobs}
    res['dice'] = {f'{s}/{v}/{a}': dice[(s, v, a)] for s, v, a in dice_jobs}
    inits = {}
    for (s, v, a), r in dice.items():
        if 'init_hash' in r:
            inits.setdefault(s, set()).add(r['init_hash'])
    res['acceptance_checks']['identical_initial_dice_weights_across_arms_and_variants'] = all(
        len(h) == 1 for h in inits.values())
    res['acceptance_checks']['stage1_frozen_in_stage2'] = all(
        nc.sha256(ckpt_path('stage1', s, v)) == stage1[(s, v)]['checkpoint']['sha256']
        for s, v in stage1 if 'checkpoint' in stage1[(s, v)])
    res['acceptance_checks']['sw_donor_identity'] = all(
        r.get('donor_differs_from_recipient', False) for (s, v, a), r in dice.items() if a == 'SW' and 'eval' in r)
    res['data']['isolation'] = {k: bool(v) for k, v in res['data']['isolation'].items()}
    with phase('grids'):
        res['grids'] = grids(stage1, dice)
    res['summary'] = summary(stage1, dice)
    with phase('deployment_timing'):
        deploy = timing(stage1, dice)
    with phase('artifact_writing'):
        PREFIX.parent.mkdir(parents=True, exist_ok=True)
        outputs = {'results': write(Path(f'{PREFIX}-results.json'), json.dumps(res, indent=1, sort_keys=True, default=float) + '\n'),
                   'manifest': write(Path(f'{PREFIX}-manifest.json'), json.dumps(manifest, sort_keys=True) + '\n')}
    counts = {}
    for rt in worker.values():
        for k, n in rt['counts'].items():
            counts[k] = counts.get(k, 0) + n
    runtime = {
        'spec_sha256': nc.sha256(SPEC), 'contract_sha256': nc.sha256(CONTRACT) if CONTRACT.exists() else None,
        'runner_sha256': nc.sha256(__file__), 'module_sha256': nc.sha256(nc.__file__),
        'revision': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'environment_overrides': {
            k: v for k, v in os.environ.items() if k.startswith('NCN_')},
        'seed_policy': f'named SeedSequence streams under ROOT_SEED {nc.ROOT_SEED}: mnist_split, subset, dice_meta, '
                       'dice_place, dice_noise, init/{digit,dice,successor}, sample/{modality,seed,role}, anchor, grid',
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu': platform.processor(), 'cpu_count': os.cpu_count(), 'workers': WORKERS, 'threads_per_worker': 1,
        'dtype': 'complex64 activations/weights, float32 pixels and losses',
        'blas': 'single-threaded (OPENBLAS/OMP/MKL_NUM_THREADS=1, torch.set_num_threads(1))',
        'determinism': 'torch.use_deterministic_algorithms(True), CPU only; fork workers with per-task streams',
        'complex_adam': f'torch {torch.__version__} Adam treats complex parameters as real/imaginary views '
                        '(checked in acceptance_checks.complex_adam_equals_real_view)',
        'phases': phases, 'worker_tasks': worker,
        'worker_cpu_seconds_total': sum(rt['cpu_seconds'] for rt in worker.values()),
        'counts_total': counts, 'deployment_timing': deploy, 'memory': memory(),
        'data_bytes': res['data']['data_bytes'],
        'checkpoint_bytes': {k: r['checkpoint']['bytes'] for k, r in {**res['stage1'], **res['dice']}.items()
                             if 'checkpoint' in r},
        'peak_workspace_bytes': None, 'peak_workspace_reason': 'torch CPU allocator peak is not exposed; see peak RSS',
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process',
        'not_applicable': ['grouping search', 'sparse code', 'covariance estimator', 'solver tolerance'],
        'outputs': outputs}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps(runtime, indent=1, sort_keys=True, default=float) + '\n')
    brief = {v: {k: res['summary'][v][k] for k in ('A_mean', 'gates_all_seeds', 'primary_pass')} for v in nc.VARIANTS}
    print(json.dumps({'acceptance_checks_pass': all(
        (all(x.values()) if isinstance(x, dict) else x) for x in res['acceptance_checks'].values()),
        'isolation': res['data']['isolation'], 'summary': brief, 'wall_seconds': runtime['wall_seconds']},
        indent=1, default=float))


if __name__ == '__main__':
    main()
