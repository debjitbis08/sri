"""Shared-core sparse memory v1, stage 0 (docs/shared-core-sparse-memory-v1-contract.md): can one core hold both relations and
enough input identity to address memory? Host arms widen the code and add a reconstruction objective; every arm trains
blocks 1-3 and keeps its block-3 checkpoint, which the readings read on symbols no host has seen. No memory runs here.

Stages (argv[2]):
  hosts    every configuration x pilot seed, sparse48 first; sparse48 must reproduce the long-term memory host's block-3 state
  read     the stage 0 readings on those checkpoints, and the frozen-encoder reference
  lock     the contract's lock table, from the pilot means

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.shared_core_stage0 pilot hosts
       (rerun the same command to resume after an interruption: experiments/durable.py)
"""
import copy
import io
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import fan_in_pilot_analysis as AN
from experiments import generic_relations as G
from experiments import limited_memory as LM
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import number_systems as X
from experiments import pairing_symbols as PS
from experiments import sequence_pilot as SP
from experiments import streaming_retrieval as SR
from experiments import transition_pilot as TP
from src import generic_net as gen
from src import number_core as ncore
from src import number_systems_net as net

nc, ns = X.nc, X.ns
SMOKE = X.SMOKE
NAME = 'shared-core-sparse-memory-v1-stage0' + ('-smoke' if SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CONTRACT = X.ROOT / 'docs' / 'shared-core-sparse-memory-v1-contract.md'
REFERENCE = X.ROOT / 'runs' / 'long-term-memory-v1' / 'cache'
BACKGROUND = X.ROOT / 'runs' / 'translation-memory-v1-stable-keys' / 'material'
WORKERS = int(os.environ.get('SC0_WORKERS', 3))
SEEDS = (392,) if SMOKE else (392, 393)
LAMBDAS = (.1, 1.)
BG_PER_UPDATE = 64
LOADS, TRAIN_RENDERS, EVAL_RENDERS = ((18, 36), 8, 2) if SMOKE else ((90, 180), 64, 5)
ADDRESS_BAR, MARGIN = .80, .05
TRANSFER_SETS = ('taps', 'forms/chinese', 'forms/devanagari_digits')
OUTER, MIDDLE = net.SIZES[:2]
CONFIGS = ('sparse48', 'wide32', *(f'{a}-l{l:g}' for a in ('wide32-recon', 'wide32-recon-bg', 'split-recon-bg') for l in LAMBDAS))
ADDED_COST = {'sparse48': 0, 'wide32': 0, 'wide32-recon': 1, 'wide32-recon-bg': 2, 'split-recon-bg': 2}


def parse(config):
    """config -> (arm, lambda or None)."""
    if '-l' in config:
        arm, lam = config.rsplit('-l', 1)
        return arm, float(lam)
    return config, None


def code_width(config):
    return 16 if config == 'sparse48' else 32


def read_width(config):
    return 16 if parse(config)[0] in ('sparse48', 'split-recon-bg') else 32


# ---------------------------------------------------------------- arms

class SplitCodeNet(gen.GenericSequenceNet):
    """sparse48's readers on the first READ coordinates of a wider code: the comparator and predictor never see the others."""
    READ = 16

    def relation_logits(self, ca, cb, head):
        return super().relation_logits(ca[:, :self.READ], cb[:, :self.READ], head)

    def predict(self, prev, cur):
        return super().predict(prev[:, :self.READ], cur[:, :self.READ])


def predict_logits_on_read(orig):
    """The prediction loss compares only the coordinates the predictor returns; nothing changes when they are the whole code."""
    return lambda model, pred, cand: orig(model, pred, cand[..., :pred.shape[-1]])


def set_width(config):
    """The inherited runners read the code width from module constants. Each task sets them for its own process."""
    w, r = code_width(config), read_width(config)
    FI.CODE, D.SIZES = r, (OUTER, MIDDLE, w)
    start = MIDDLE + w + OUTER + MIDDLE
    P.INNER = slice(start, start + w)


def wide_readers(model, seed):
    """As LM.make_model's sparse48, with the masks drawn from all of the code: one coordinate per operand, 48 units, no memory part."""
    for name, seq in LM.readers(model):
        base = seq[0]
        tmp = copy.deepcopy(base)
        FI.reader_mask(tmp, FI.CODE, nc.stream('lm_mask', seed, name, 'a'), nc.stream('lm_mask', seed, name, 'b'))
        struct = nn.Linear(base.weight.shape[1], LM.STRUCT)
        struct.weight.data, struct.bias.data = tmp.weight.data[:LM.STRUCT].clone(), tmp.bias.data[:LM.STRUCT].clone()
        struct.__class__ = FI.MaskedLinear
        struct.register_buffer('mask', tmp.mask[:LM.STRUCT].clone())
        struct.register_buffer('swaps', torch.zeros((), dtype=torch.long))
        seq[0] = LM.SplitLinear(struct, None)


def widen_split(model, seed):
    """The projection's first 16 rows stay sparse48's; the other 16 start from their own stream, at the standard initialization."""
    model.__class__ = SplitCodeNet
    model.sizes, model.state = (OUTER, MIDDLE, 32), OUTER + MIDDLE + 32
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(nc.stream('sc0_split_rows', seed).integers(2 ** 62)))
        for sense in ('image', 'sequence'):
            old, new = model.proj[sense], nn.Linear(OUTER, 32)
            new.weight.data[:16], new.bias.data[:16] = old.weight.data, old.bias.data
            model.proj[sense] = new


def make_host(seed, config):
    arm = parse(config)[0]
    if arm == 'sparse48':
        return LM.make_model(seed, 'sparse48')
    if arm == 'split-recon-bg':
        enc, model = LM.make_model(seed, 'sparse48')
        widen_split(model, seed)
        return enc, model
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(OUTER), 'sequence': ncore.MorseEncoder(OUTER)}
    model = gen.GenericSequenceNet(('image', 'sequence'), sizes=(OUTER, MIDDLE, 32))
    wide_readers(model, seed)
    return enc, model


class Decoder(nn.Module):
    """Code to image, the image encoder's mirror; used only in training."""

    def __init__(self, width):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, 128), nn.Linear(128, 32 * 49)
        self.d1, self.d2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ConvTranspose2d(16, 1, 3, 2, 1, 1)

    def forward(self, c):
        h = F.relu(self.fc2(F.relu(self.fc1(c)))).reshape(-1, 32, 7, 7)
        return torch.sigmoid(self.d2(F.relu(self.d1(h))))[:, 0]


def make_decoder(seed, config):
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(nc.stream('sc0_decoder', seed, config).integers(2 ** 62)))
        dec = Decoder(code_width(config))
    return dec.to(DEV.NAME) if DEV.ACTIVE else dec


class Reconstruction:
    """TP's EXTRA_LOSS: the sequence loss, plus lambda times the per-pixel mean squared error over the batch's images and any background images."""

    def __init__(self, enc, decoder, lam, background, rng):
        self.enc, self.decoder, self.lam, self.background, self.rng = enc, decoder, lam, background, rng
        self.log = {'curriculum': [], 'background': []}
        self.exposures = {'curriculum_images': 0, 'background_images': 0}

    def __call__(self, model, material, c, cand):
        items = material[0]
        rows = [k for k, it in enumerate(items) if it[0] != 'taps']
        x = torch.from_numpy(P.images([items[k] for k in rows]))
        err = (self.decoder(c[torch.as_tensor(rows)]) - x).pow(2).mean((1, 2))
        self.log['curriculum'].append(float(err.detach().mean()))
        self.exposures['curriculum_images'] += len(rows)
        if self.background is not None:
            n, r = self.background.shape[:2]
            xb = torch.from_numpy(np.ascontiguousarray(self.background[self.rng.integers(n, size=BG_PER_UPDATE), self.rng.integers(r, size=BG_PER_UPDATE)], dtype=np.float32))
            eb = (self.decoder(model.proj['image'](self.enc['image'](xb))) - xb).pow(2).mean((1, 2))
            self.log['background'].append(float(eb.detach().mean()))
            self.exposures['background_images'] += BG_PER_UPDATE
            err = torch.cat([err, eb])
        return SP.sequence_loss(model, material, c, cand) + self.lam * err.mean()


def background_images(seed):
    """The frozen encoder's background material, without its labels."""
    return np.load(BACKGROUND / f'seed{seed}.npz')['bg']


# ---------------------------------------------------------------- hosts

def state_hash(enc, model):
    """long_term_memory_host.state_hash: on the host's own state, so sparse48 can be compared with its cache."""
    import hashlib
    h = hashlib.sha256()
    for sd in (enc['image'].state_dict(), enc['sequence'].state_dict(), model.state_dict()):
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def checkpoint_path(config, seed):
    return RUNS / 'checkpoints' / f'{config}-seed{seed}.pt'


def set_up_task(config):
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    set_width(config)
    LM.install()
    X.make_model = lambda seed: DEV.build(lambda: make_host(seed, LM.CURRENT['arm']))
    TP.predict_logits = predict_logits_on_read(ORIG_PREDICT_LOGITS)
    TP.EXTRA_LOSS = SP.sequence_loss


ORIG_PREDICT_LOGITS = TP.predict_logits


def fit_host(task):
    config, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    set_up_task(config)
    arm, lam = parse(config)
    withheld = D.withheld_counts(seed)
    enc, model, opt = LM.setup(config, seed)
    LM.CURRENT['end'] = None
    decoder, recon = None, None
    if lam is not None:
        decoder = make_decoder(seed, config)
        opt.add_param_group({'params': list(decoder.parameters())})
        recon = Reconstruction(enc, decoder, lam, background_images(seed) if arm.endswith('-bg') else None, nc.stream('sc0_background', seed, config))
        TP.EXTRA_LOSS = recon
    rng = nc.stream('partA_train', seed, 'A-order')
    trail = {}
    for b, lang in enumerate(P.LANGS):
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            LM.train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
        if recon is not None:
            trail[f'block{b + 1}'] = {k: float(np.mean(v[-200:])) if v else None for k, v in recon.log.items()}
    path = checkpoint_path(config, seed)
    LM.save_atomic({**LM.weights(enc, model, withheld, config, seed), 'decoder': decoder.state_dict() if decoder is not None else None}, path)
    res = {'hash': state_hash(enc, model), 'updates': 3 * D.STEPS, 'parameters': {'host': sum(p.numel() for m in (*enc.values(), model) for p in m.parameters()),
                                                                                  'decoder': sum(p.numel() for p in decoder.parameters()) if decoder is not None else 0},
           'reconstruction_trailing_200': trail, 'exposures': recon.exposures if recon is not None else {}, 'checkpoint': {'file': str(path.relative_to(X.ROOT)), 'sha256': nc.sha256(path)}}
    return f'{config}/seed{seed}', res, timing(t0, c0)


def timing(t0, c0):
    return {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            **({'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated()} if DEV.ACTIVE else {})}


def reference_hash(seed):
    p = REFERENCE / f'sparse48-seed{seed}-full-N45.json'
    return json.load(open(p))['hashes']['3']


def prepare():
    X.prepare(); P.prepare(); D.prepare()


def pool_map(fn, tasks):
    assert not torch.cuda.is_initialized(), 'CUDA must not be initialised before the pool forks'
    with mp.get_context('fork').Pool(WORKERS, maxtasksperchild=1) as pool:
        return list(pool.imap_unordered(fn, tasks))


def hosts():
    t0 = time.perf_counter()
    prepare()
    run = DUR.durable(fit_host, RUNS / 'done-hosts', '{0}-seed{1}')
    first = pool_map(run, [('sparse48', s) for s in SEEDS])
    gate = {}
    for key, res, _ in first:
        seed = int(key.split('seed')[1])
        gate[str(seed)] = {'hash': res['hash'], 'reference': None if SMOKE else reference_hash(seed)}
        gate[str(seed)]['reproduces'] = None if SMOKE else res['hash'] == gate[str(seed)]['reference']
    save_json(f'{REPORT}-gate.json', {'gate': 'sparse48 through this runner against the long-term memory host block-3 state', 'seeds': gate})
    if not SMOKE and not all(g['reproduces'] for g in gate.values()):
        raise SystemExit(f'sparse48 does not reproduce its reference: {gate}')
    rest = pool_map(run, [(c, s) for c in CONFIGS if c != 'sparse48' for s in SEEDS])
    all_ = first + rest
    save_json(f'{REPORT}-hosts.json', {k: {x: r[x] for x in ('hash', 'updates', 'parameters', 'reconstruction_trailing_200', 'exposures')} for k, r, _ in all_})
    save_json(f'{REPORT}-hosts-runtime.json', runtime({'phase': 'host fits, including rejected configurations; selection cost', 'wall_seconds': time.perf_counter() - t0,
                                                       'fits': {k: t for k, _, t in all_}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in all_))}))
    print(json.dumps(gate), flush=True)


# ---------------------------------------------------------------- readings

def material_path(seed):
    return RUNS / 'material' / f'seed{seed}.npz'


def test_material(seed):
    """Symbols of the load sets, none seen by any host arm, rendered once and shared by every reading of a seed."""
    path = material_path(seed)
    pool = PS.survivors()
    sets = {L: PS.draw_sets(seed, L // 2, pool) for L in LOADS}
    cps = {L: [*sets[L]['A'], *sets[L]['B']] for L in LOADS}
    big = cps[LOADS[-1]]
    assert all(cps[L] == big[:L] for L in LOADS)
    if not path.exists():
        buf = io.BytesIO()
        np.savez_compressed(buf, train=np.stack([PS.renderings(c, pool[c], seed, 'train', TRAIN_RENDERS) for c in big]), eval=np.stack([PS.renderings(c, pool[c], seed, 'eval', EVAL_RENDERS) for c in big]), cps=np.array(big))
        DUR.atomic_write(path, buf.getvalue())
    z = np.load(path)
    counts = {L: np.array([sets[L]['A'].get(c, sets[L]['B'].get(c)) for c in cps[L]]) for L in LOADS}
    return z['train'], z['eval'], counts


@torch.no_grad()
def batched(fn, x, size=512):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return torch.cat([fn(torch.from_numpy(x[i:i + size])) for i in range(0, len(x), size)])


@torch.no_grad()
def address_reading(train, ev, counts):
    """train (S, R, d), ev (S, E, d), counts (S,). Nearest cosine prototype: identity, and the explicit store's count with perfect contents."""
    S, E = ev.shape[:2]
    row = torch.arange(S, device=ev.device).repeat_interleave(E)
    c = torch.as_tensor(counts, device=ev.device)

    def read(t, q):
        proto = F.normalize(F.normalize(t, dim=-1).mean(1), dim=-1)
        sim = F.normalize(q, dim=-1).reshape(S * E, -1) @ proto.T
        near = sim.argmax(1)
        same = sim[torch.arange(S * E, device=ev.device), row]
        return float((near == row).float().mean()), float((c[near] == c[row]).float().mean()), float(same.mean()), float((sim.sum(1) - same).mean() / (S - 1))

    identity, store, same, other = read(train, ev)
    mean = train.reshape(-1, train.shape[-1]).mean(0)
    centred = read(train - mean, ev - mean)
    return {'identity': identity, 'store': store, 'same_symbol_similarity': same, 'other_symbol_similarity': other, 'identity_centred': centred[0], 'store_centred': centred[1], 'chance_identity': 1 / S, 'chance_store': 1 / 9}


def split_codes(fn, train, ev):
    S, R = train.shape[:2]
    S2, E = ev.shape[:2]
    return batched(fn, train.reshape(-1, 28, 28)).reshape(S, R, -1), batched(fn, ev.reshape(-1, 28, 28)).reshape(S2, E, -1)


def all_addresses(fn, train, ev, counts):
    t, q = split_codes(fn, train, ev)
    return {str(L): address_reading(t[:L], q[:L], counts[L]) for L in LOADS}, t, q


@torch.no_grad()
def value_ceiling(code, model, seed):
    """The right count's mean code as retrieved content, read among nine candidates, and count prototypes on collections."""
    E = P.EVAL_PER
    fit = torch.stack([batched(code, P.DOTS['train'][:, k]).mean(0) for k in range(9)])
    fit_unit = torch.stack([F.normalize(batched(code, P.DOTS['train'][:, k]), dim=-1).mean(0) for k in range(9)])
    rng = nc.stream('sc0_value_candidates', seed)
    n = 450 if not SMOKE else 45
    truth = rng.integers(9, size=n)
    order = np.stack([rng.permutation(9) for _ in range(n)])
    j = rng.integers(E, size=n)
    imgs = np.stack([P.DOTS['succ'][int(k) + 1][j[t]] for t in range(n) for k in order[t]])
    cand = batched(code, imgs).reshape(n, 9, -1)
    content = fit[torch.as_tensor(truth, device=cand.device)]
    pos = np.argmax(order == truth[:, None], 1)
    out = {'cosine': float(((F.normalize(content, dim=-1)[:, None] * F.normalize(cand, dim=-1)).sum(-1).argmax(1).cpu().numpy() == pos).mean())}
    if model is not None:
        out['head'] = float((G.pair(model, content, cand, 1).argmax(1).cpu().numpy() == pos).mean())
    colls = torch.stack([batched(code, P.DOTS['succ'][k + 1]) for k in range(9)])
    near = (F.normalize(colls, dim=-1).reshape(9 * E, -1) @ F.normalize(fit_unit, dim=-1).T).argmax(1).cpu().numpy()
    out['count_prototypes'] = float((near == np.repeat(np.arange(9), E)).mean())
    return out


@torch.no_grad()
def shape(symbols, colls):
    """symbols (N, d), colls (9, E, d). Participation ratio, the top direction's share, and its relation to count."""
    def spectrum(x):
        x = x.double() - x.double().mean(0)
        val, vec = torch.linalg.eigh(x.T @ x / len(x))
        val, vec = val.flip(0).clamp_min(0), vec.flip(1)
        return val, vec
    sv, sd = spectrum(symbols)
    flat = colls.reshape(-1, colls.shape[-1])
    cv, cd = spectrum(flat)
    y = torch.arange(9., dtype=torch.float64, device=flat.device).repeat_interleave(colls.shape[1])
    corr = lambda d: float(torch.corrcoef(torch.stack([flat.double() @ d, y]))[0, 1])
    d = symbols.shape[-1]
    var = symbols.double().var(0)
    return {'symbols_participation_ratio': float(sv.sum() ** 2 / sv.pow(2).sum()), 'symbols_top_share': float(sv[0] / sv.sum()),
            'collections_participation_ratio': float(cv.sum() ** 2 / cv.pow(2).sum()), 'collections_top_share': float(cv[0] / cv.sum()),
            'collections_top_correlation_with_count': abs(corr(cd[:, 0])), 'symbols_top_correlation_with_count_on_collections': abs(corr(sd[:, 0])),
            'top_directions_cosine': abs(float(sd[:, 0] @ cd[:, 0])), 'symbol_variance_share_first_half': float(var[:d // 2].sum() / var.sum())}


@torch.no_grad()
def reader_use(enc, model, width):
    """Connections of live reader units, per code coordinate: are the readers using the added ones? A unit is live if it fires on some sampled pair."""
    code = G.codes_fn(enc, model)
    colls = torch.stack([code(P.DOTS['succ'][n], 'image') for n in ns.NUMBERS]).reshape(-1, D.SIZES[2])
    forms = torch.cat([torch.stack([code(X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]], 'image') for n in ns.NUMBERS]).reshape(-1, D.SIZES[2]) for L in P.LANGS])
    allc = torch.cat([colls, forms])[:, :width]
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(allc), size=(2, 2000))
    a, b = allc[i], allc[j]
    out = {}
    for name, seq in LM.readers(model):
        lin = seq[0].struct
        if name == 'g':
            act = torch.stack([lin(torch.cat([a, b, model.token.weight[h].expand(len(a), -1)], 1)) > 0 for h in range(3)]).any(0).any(0)
        else:
            act = (lin(torch.cat([a, b], 1)) > 0).any(0)
        m = lin.mask[act][:, :2 * width]
        out[name] = {'live_units': int(act.sum()), 'reads': (m[:, :width] + m[:, width:]).sum(0).cpu().numpy().astype(int).tolist()}
    out['coordinates_read'] = sorted(int(k) for k in torch.nonzero(torch.tensor(out['g']['reads']) + torch.tensor(out['P']['reads'])).flatten().tolist())
    return out


def coordinate_organization(enc, model, read):
    probe = AN.coordinate_probe(enc, model)
    avg = lambda idx: float(np.mean([np.mean(np.array(probe[k])[idx]) for k in TRANSFER_SETS])) if len(idx) else None
    return {'all_coordinates': avg(np.arange(len(probe['r2']))), 'coordinates_read': avg(np.array(read, dtype=int))}


def read_frozen(task):
    _, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    train, ev, counts = test_material(seed)
    device = DEV.NAME if DEV.ACTIVE else 'cpu'
    key = SR.key_encoder(seed).to(device).requires_grad_(False)
    addr, t, _ = all_addresses(lambda x: key(x), train, ev, counts)
    code = lambda x: key(x)
    colls = torch.stack([batched(code, P.DOTS['succ'][k + 1]) for k in range(9)])
    return f'frozen-encoder/seed{seed}', {'address': addr, 'value_ceiling': value_ceiling(code, None, seed), 'shape': shape(t[:LOADS[0]].reshape(-1, t.shape[-1]), colls), 'persistent_bytes': 4 * sum(p.numel() for p in key.parameters())}, timing(t0, c0)


def read_host(task):
    config, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    set_up_task(config)
    arm, lam = parse(config)
    train, ev, counts = test_material(seed)
    ck = torch.load(checkpoint_path(config, seed), weights_only=False, map_location='cpu')
    enc, model, _ = LM.setup(config, seed)
    model.load_state_dict(ck['sri']); enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    withheld = {k: tuple(v) for k, v in ck['withheld'].items()}
    for m in (*enc.values(), model):
        m.requires_grad_(False)
    enc['image'].eval(); enc['sequence'].eval()
    code = lambda x: model.proj['image'](enc['image'](x))
    addr, t, q = all_addresses(code, train, ev, counts)
    out = {'address': addr}
    out['address_before_projection'] = all_addresses(enc['image'], train, ev, counts)[0]
    if arm == 'split-recon-bg':  # descriptive: what each part holds
        out['address_read_part'] = {str(L): address_reading(t[:L, :, :16], q[:L, :, :16], counts[L]) for L in LOADS}
        out['address_added_part'] = {str(L): address_reading(t[:L, :, 16:], q[:L, :, 16:], counts[L]) for L in LOADS}
    t8, match = LM.transfer8(enc, model, withheld)
    out['transfer8'] = t8
    out['transfer8_mean'] = float(np.mean(list(t8.values())))
    out['matching'] = match
    out['value_ceiling'] = value_ceiling(code, model, seed)
    colls = torch.stack([batched(code, P.DOTS['succ'][k + 1]) for k in range(9)])
    out['shape'] = shape(t[:LOADS[0]].reshape(-1, t.shape[-1]), colls)
    out['readers'] = reader_use(enc, model, read_width(config))
    out['coordinate_organization'] = coordinate_organization(enc, model, out['readers']['coordinates_read'])
    if lam is not None:
        dec = Decoder(code_width(config))
        dec.load_state_dict(ck['decoder'])
        dec = dec.to(DEV.NAME) if DEV.ACTIVE else dec
        xq = torch.from_numpy(np.ascontiguousarray(ev[:LOADS[0]].reshape(-1, 28, 28), dtype=np.float32))
        mean = torch.from_numpy(np.ascontiguousarray(train[:LOADS[0]].reshape(-1, 28, 28).mean(0), dtype=np.float32))
        with torch.no_grad():
            out['reconstruction_new_symbols'] = {'mse': float(batched(lambda x: (dec(code(x)) - x).pow(2).mean((1, 2)), xq.cpu().numpy()).mean()), 'mean_image_mse': float((xq - mean).pow(2).mean()),
                                                 'value_range': [float(train.min()), float(train.max())]}
    return f'{config}/seed{seed}', out, timing(t0, c0)


def read():
    t0 = time.perf_counter()
    prepare()
    for seed in SEEDS:
        test_material(seed)
    material_s = time.perf_counter() - t0
    res = pool_map(DUR.durable(read_host, RUNS / 'done-read', '{0}-seed{1}'), [(c, s) for c in CONFIGS for s in SEEDS])
    res += pool_map(DUR.durable(read_frozen, RUNS / 'done-read', '{0}-seed{1}'), [('frozen-encoder', s) for s in SEEDS])
    save_json(f'{REPORT}-readings.json', {'loads': LOADS, 'train_renderings': TRAIN_RENDERS, 'eval_renderings': EVAL_RENDERS, 'bar': ADDRESS_BAR, 'results': {k: r for k, r, _ in res}})
    save_json(f'{REPORT}-readings-runtime.json', runtime({'phase': 'readings at block 3', 'wall_seconds': time.perf_counter() - t0, 'setup_seconds': material_s, 'tasks': {k: t for k, _, t in res},
                                                          'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


# ---------------------------------------------------------------- lock

def lock_table(res, seeds=SEEDS):
    """The contract's lock table: per configuration the pilot means, then the route, host, address source and candidate."""
    low = str(LOADS[0])
    mean = lambda f: float(np.mean([f(res[f'{{}}/seed{s}'.format(cfg)], s) for s in seeds]))
    rows = {}
    base = {s: res[f'sparse48/seed{s}'] for s in seeds}
    for cfg in CONFIGS:
        row = {'identity': mean(lambda r, s: r['address'][low]['identity']), 'store': mean(lambda r, s: r['address'][low]['store']),
               'transfer8_mean': mean(lambda r, s: r['transfer8_mean']), 'transfer8_difference': mean(lambda r, s: r['transfer8_mean'] - base[s]['transfer8_mean']),
               'difference_chinese_withheld': mean(lambda r, s: r['transfer8']['chinese_withheld_matching'] - base[s]['transfer8']['chinese_withheld_matching']),
               'difference_devanagari_withheld': mean(lambda r, s: r['transfer8']['devanagari_withheld_matching'] - base[s]['transfer8']['devanagari_withheld_matching']),
               'identity_before_projection': mean(lambda r, s: r['address_before_projection'][low]['identity']), 'store_before_projection': mean(lambda r, s: r['address_before_projection'][low]['store'])}
        row['addresses'] = row['identity'] >= ADDRESS_BAR and row['store'] >= ADDRESS_BAR
        row['holds_on_pilot'] = row['transfer8_difference'] >= -MARGIN
        rows[cfg] = row
    arm_of = lambda c: parse(c)[0]
    passing = [c for c in CONFIGS if c != 'sparse48' and rows[c]['addresses'] and rows[c]['holds_on_pilot']]
    if rows['sparse48']['addresses']:
        passing.append('sparse48')
    pre_ok = rows['sparse48']['identity_before_projection'] >= ADDRESS_BAR and rows['sparse48']['store_before_projection'] >= ADDRESS_BAR
    added = [c for c in CONFIGS if c != 'sparse48']
    if passing:
        host = min(passing, key=lambda c: (code_width(c), ADDED_COST[arm_of(c)], -rows[c]['transfer8_mean'], parse(c)[1] or 0.))
        decision = {'route': 'core', 'host': host, 'address_source': 'its code', 'candidate': host}
    else:
        decision = {'route': 'conjunctive', 'host': 'sparse48', 'address_source': 'its encoder output before projection' if pre_ok else 'frozen encoder', 'candidate': max(added, key=lambda c: (rows[c]['identity'], -CONFIGS.index(c)))}
    return {'rows': rows, 'passing': passing, 'sparse48_before_projection_passes': pre_ok, 'decision': decision}


def lock():
    res = json.load(open(f'{REPORT}-readings.json'))['results']
    out = lock_table(res)
    save_json(f'{REPORT}-lock.json', out)
    print(json.dumps(out['decision']))


# ---------------------------------------------------------------- output

def save_json(path, obj):
    DUR.atomic_write(path, (json.dumps(obj, indent=1, sort_keys=True, default=float) + '\n').encode())


def runtime(extra):
    return {'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')), 'contract_sha256': nc.sha256(CONTRACT),
            'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cpu': platform.processor(), 'workers': WORKERS, 'threads_per_worker': FI.THREADS, 'dtype': 'float32',
            **DEV.info(), 'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'rss_convention': 'ru_maxrss in KiB, per process; children is the largest single child', **extra}


if __name__ == '__main__':
    stage = sys.argv[2] if len(sys.argv) > 2 else 'hosts'
    {'hosts': hosts, 'read': read, 'lock': lock}[stage]()
