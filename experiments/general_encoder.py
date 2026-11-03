"""What training images build a general frozen encoder? (docs/general-frozen-encoder.md.)

Stages (argv[2]):
  stability label-free kind centres through frozen DINOv2: read every set after every later kind
  transfer  layers of already-trained networks read on kinds they never saw; no training
  memory    step 1: six kinds in one memory through one frozen encoder; no training; the sequence pilot's facts and draws
            with GE_DOMAIN=1: one centre per domain (natural, rendered) against one per kind, and two encoders routed by domain
  heldout   step 2: trains on CIFAR-100 with categories held out, then asks what carries over to them (needs `general_encoder_fetch`)
  diet      step 3: one architecture trained on each diet of images, read on kinds outside that diet

Usage: NFI_DEVICE=cuda SC0_WORKERS=2 /data/venvs/sri-cuda/bin/python -m experiments.general_encoder pilot memory
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from experiments import device as DEV
from experiments import dinov2_frozen as DINO
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import natural_memory as NM
from experiments import natural_memory_sequence as SEQ
from experiments import sense_image_free as SIF
from experiments import sense_memory as SM
from experiments import sense_relational as SRL
from experiments import shared_core_stage0 as S0

nc = S0.nc
NAME = 'general-encoder-v1' + os.environ.get('GE_TAG', '')
RUNS = S0.X.ROOT / 'runs' / NAME
REPORT = S0.X.ROOT / 'reports' / NAME
CIFAR100 = S0.X.ROOT / 'runs' / 'external' / 'cifar100' / 'cifar100-full.npz'
SEEDS = tuple(int(s) for s in os.environ.get('GE_SEEDS', '392,393').split(','))
UPDATES = int(os.environ.get('GE_UPDATES', 40000))
N, E = SIF.N, SIF.E
TAUS = (.85, .7)
CENTRINGS = ('none', 'natural', 'kind', 'recent')
CENTRE_IMAGES = 5120
MATCHED = .13
FORM, MEMBER_QUANTILE = 512, .01  # inputs that form a kind centre before it freezes; the share of its own members a centre may fail to claim
GALLERY = 300
WITH_DINO = bool(os.environ.get('GE_DINO'))
DOMAIN_TEST = bool(os.environ.get('GE_DOMAIN'))

# CIFAR-100's coarse labels. People are left out of both sides.
COARSE = ('aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2')
ANIMALS_SEEN = (0, 1, 12, 13, 16)
ANIMALS_HELD = (7, 8, 11, 15)
NOT_ANIMALS = (2, 3, 4, 5, 6, 9, 10, 17, 18, 19)
HELDOUT_ARMS = {'no-animals': NOT_ANIMALS, 'some-animals': NOT_ANIMALS + ANIMALS_SEEN, 'everything': NOT_ANIMALS + ANIMALS_SEEN + ANIMALS_HELD}
SHOTS = (1, 5, 20)
DIETS = ('natural', 'glyphs', 'digits', 'dead-leaves', 'coarse-noise', 'natural+glyphs', 'natural+digits', 'all')
DIGIT_KINDS = ('mnist', 'tree', 'thai_digits', 'banana', 'chinese')


# ---------------------------------------------------------------- pure

def centred(z, c):
    """Subtract a centre, then back to unit length. A code at the centre stays zero, which a read can never match."""
    y = z - c
    return y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-9)


def kind_labels(kind, seed):
    """Category labels of a kind's cues and unseen images: the classes of natural images, the counts of the rendered kinds. The same draw as the sequence pilot's kind_data, so the same images."""
    rng = nc.stream('nm_kind_facts', seed, kind)
    y = NM.data()['y_test'] if kind == 'natural' else np.asarray(S0.X.DATA[(kind, 'eval')][1])
    pick = rng.permutation(len(y))[:2 * N + SEQ.N_CANDIDATES]
    return y[pick[:N]], y[pick[N:2 * N]]


def resemblance(unit_rows, sim, tau, kind_index, unseen_labels, cue_labels):
    """When an unseen image is answered, is the stored fact it reaches related to it? unit_rows: the fact row of each unseen query's nearest unit. cue_labels: (kinds, N) labels of stored facts. Share of answered queries reaching the same kind, and the same kind and category; and, answering all, the same category."""
    k, f = unit_rows // N, unit_rows % N
    same_kind = k == kind_index
    same_class = same_kind & (cue_labels[k, f] == unseen_labels)
    ok = sim >= tau
    answered = max(int(ok.sum()), 1)
    return {'answered_same_kind': float((ok & same_kind).sum() / answered), 'answered_same_class': float((ok & same_class).sum() / answered), 'all_same_kind': float(same_kind.mean()), 'all_same_class': float(same_class.mean()),
            'chance_same_class_in_kind': float(np.mean(cue_labels[kind_index][:, None] == unseen_labels[None]))}


def score(tag_ok, sim, val_ok, unseen_sim, tau):
    """As the sequence pilot: abstentions wrong, at the write threshold, with no threshold, and at matched false answers."""
    out = {'unseen_at_tau': float((unseen_sim >= tau).mean())}
    for name, thr in {'tau': tau, 'answer-all': -2., f'matched-{MATCHED}': NM.matched_threshold(unseen_sim, MATCHED)}.items():
        ok = sim >= thr
        out[name] = {'threshold': float(thr), 'coverage': float(ok.mean()), 'address': float((ok & tag_ok).mean()), 'value': float((ok & val_ok).mean())}
    return out


def nearest_label(K, Q, yk, yq):
    """Share of unit queries whose nearest unit key has their label."""
    return float((yk[(Q @ K.T).argmax(1)] == yq).mean())


def few_shot(z, y, shots, rng, episodes=20):
    """Nearest class mean from `shots` labelled examples per class, on the rest. Unit codes; the mean of many draws."""
    classes = np.unique(y)
    out = []
    for _ in range(episodes):
        keys, query, truth = [], [], []
        for c in classes:
            at = rng.permutation(np.flatnonzero(y == c))
            keys.append(z[at[:shots]].mean(0))
            query.append(z[at[shots:]])
            truth.append(np.full(len(at) - shots, c))
        K = np.stack(keys)
        K = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-9)
        out.append(nearest_label(K, np.concatenate(query), classes, np.concatenate(truth)))
    return float(np.mean(out))


def dead_leaves(n, rng, size=28, discs=60):
    """Procedural images: overlapping discs of random size and grey level, painted back to front, with sizes from a power law, as in dead-leaves models of natural image statistics. (n, size, size) in [0, 1]."""
    out = np.zeros((n, size, size), np.float32)
    yy, xx = np.mgrid[:size, :size]
    for i in range(n):
        img = np.full((size, size), rng.random(), np.float32)
        for _ in range(discs):
            r = (rng.random() * (1 / 2 ** 3 - 1 / (size / 2) ** 3) + 1 / (size / 2) ** 3) ** (-1 / 3)
            cx, cy = rng.random(2) * size
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = rng.random()
        out[i] = img
    return out


def coarse_noise(n, rng, size=28, cell=4):
    grid = rng.random((n, size // cell, size // cell), dtype=np.float32)
    return grid.repeat(cell, 1).repeat(cell, 2)


def domain_of(kind):
    return 'natural' if kind == 'natural' else 'rendered'


def memory_arms():
    """Net, layer, centring. Layers and centrings vary on the natural-trained deep network; the others are read by their key.
    With DOMAIN_TEST, only the arms of the domain test: `domain` centres each input on the mean of its domain, the rendered centre weighting the five rendered kinds equally;
    `routed` writes and reads each domain in its own memory, natural images through the natural-trained deep network (stage 4), rendered ones through the glyph-trained small network."""
    if DOMAIN_TEST:
        arms = [('dinov2-s', lay, c) for lay in ('cls', 'patch') for c in ('none', 'kind', 'domain')] if WITH_DINO else []
        arms += [('natural-deep', 'stage4', c) for c in ('none', 'kind', 'domain')] + [('glyph-small', 'key', c) for c in ('none', 'kind', 'domain')]
        return arms + [('routed', 'split', c) for c in ('none', 'domain')]
    arms = [('natural-deep', lay, c) for lay in ('key', 'stage4', 'stage3') for c in CENTRINGS]
    arms += [(net, 'key', c) for net in ('natural-small', 'glyph-small') for c in CENTRINGS]
    arms += [('untrained-deep', 'key', 'none'), ('pixels', 'code', 'none'), ('pca', 'code', 'none')]
    if WITH_DINO:
        arms += [('dinov2-s', lay, c) for lay in ('cls', 'patch') for c in CENTRINGS]
    return arms


def arm_name(arm):
    return '/'.join(arm)


# ---------------------------------------------------------------- networks and their layers

def deep_layers(m):
    def fn(x):
        h = m.first(x[:, None])
        out = {}
        for i, block in enumerate(m.blocks):
            h = block(h)
            if i in (1, 3, 5, 7):
                out[f'stage{(i + 1) // 2}'] = h.mean((2, 3))
                out[f'stage{(i + 1) // 2}-spatial'] = h.flatten(1)
        out['key'] = m.out(Fn.relu(m.fc(h.mean((2, 3)))))
        return out
    return fn


def small_layers(m):
    def fn(x):
        a = Fn.relu(m.c1(x[:, None]))
        b = Fn.relu(m.c2(a))
        h = Fn.relu(m.fc(b.flatten(1)))
        return {'stage1': a.mean((2, 3)), 'stage1-spatial': a.flatten(1), 'stage2': b.mean((2, 3)), 'stage2-spatial': b.flatten(1), 'hidden': h, 'key': m.out(h)}
    return fn


def run_layers(fn, x, keep=None, size=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), size):
            r = fn(torch.from_numpy(np.ascontiguousarray(x[i:i + size], dtype=np.float32)))
            out.append({k: v.detach().cpu().numpy().astype(np.float32) for k, v in r.items() if keep is None or k in keep})
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def load_trained(config, seed, u):
    arch = NM.CONFIGS[config][0]
    m = NM.build(arch, 'scratch', seed)
    m.load_state_dict(torch.load(NM.encoder_path(config, seed, u), map_location='cpu', weights_only=False)['model'])
    return m.eval().requires_grad_(False)


def trained_nets(seed):
    extra = {'dinov2-s': DINO.layers(DINO.load(DEV.NAME if DEV.ACTIVE else 'cpu'))} if WITH_DINO else {}
    return {**extra, 'natural-deep': deep_layers(load_trained('deep-grey-strong', seed, 40000)), 'untrained-deep': deep_layers(load_trained('deep-grey-strong', seed, 0)),
            'natural-small': small_layers(load_trained('small-scratch-strong', seed, 40000)), 'glyph-small': small_layers(SM.encoder('unlabelled', seed).eval().requires_grad_(False))}


# ---------------------------------------------------------------- transfer: layers on kinds never seen

def transfer_scores(fn, seed):
    """Every layer of one network: glyph identity, across fonts, and finding an image of each kind from a strong view. Reads centre on the batch's mean, so this measures what is there, not what a memory can use yet."""
    train, ev, _ = S0.test_material(seed)
    mat = SIF.load_material(seed)
    protos = np.ascontiguousarray(train[:2 * N], dtype=np.float32)
    q = np.ascontiguousarray(ev[:2 * N], dtype=np.float32)
    mask = mat['font_mask'][:2 * N]
    other = mat['other_font'][:2 * N]
    sym = np.arange(2 * N)
    yq, yo = np.repeat(sym, q.shape[1]), np.repeat(sym[mask], other.shape[1])
    P1, P16 = run_layers(fn, protos[:, 0]), run_layers(fn, protos[:, :16].reshape(-1, 28, 28))
    Q, O = run_layers(fn, q.reshape(-1, 28, 28)), run_layers(fn, other[mask].reshape(-1, 28, 28))
    out = {}
    for lay in Q:
        c = Q[lay].mean(0)
        k16 = P16[lay].reshape(2 * N, 16, -1).mean(1)
        out[lay] = {'glyph_one': nearest_label(centred(P1[lay], c), centred(Q[lay], c), sym, yq), 'glyph_16': nearest_label(centred(k16, c), centred(Q[lay], c), sym, yq), 'glyph_font': nearest_label(centred(k16, c), centred(O[lay], c), sym, yo)}
    rng = nc.stream('ge_transfer_views', seed)
    for kind in SEQ.KINDS:
        held = NM.grey28(NM.data()['x_test']) if kind == 'natural' else np.ascontiguousarray(S0.X.DATA[(kind, 'eval')][0], dtype=np.float32)
        g = held[rng.permutation(len(held))[:GALLERY]]
        G, V = run_layers(fn, g), run_layers(fn, NM.view_arrays(g, rng, NM.VIEWS['strong']))
        for lay in G:
            c = G[lay].mean(0)
            out[lay][f'strong_{kind}'] = nearest_label(centred(G[lay], c), centred(V[lay], c), np.arange(GALLERY), np.arange(GALLERY))
    return out


def transfer_task(task):
    seed, = task
    t0, c0 = time.perf_counter(), time.process_time()
    S0.set_up_task('sparse48')
    pixels = lambda x: {'pixels': x.flatten(1)}
    res = {net: transfer_scores(fn, seed) for net, fn in {**trained_nets(seed), 'pixels': pixels}.items()}
    return f'{seed}', {'seed': seed, 'nets': res}, S0.timing(t0, c0)


def transfer():
    t0 = time.perf_counter()
    S0.prepare()
    NM.data()
    res = S0.pool_map(DUR.durable(transfer_task, RUNS / 'done-transfer', '{0}'), [(s,) for s in SEEDS])
    S0.save_json(f'{REPORT}-transfer.json', {'status': 'exploratory; read only; networks trained before this stage', 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-transfer-runtime.json', S0.runtime({'phase': 'layers on kinds never seen', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


# ---------------------------------------------------------------- step 1: memory through one frozen encoder

def memory_task(task):
    seed, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    S0.set_up_task('sparse48')
    kinds = SEQ.kind_order(seed, order)
    data = {k: SEQ.kind_data(k, seed) for k in kinds}
    index = {k: i for i, k in enumerate(kinds)}
    wrng = nc.stream('nm_seq_writes', seed, order)  # the sequence pilot's streams, so its fixed arms must read the same
    qrng = nc.stream('nm_seq_queries', seed)
    q = {k: {v: NM.view_arrays(data[k]['cues'], qrng, c) for v, c in NM.VIEWS.items()} for k in SEQ.KINDS}
    qu = {k: {v: NM.view_arrays(data[k]['unseen'], qrng, c) for v, c in NM.VIEWS.items()} for k in SEQ.KINDS}
    arms = memory_arms()
    nets = trained_nets(seed)
    fx = {k: NM.fixed_codes(k, 'small', seed) for k in ('pixels', 'pca')}
    layers_of = lambda net: sorted({a[1] for a in arms if a[0] == net})
    features = lambda net, x: {'code': fx[net][0](x)} if net in fx else run_layers(nets[net], x, keep=layers_of(net))
    all_nets = sorted({a[0] for a in arms} - {'routed'})
    route = lambda arm, kind: (('natural-deep', 'stage4') if kind == 'natural' else ('glyph-small', 'key')) + (arm[2],) if arm[0] == 'routed' else arm

    crng = nc.stream('ge_centres', seed)
    mean_of = lambda net, x: {lay: f.mean(0) for lay, f in features(net, x).items()}
    sample = lambda pool: np.ascontiguousarray(pool[crng.permutation(len(pool))[:CENTRE_IMAGES]], dtype=np.float32)
    samples = {k: sample(data[k]['pool']) for k in kinds}  # stand-ins for inputs passing at the time; discarded; only the mean is kept
    centre_kind = {net: {k: mean_of(net, samples[k]) for k in kinds} for net in all_nets}
    centre_natural = {net: centre_kind[net]['natural'] for net in all_nets}
    centre_domain = {net: {'natural': centre_natural[net], 'rendered': {lay: np.mean([centre_kind[net][k][lay] for k in kinds if k != 'natural'], 0) for lay in centre_natural[net]}} for net in all_nets}

    def centre(arm, kind, f, batch_mean=False):
        net, lay, how = arm
        if how == 'none':
            return NM.unit(f)
        if how == 'natural':
            return centred(f, centre_natural[net][lay])
        if how == 'recent' and batch_mean:
            return centred(f, f.mean(0))
        if how == 'domain':
            return centred(f, centre_domain[net][domain_of(kind)][lay])
        return centred(f, centre_kind[net][kind][lay])

    facts = len(SEQ.KINDS) * N
    width_of = lambda a: run_layers(nets[a[0]], np.zeros((1, 28, 28), np.float32), keep=(a[1],))[a[1]].shape[1] if a[0] in nets else fx[a[0]][1]
    tagged = lambda a, tau: NM.Tagged(SEQ.CAPACITY, tau, width_of(a), width_of(a), facts)
    # a routed arm keeps one memory per domain, each with the full capacity; units stay far below it here
    mems = {(a, tau): {d: tagged(route(a, d), tau) for d in ('natural', 'rendered')} if a[0] == 'routed' else tagged(a, tau) for a in arms for tau in (TAUS if a[0] != 'pixels' else (TAUS[0],))}
    mem_of = lambda m, kind: m[domain_of(kind)] if isinstance(m, dict) else m

    write_wall = 0.
    for kind in kinds:
        t = time.perf_counter()
        plan = NM.plan_writes(wrng, np.arange(N), SEQ.EXPOSURES, SEQ.BLOCK)
        cues, targets, rows = [], [], []
        for batch in plan:
            if not len(batch):
                continue
            cues.append(NM.view_arrays(data[kind]['cues'][batch], wrng, NM.VIEWS['mild']))
            targets.append(NM.view_arrays(data[kind]['targets'][batch], wrng, NM.VIEWS['mild']))
            rows.append(SEQ.fact_row(index[kind], batch))
        cues, targets, rows = np.concatenate(cues), np.concatenate(targets), np.concatenate(rows)
        fc, ft = {n: features(n, cues) for n in all_nets}, {n: features(n, targets) for n in all_nets}
        for (arm, tau), mem in mems.items():
            arm, mem = route(arm, kind), mem_of(mem, kind)
            net, lay, _ = arm
            z, v = centre(arm, kind, fc[net][lay]), centre(arm, kind, ft[net][lay])
            for i in range(len(rows)):
                mem.write_tagged(z[i], v[i], rows[i])
        write_wall += time.perf_counter() - t

    reads = {}
    truth = np.arange(N)
    labels = {k: kind_labels(k, seed) for k in kinds}
    cue_labels = np.stack([labels[k][0] for k in kinds])
    for kind in kinds:
        rows = SEQ.fact_row(index[kind], truth)
        fcand = {n: features(n, data[kind]['targets']) for n in all_nets}
        for view in NM.VIEWS:
            fq, fu = {n: features(n, q[kind][view]) for n in all_nets}, {n: features(n, qu[kind][view]) for n in all_nets}
            for (name, tau), mem in mems.items():
                arm, mem = route(name, kind), mem_of(mem, kind)
                net, lay, _ = arm
                Z, Zu, Zc = centre(arm, kind, fq[net][lay], True), centre(arm, kind, fu[net][lay], True), centre(arm, kind, fcand[net][lay], True)
                h = mem.h
                i, s = mem.nearest(Z)
                iu, su = mem.nearest(Zu)
                val = mem.u[:h] / mem.n[:h, None]
                reads[f'{arm_name(name)}|{tau}|{kind}-{view}'] = {**score(mem.tags[:h][i].argmax(1) == rows, s, (val[i] @ Zc.T).argmax(1) == truth, su, tau), 'units': h, 'payload_bytes': mem.payload_bytes(),
                                                                  'resemblance': resemblance(mem.tags[:h][iu].argmax(1), su, tau, index[kind], labels[kind][1], cue_labels)}
    return f'{seed}/{order}', {'reads': reads, 'kinds': kinds, 'seed': seed, 'order': order, 'write_wall_seconds': write_wall}, S0.timing(t0, c0)


def memory():
    t0 = time.perf_counter()
    S0.prepare()
    NM.data()
    res = S0.pool_map(DUR.durable(memory_task, RUNS / 'done-memory', '{0}-{1}'), [(s, o) for s in SEEDS for o in SEQ.ORDERS])
    status = 'exploratory, pilot seeds, an order of kinds and its reverse; no training; one memory through one frozen encoder; no image held'
    if DOMAIN_TEST:
        status += '; domain test: one centre per domain against one per kind, and two encoders routed by domain into a memory each'
    S0.save_json(f'{REPORT}-memory.json', {'status': status, 'arms': memory_arms(), 'taus': TAUS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-memory-runtime.json', S0.runtime({'phase': 'writes and reads through frozen encoders', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


# ---------------------------------------------------------------- label-free kind centres

class KindCentres:
    """Kind centres from passing inputs, with no labels. An input joins the nearest frozen centre whose own threshold it reaches; otherwise it joins the centre being formed, or starts one. A forming centre freezes after FORM inputs, keeping as its threshold the MEMBER_QUANTILE of its members' similarity to it. Assumes a new kind arrives for a while at a time, as in these streams. Raw codes in, unit similarities inside."""
    def __init__(self, form=FORM, q=MEMBER_QUANTILE):
        self.form, self.q = form, q
        self.centres, self.thresholds, self.forming = [], [], []

    def _near(self, z):
        if not self.centres:
            return -1, -2.
        C = np.stack(self.centres)
        s = NM.unit(C) @ NM.unit(z[None])[0]
        i = int(s.argmax())
        return i, float(s[i])

    def observe(self, z):
        i, s = self._near(z)
        if i >= 0 and s >= self.thresholds[i]:
            return i
        self.forming.append(z)
        if len(self.forming) >= self.form:
            members = np.stack(self.forming)
            c = members.mean(0)
            self.centres.append(c)
            self.thresholds.append(float(np.quantile(NM.unit(members) @ NM.unit(c[None])[0], self.q)))
            self.forming = []
        return len(self.centres) - 1 if not self.forming else -1

    def assign(self, Z):
        """Nearest frozen centre of each raw code, whether or not it reaches that centre's threshold."""
        C = NM.unit(np.stack(self.centres))
        return (NM.unit(Z) @ C.T).argmax(1)

    def centre(self, Z):
        """Unit codes, each centred on its own nearest centre."""
        a = self.assign(Z)
        return centred(Z, np.stack(self.centres)[a]), a


def stability_task(task):
    seed, order = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    S0.set_up_task('sparse48')
    kinds = SEQ.kind_order(seed, order)
    data = {k: SEQ.kind_data(k, seed) for k in kinds}
    index = {k: i for i, k in enumerate(kinds)}
    wrng = nc.stream('nm_seq_writes', seed, order)
    qrng = nc.stream('nm_seq_queries', seed)
    q = {k: {v: NM.view_arrays(data[k]['cues'], qrng, c) for v, c in NM.VIEWS.items()} for k in SEQ.KINDS}
    qu = {k: {v: NM.view_arrays(data[k]['unseen'], qrng, c) for v, c in NM.VIEWS.items()} for k in SEQ.KINDS}
    dino = DINO.layers(DINO.load(DEV.NAME if DEV.ACTIVE else 'cpu'))
    feats = lambda x: run_layers(dino, x, size=64)
    crng, prng = nc.stream('ge_centres', seed), nc.stream('ge_passing', seed, order)
    told = {k: {lay: f.mean(0) for lay, f in feats(np.ascontiguousarray(data[k]['pool'][crng.permutation(len(data[k]['pool']))[:CENTRE_IMAGES]], dtype=np.float32)).items()} for k in kinds}
    layers_ = ('cls', 'patch')
    found = {lay: KindCentres() for lay in layers_}
    arms = [(lay, how) for lay in layers_ for how in ('none', 'told', 'found')]
    facts = len(SEQ.KINDS) * N
    mems = {a: NM.Tagged(SEQ.CAPACITY, TAUS[0], DINO.DIM, DINO.DIM, facts) for a in arms}
    labels = {k: kind_labels(k, seed) for k in kinds}
    cue_labels = np.stack([labels[k][0] for k in kinds])

    def code(arm, kind, f):
        lay, how = arm
        if how == 'none':
            return NM.unit(f), None
        if how == 'told':
            return centred(f, told[kind][lay]), None
        return found[lay].centre(f)

    cache, reads, centre_log, taught = {}, {}, [], []
    truth = np.arange(N)
    for kind in kinds:
        passing = np.ascontiguousarray(data[kind]['pool'][prng.permutation(len(data[kind]['pool']))[:2 * FORM]], dtype=np.float32)  # inputs passing before teaching; discarded
        fp = feats(passing)
        for lay in layers_:
            for z in fp[lay]:
                found[lay].observe(z)
        centre_log.append({lay: {'centres': len(found[lay].centres), 'forming': len(found[lay].forming)} for lay in layers_})
        plan = NM.plan_writes(wrng, np.arange(N), SEQ.EXPOSURES, SEQ.BLOCK)
        cues, targets, rows = [], [], []
        for batch in plan:
            if len(batch):
                cues.append(NM.view_arrays(data[kind]['cues'][batch], wrng, NM.VIEWS['mild']))
                targets.append(NM.view_arrays(data[kind]['targets'][batch], wrng, NM.VIEWS['mild']))
                rows.append(SEQ.fact_row(index[kind], batch))
        fc, ft, rows = feats(np.concatenate(cues)), feats(np.concatenate(targets)), np.concatenate(rows)
        for arm in arms:
            z, _ = code(arm, kind, fc[arm[0]])
            v, _ = code(arm, kind, ft[arm[0]])
            for i in range(len(rows)):
                mems[arm].write_tagged(z[i], v[i], rows[i])
        taught.append(kind)
        stage = {}
        for k in taught:
            if k not in cache:
                cache[k] = {'cand': feats(data[k]['targets']), **{f'q-{v}': feats(q[k][v]) for v in NM.VIEWS}, **{f'u-{v}': feats(qu[k][v]) for v in NM.VIEWS}}
            rows_k = SEQ.fact_row(index[k], truth)
            for v in NM.VIEWS:
                for arm in arms:
                    lay = arm[0]
                    mem = mems[arm]
                    Z, a = code(arm, k, cache[k][f'q-{v}'][lay])
                    Zu, au = code(arm, k, cache[k][f'u-{v}'][lay])
                    Zc, _ = code(arm, k, cache[k]['cand'][lay])
                    h = mem.h
                    i, sim = mem.nearest(Z)
                    iu, su = mem.nearest(Zu)
                    val = mem.u[:h] / mem.n[:h, None]
                    rec = {**score(mem.tags[:h][i].argmax(1) == rows_k, sim, (val[i] @ Zc.T).argmax(1) == truth, su, TAUS[0]), 'units': h,
                           'resemblance': resemblance(mem.tags[:h][iu].argmax(1), su, TAUS[0], index[k], labels[k][1], cue_labels)}
                    if a is not None:
                        rec['centre_of_queries'] = np.bincount(a, minlength=len(found[lay].centres)).tolist()
                    stage[f'{lay}/{arm[1]}|{k}-{v}'] = rec
        reads[f'{len(taught)}:{kind}'] = stage
    res = {'reads': reads, 'kinds': kinds, 'centres': centre_log, 'thresholds': {lay: found[lay].thresholds for lay in layers_}, 'seed': seed, 'order': order}
    return f'{seed}/{order}', res, S0.timing(t0, c0)


def stability():
    t0 = time.perf_counter()
    S0.prepare()
    NM.data()
    res = S0.pool_map(DUR.durable(stability_task, RUNS / 'done-stability', '{0}-{1}'), [(s, o) for s in SEEDS for o in SEQ.ORDERS])
    S0.save_json(f'{REPORT}-stability.json', {'status': 'exploratory, pilot seeds, both orders; frozen DINOv2; kind centres found with no labels from passing inputs; every set read after every later kind', 'form': FORM, 'member_quantile': MEMBER_QUANTILE, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-stability-runtime.json', S0.runtime({'phase': 'frozen DINOv2, label-free centres, writes and repeated reads', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


# ---------------------------------------------------------------- training on a chosen diet, for steps 2 and 3

def fit(next_images, arch, seed, tag, path):
    """Contrastive training on what next_images(rng, n) returns. The encoder stage's settings. Saves the final weights."""
    model = NM.build(arch, 'scratch', seed)
    opt = torch.optim.Adam(model.parameters(), NM.LR)
    rng = nc.stream('ge_stream', seed, tag)
    n = NM.BASES[arch]
    partner = NM.partners(n)
    losses = []
    for _ in range(UPDATES):
        base = torch.from_numpy(np.ascontiguousarray(next_images(rng, n), dtype=np.float32))
        x = torch.stack([NM.stream_views(base, rng, 'strong'), NM.stream_views(base, rng, 'strong')], 1)
        x = x.reshape(-1, *x.shape[2:])
        z = model(x)
        loss = SRL.info_nce(z @ z.T / NM.T, partner)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    torch.save(DEV.to_cpu(model.state_dict()), path)
    return model.eval().requires_grad_(False), float(np.mean(losses[-200:]))


# ---------------------------------------------------------------- step 2: categories held out

def cifar100():
    z = np.load(CIFAR100)
    return {k: z[k] for k in z.files}


def heldout_codes(m, x):
    return NM.unit(NM.encode_fn(m)(NM.colour32(x)))


def heldout_task(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    d = cifar100()
    allowed = np.flatnonzero(np.isin(d['coarse_train'], HELDOUT_ARMS[arm]))
    path = RUNS / 'heldout' / f'{arm}-seed{seed}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    model, loss = fit(lambda rng, n: NM.colour32(d['x_train'][allowed[rng.integers(len(allowed), size=n)]]), 'deep-colour', seed, f'heldout-{arm}', path)
    rng = nc.stream('ge_few_shot', seed, arm)
    test = np.isin(d['coarse_test'], ANIMALS_HELD)
    with torch.no_grad():
        z = heldout_codes(model, d['x_test'][test])
        z_all = heldout_codes(model, d['x_test'])
    z = z - z.mean(0)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    out = {'fine': {str(s): few_shot(z, d['fine_test'][test], s, rng) for s in SHOTS}, 'coarse': {str(s): few_shot(z, d['coarse_test'][test], s, rng) for s in SHOTS}}
    # "understands something": animal or not, from a few labelled images, on all test images whose categories this network never trained on or trained on
    is_animal = np.isin(d['coarse_test'], ANIMALS_SEEN + ANIMALS_HELD).astype(np.int64)
    keep = d['coarse_test'] != 14
    out['animal_or_not'] = {str(s): few_shot(z_all[keep], is_animal[keep], s, rng) for s in SHOTS}
    out['loss'] = loss
    return f'{arm}/{seed}', {'arm': arm, 'seed': seed, 'scores': out}, S0.timing(t0, c0)


def heldout():
    t0 = time.perf_counter()
    res = S0.pool_map(DUR.durable(heldout_task, RUNS / 'done-heldout', '{0}-{1}'), [(a, s) for a in HELDOUT_ARMS for s in SEEDS])
    S0.save_json(f'{REPORT}-heldout.json', {'status': 'exploratory; CIFAR-100 with categories held out of training', 'arms': HELDOUT_ARMS, 'held': ANIMALS_HELD, 'names': COARSE, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-heldout-runtime.json', S0.runtime({'phase': 'fits and few-shot reads', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


# ---------------------------------------------------------------- step 3: diets

def diet_images(diet, seed):
    """A function giving n images of the diet, grey 28 x 28. Mixed diets draw each image's source at random. Test-split images are never used."""
    natural = NM.grey28(NM.data()['x_train'])
    glyphs = S0.background_images(seed).reshape(-1, 28, 28)
    digits = np.concatenate([S0.X.DATA[(k, 'train')][0] for k in DIGIT_KINDS]).astype(np.float32)
    leaves = dead_leaves(20000, nc.stream('ge_leaves', seed))
    source = {'natural': natural, 'glyphs': glyphs, 'digits': digits, 'dead-leaves': leaves}
    def draw(pool, rng, n):
        return pool[rng.integers(len(pool), size=n)]
    parts = {'natural+glyphs': ('natural', 'glyphs'), 'natural+digits': ('natural', 'digits'), 'all': ('natural', 'glyphs', 'digits', 'dead-leaves')}
    if diet == 'coarse-noise':
        return lambda rng, n: coarse_noise(n, rng)
    if diet in source:
        return lambda rng, n: draw(source[diet], rng, n)
    def mixed(rng, n):
        pick = rng.integers(len(parts[diet]), size=n)
        return np.stack([draw(source[parts[diet][int(p)]], rng, 1)[0] for p in pick])
    return mixed


def diet_task(task):
    diet, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    S0.set_up_task('sparse48')
    path = RUNS / 'diets' / f'{diet}-seed{seed}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    model, loss = fit(diet_images(diet, seed), 'deep-grey', seed, f'diet-{diet}', path)
    return f'{diet}/{seed}', {'diet': diet, 'seed': seed, 'loss': loss, 'scores': transfer_scores(deep_layers(model), seed)}, S0.timing(t0, c0)


def diet():
    t0 = time.perf_counter()
    S0.prepare()
    NM.data()
    res = S0.pool_map(DUR.durable(diet_task, RUNS / 'done-diet', '{0}-{1}'), [(d, s) for d in DIETS for s in SEEDS])
    S0.save_json(f'{REPORT}-diet.json', {'status': 'exploratory; one architecture on each diet of training images, read on all kinds', 'diets': DIETS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-diet-runtime.json', S0.runtime({'phase': 'fits on diets and transfer reads', 'wall_seconds': time.perf_counter() - t0, 'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'transfer': transfer, 'memory': memory, 'stability': stability, 'heldout': heldout, 'diet': diet}[sys.argv[2]]()
