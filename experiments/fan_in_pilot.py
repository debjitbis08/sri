"""Fan-in pressure pilot, stage 1 (docs/fan-in-pressure-proposal.md): the sequence pilot network with hard fan-in limits.

Each limited unit reads exactly k units of the layer below, through a fixed binary mask held for the whole run. Sites:
  encoder  the front ends' `out` (128 -> 64) and the projections `proj` (64 -> 16), per sense;
  readers  the first layers of the comparator g and the predictor P: each hidden unit reads the same k of the 16 code
           dimensions of both operands; g's relation-token columns stay fully connected.
Fan-in fraction phi = 1/4 or 1/16 (k = phi * inputs; for readers, of the 16 code dimensions). Masks are balanced (every
input is read by floor or ceil of m k / n units) and drawn from their own named streams, so batches and draws match the
sequence pilot by seed. Surviving weights (code columns for readers) are rescaled by 1/sqrt(phi) after the standard
initialization. 'dense' converts no layer and must reproduce the sequence pilot's D-varied+rehearsal runs exactly.

Training and evaluation are the sequence pilot's, D-varied+rehearsal only; the run's arm name is the configuration.

Confirmation ('confirm', seeds 200-209; proposal, "Confirmation after stage 1"): readers at one code dimension per
operand, with random tied, random untied, learned tied and learned untied selection, against dense. Learned selection
reads exactly the argmax dimension in the forward pass and trains the selection logits through a straight-through
softmax; it starts from the random arm's selection and weights for the same seed.

Learned grouping ('rigl', seeds 210-219; proposal, "Learned grouping"): prune and regrow (RigL) on the encoder at
phi = 1/16 and on untied readers at 4 dimensions per operand, against the same limits with fixed random masks. Every
DT updates until END, each unit drops its smallest-magnitude active connections and grows the inactive ones with the
largest gradient magnitude, keeping exactly k per unit (per operand for readers).

Speed settings: NFI_THREADS (torch threads per worker, default 1) and NFI_CHANNELS_LAST=1 (channels-last image encoder
activations; same function, different rounding).

'rigl1' (seeds 220-229): RigL on untied readers at one dimension per operand, against the same random limit and dense.

'fixed' (seeds 230-239; proposal, "Replication with a fixed coordinate"): rigl1's arms again, plus readers whose units
read one coordinate d of both operands, chosen per seed without data: all units ('fixed') or 48 of 64 per reader, the
rest keeping the random arm's masks ('fixed75').

'timing' (seeds 230-239 again, paired with 'fixed'; proposal, "Timing or choice of the coordinate"): a 2 x 2 of when
readers concentrate (from the start, or gradually on RigL's schedule) and which coordinate (drawn without data, or the
one RigL converged on for the same seed in 'fixed'). Gradual arms start from the random arm's masks; at each of RigL's
connectivity updates a growing share of units, in a fixed random order, moves both operands onto the coordinate, keeping
their weights (moved to the new columns, Adam moments reset). dense, RigL and fixed are rerun and must reproduce 'fixed'.

'reset' (seeds 230-239 again; proposal, "Resets, rewiring and predictor capacity"): what in RigL at one dimension per
operand produces its transfer. RigL on P only or g only; RigL carrying the old weight to the new coordinate instead of
restarting it at zero; random rewiring with zero restart at RigL's mean switch rate; and the random arm with as many
reader units silenced from the start as RigL leaves dead. dense, random and RigL are rerun and must reproduce 'fixed'.

'erasure' (seeds 230-239 again; proposal, "Erasure, moving, sparsity and rate"): which part of random rewiring gives its
transfer. Connections reset to zero in place, without moving; random moves that keep their weight; dense readers whose
code weights reset in place at the same per-connection rate; random rewiring at half and twice the rate. dense, random
and random rewiring are rerun and must reproduce 'reset'.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.fan_in_pilot pilot [confirm|rigl|rigl1]
Then: .venv/bin/python -m experiments.fan_in_pilot_analysis pilot [confirm|rigl|rigl1]
"""
import json
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import generic_relations as G
from experiments import transition_pilot as TP
from experiments import sequence_pilot as SP
from experiments import device as DEV
from experiments import durable as DUR

nc = X.nc
STAGE = sys.argv[2] if len(sys.argv) > 2 else 'stage1'
NAME = 'fan-in-pilot-v1-' + X.MODE + ('' if STAGE == 'stage1' else f'-{STAGE}') + ('-smoke' if X.SMOKE else '') + os.environ.get('NFI_TAG', '')
ONLY = os.environ.get('NFI_ONLY')  # 'config:seed,...': run only these configurations and seeds (timing and checks); use with NFI_TAG
BASE_ARM = 'D-varied+rehearsal'
# (encoder 1/phi, readers 1/phi, reader selection); None = dense
STAGE1 = {'dense': (None, None, None), 'enc-q4': (4, None, None), 'enc-q16': (16, None, None), 'read-q4': (None, 4, 'tied'),
          'read-q16': (None, 16, 'tied'), 'both-q4': (4, 4, 'tied'), 'both-q16': (16, 16, 'tied')}
CONFIRM = {'dense': (None, None, None), 'read-q16': (None, 16, 'tied'), 'read-q16-untied': (None, 16, 'untied'),
           'learn-q16-tied': (None, 16, 'learn-tied'), 'learn-q16-untied': (None, 16, 'learn-untied')}
RIGL = {'dense': (None, None, None), 'enc-q16': (16, None, None), 'enc-q16-rigl': (16, None, None),
        'read-q4-untied': (None, 4, 'untied'), 'read-q4-untied-rigl': (None, 4, 'untied')}
RIGL1 = {'dense': (None, None, None), 'read-q16-untied': (None, 16, 'untied'), 'read-q16-untied-rigl': (None, 16, 'untied')}
FIXED = {**RIGL1, 'read-q16-fixed': (None, 16, 'fixed'), 'read-q16-fixed75': (None, 16, 'fixed75')}
TIMING = {'dense': (None, None, None), 'read-q16-untied-rigl': (None, 16, 'untied'), 'read-q16-fixed': (None, 16, 'fixed'),
          'read-q16-gradual': (None, 16, 'gradual'), 'read-q16-fixed-rigld': (None, 16, 'fixed-rigld'),
          'read-q16-gradual-rigld': (None, 16, 'gradual-rigld')}
FIXED_SHARE = {'fixed': 1., 'fixed75': .75, 'fixed-rigld': 1.}  # fraction of reader units moved onto the coordinate at the start
GRADUAL = ('gradual', 'gradual-rigld')
# The live g units' most-read dimension of read-q16-untied-rigl, seeds 230-239, in reports/fan-in-pilot-v1-fixed-comparison.json.gz
RIGL_COORDINATE = dict(zip(range(230, 240), (5, 1, 15, 7, 13, 12, 7, 2, 9, 2)))
RESET = {'dense': (None, None, None), 'read-q16-untied': (None, 16, 'untied'), 'read-q16-untied-rigl': (None, 16, 'untied'),
         'read-q16-untied-riglP': (None, 16, 'untied'), 'read-q16-untied-riglg': (None, 16, 'untied'),
         'read-q16-untied-riglcarry': (None, 16, 'untied'), 'read-q16-untied-rewire': (None, 16, 'untied'),
         'read-q16-untied-dead': (None, 16, 'untied')}
RIGL_BASE = {'enc-q16-rigl': 'enc-q16', 'read-q4-untied-rigl': 'read-q4-untied', 'read-q16-untied-rigl': 'read-q16-untied',
             'read-q16-untied-riglP': 'read-q16-untied', 'read-q16-untied-riglg': 'read-q16-untied',
             'read-q16-untied-riglcarry': 'read-q16-untied', 'read-q16-untied-rewire': 'read-q16-untied'}  # RigL arm -> random arm with the same start
RIGL_SITES = {'read-q16-untied-riglP': ('P/first',), 'read-q16-untied-riglg': ('g/first',)}  # default: every limited layer
RIGL_MODE = {'read-q16-untied-riglcarry': 'carry', 'read-q16-untied-rewire': 'random'}  # default: 'zero' (RigL)
# Random rewiring: per connection and connectivity update, the probability of moving to another coordinate, matched to
# read-q16-untied-rigl's mean grown connections in 'fixed' (g 2054.7, P 1322.0 per run) over 45 updates x 128 connections
REWIRE_P = {'g/first': 2054.7 / (45 * 128), 'P/first': 1322.0 / (45 * 128)}
DEAD_UNITS = {'g': 7, 'P': 32}  # read-q16-untied-dead: RigL's mean units dead for every code input in 'fixed', rounded
DEAD_BIAS = -1.
ERASURE = {'dense': (None, None, None), 'read-q16-untied': (None, 16, 'untied'), 'read-q16-untied-rewire': (None, 16, 'untied'),
           'read-q16-untied-inplace': (None, 16, 'untied'), 'read-q16-untied-rewirecarry': (None, 16, 'untied'),
           'dense-inplace': (None, None, None), 'read-q16-untied-rewire-half': (None, 16, 'untied'),
           'read-q16-untied-rewire-double': (None, 16, 'untied')}
RIGL_BASE.update({c: 'read-q16-untied' for c in ERASURE if c.startswith('read-q16-untied-')})
RIGL_BASE['dense-inplace'] = 'dense'
RIGL_MODE.update({'read-q16-untied-inplace': 'inplace', 'dense-inplace': 'inplace', 'read-q16-untied-rewirecarry': 'random-carry',
                  'read-q16-untied-rewire-half': 'random', 'read-q16-untied-rewire-double': 'random'})
RATE = {'read-q16-untied-rewire-half': .5, 'read-q16-untied-rewire-double': 2.}  # multiplies REWIRE_P
DENSE_READERS = ('dense-inplace',)  # dense readers held as MaskedLinear with an all-ones mask (same function, exactly)
CONFIGS = {'stage1': STAGE1, 'confirm': CONFIRM, 'rigl': RIGL, 'rigl1': RIGL1, 'fixed': FIXED, 'timing': TIMING, 'reset': RESET, 'erasure': ERASURE}.get(STAGE, {})  # other runners (limited_memory) import this module with their own stage names
CONFIRM_SEEDS = tuple(range(*{'confirm': (200, 210), 'rigl': (210, 220), 'rigl1': (220, 230), 'fixed': (230, 240), 'timing': (230, 240), 'reset': (230, 240), 'erasure': (230, 240)}.get(STAGE, (0, 0))))
DT, ALPHA, END_FRACTION = (10 if X.SMOKE else 100), .3, .75  # RigL: update interval, initial drop fraction, end of updates (fraction of training)
RANDOM_OF = {'learn-tied': ('read-q16', 'tied'), 'learn-untied': ('read-q16-untied', 'untied')}  # learned arm's starting selection
START_OF = {**RANDOM_OF, **{m: ('read-q16-untied', 'untied') for m in ('fixed', 'fixed75', 'fixed-rigld', 'gradual', 'gradual-rigld')}}  # masks an arm starts from
CODE = 16
THREADS = int(os.environ.get('NFI_THREADS', '1'))  # torch threads per worker
CHANNELS_LAST = os.environ.get('NFI_CHANNELS_LAST') == '1'  # image encoder convolutions in channels-last layout


class MaskedLinear(nn.Linear):
    """nn.Linear whose weight is multiplied by a binary mask (buffer `mask`, shape of the weight). While `collect` is set,
    a zero tensor `probe` is added to the masked weight so that probe.grad holds the dense gradient (RigL's grow score)."""

    def forward(self, x):
        w = self.weight * self.mask
        if getattr(self, 'collect', False):
            w = w + self.probe
        return F.linear(x, w, self.bias)


def balanced_subsets(rng, m, n, k):
    """m subsets of size k from range(n); every index is used floor(mk/n) or ceil(mk/n) times (ties broken at random)."""
    used, out = np.zeros(n, int), []
    for _ in range(m):
        order = np.lexsort((rng.random(n), used))  # least used first, random among equals
        s = np.sort(order[:k])
        used[s] += 1
        out.append(s)
    return out


def limit(lin, mask, scale_cols):
    """Convert lin to MaskedLinear in place; rescale its surviving weights in scale_cols by the inverse sqrt of the kept fraction."""
    lin.__class__ = MaskedLinear
    lin.register_buffer('mask', mask)
    with torch.no_grad():
        kept = mask[:, scale_cols].sum(1, keepdim=True) / len(scale_cols)
        lin.weight[:, scale_cols] *= kept.rsqrt()


def encoder_mask(lin, q, rng):
    m, n = lin.weight.shape
    mask = torch.zeros(m, n)
    for j, s in enumerate(balanced_subsets(rng, m, n, n // q)):
        mask[j, s] = 1.
    limit(lin, mask, np.arange(n))


def reader_mask(lin, q, rng, rng_b=None):
    """First layer of g (code a, code b, token) or P (code prev, code cur): k code dims per operand, the same for both
    operands (tied), or drawn independently for the second operand from rng_b (untied)."""
    m, n = lin.weight.shape
    mask = torch.ones(m, n)
    mask[:, :2 * CODE] = 0.
    sa = balanced_subsets(rng, m, CODE, CODE // q)
    sb = sa if rng_b is None else balanced_subsets(rng_b, m, CODE, CODE // q)
    for j in range(m):
        mask[j, sa[j]] = 1.
        mask[j, CODE + sb[j]] = 1.
    limit(lin, mask, np.arange(2 * CODE))


def fixed_coordinate(seed, mode='fixed'):
    if mode.endswith('rigld'):
        return RIGL_COORDINATE[seed]
    return int(nc.stream('fanin_mask', seed, 'fixed-coordinate').integers(CODE))


def gradual_order(seed, name, m):
    return nc.stream('fanin_mask', seed, 'gradual-order', name).permutation(m)


def fix_rows(lin, d, share, rng):
    """Move round(share * units) reader units, chosen by rng, onto code dimension d of both operands. The kept fraction
    per unit is unchanged, so the rescaling applied by reader_mask still holds."""
    m = lin.mask.shape[0]
    rows = torch.as_tensor(np.sort(rng.permutation(m)[:round(share * m)]))
    lin.mask[rows, :2 * CODE] = 0.
    lin.mask[rows, d] = 1.
    lin.mask[rows, CODE + d] = 1.


def straight_through(logits):
    soft = logits.softmax(1)
    hard = F.one_hot(logits.argmax(1), logits.shape[1]).to(soft)
    return hard + soft - soft.detach()


class SelectLinear(nn.Module):
    """First reader layer with one learned code dimension per operand and hidden unit: exactly the argmax dimension is
    read in the forward pass; the selection logits learn through a straight-through softmax. Built from a MaskedLinear
    with one dimension per operand, whose selection and weights it starts from (logits 0, +1 on the selected dimension)."""

    def __init__(self, lin, tied):
        super().__init__()
        W, M = lin.weight.detach(), lin.mask
        sa, sb = M[:, :CODE].argmax(1), M[:, CODE:2 * CODE].argmax(1)
        rows = torch.arange(len(W))
        self.wa, self.wb = nn.Parameter(W[rows, sa].clone()), nn.Parameter(W[rows, CODE + sb].clone())
        self.wt = nn.Parameter(W[:, 2 * CODE:].clone()) if W.shape[1] > 2 * CODE else None
        self.bias = nn.Parameter(lin.bias.detach().clone())
        self.la = nn.Parameter(F.one_hot(sa, CODE).float())
        self.lb = None if tied else nn.Parameter(F.one_hot(sb, CODE).float())
        self.n_in = W.shape[1]

    def selections(self):
        return self.la.argmax(1), (self.la if self.lb is None else self.lb).argmax(1)

    @property
    def mask(self):
        sa, sb = self.selections()
        m = torch.zeros(len(sa), self.n_in)
        m[:, 2 * CODE:] = 1.
        rows = torch.arange(len(sa))
        m[rows, sa] = 1.
        m[rows, CODE + sb] = 1.
        return m

    def forward(self, x):
        Sa = straight_through(self.la)
        Sb = Sa if self.lb is None else straight_through(self.lb)
        out = self.wa * (x[:, :CODE] @ Sa.T) + self.wb * (x[:, CODE:2 * CODE] @ Sb.T) + self.bias
        return out if self.wt is None else out + x[:, 2 * CODE:] @ self.wt.T


def image_forward_channels_last(self, x):
    """ImageEncoder.forward with channels-last activations: the same function, faster convolutions on CPU, different
    floating-point rounding."""
    h = F.relu(self.c2(F.relu(self.c1(x[:, None].contiguous(memory_format=torch.channels_last))))).contiguous()
    return self.out(F.relu(self.fc(h.flatten(1))))


def rigl_groups(name, lin):
    """Column groups within which each unit keeps its fan-in: all columns (encoder), or each code operand (readers)."""
    n = lin.weight.shape[1]
    return [torch.arange(n)] if name.endswith(('/out', '/proj')) else [torch.arange(CODE), torch.arange(CODE, 2 * CODE)]


@torch.no_grad()
def rigl_step(lin, groups, opt, alpha, mode='zero', rng=None, p=None):
    """Per unit and group: drop the max(1, round(alpha k)) smallest-|w| active connections, grow as many inactive ones
    (dropped ones included) with the largest |dense gradient|. Grown connections start at w = 0 with zeroed Adam moments;
    a connection dropped and regrown at once is unchanged.
    mode 'carry' (k = 1 only): a grown connection takes the dropped connection's weight; its Adam moments are zeroed.
    mode 'random' (k = 1 only): no gradient; each connection moves with probability p to a uniformly drawn other
    coordinate, starting at w = 0 with zeroed moments; 'random-carry' likewise, keeping its weight.
    mode 'inplace': no gradient and no move; each code connection restarts at w = 0 with zeroed moments with probability p."""
    W, M = lin.weight, lin.mask
    if mode in ('random', 'random-carry'):
        return rewire_step(lin, groups, opt, rng, p, carry=mode == 'random-carry')
    if mode == 'inplace':
        return reset_step(lin, opt, rng, p)
    Gd = lin.probe.grad
    state = opt.state.get(W, {})
    for cols in groups:
        m = M[:, cols]
        k = int(m[0].sum())
        if k == len(cols):
            continue
        n = max(1, round(alpha * k))
        drop = torch.where(m > 0, W[:, cols].abs(), torch.inf).topk(n, 1, largest=False).indices
        after = m.clone().scatter_(1, drop, 0.)
        grow = torch.where(after > 0, -torch.inf, Gd[:, cols].abs()).topk(n, 1).indices
        new = after.clone().scatter_(1, grow, 1.)
        born = (new > 0) & (m == 0)
        rows, idx = born.nonzero(as_tuple=True)
        full = cols[idx]
        if mode == 'carry':
            assert n == 1 and k == 1
            W[rows, full] = W[rows, cols[drop[rows, 0]]]
        else:
            W[rows, full] = 0.
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state:
                state[key][rows, full] = 0.
        M[:, cols] = new
        lin.swaps += born.sum()


@torch.no_grad()
def rewire_step(lin, groups, opt, rng, p, carry=False):
    W, M = lin.weight, lin.mask
    state = opt.state.get(W, {})
    for cols in groups:
        m = M[:, cols]
        assert int(m[0].sum()) == 1
        old = m.argmax(1)
        move = torch.as_tensor(rng.random(len(m)) < p)
        new = (old + torch.as_tensor(rng.integers(1, len(cols), len(m)))) % len(cols)  # uniform over the other coordinates
        rows = move.nonzero(as_tuple=True)[0]
        if len(rows):
            M[rows, cols[old[rows]]] = 0.
            M[rows, cols[new[rows]]] = 1.
            W[rows, cols[new[rows]]] = W[rows, cols[old[rows]]] if carry else 0.
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key][rows, cols[new[rows]]] = 0.
            lin.swaps += len(rows)


@torch.no_grad()
def reset_step(lin, opt, rng, p):
    """Each active code connection (columns of both operands; g's relation-token columns excluded) restarts at w = 0 with
    zeroed Adam moments with probability p; the mask is unchanged."""
    W, M = lin.weight, lin.mask
    state = opt.state.get(W, {})
    hit = (M[:, :2 * CODE] > 0) & torch.as_tensor(rng.random((len(M), 2 * CODE)) < p)
    rows, cols = hit.nonzero(as_tuple=True)
    W[rows, cols] = 0.
    for key in ('exp_avg', 'exp_avg_sq'):
        if key in state:
            state[key][rows, cols] = 0.
    lin.swaps += len(rows)


def rigl_layers(enc, model):
    return {k: v for k, v in limited_layers(enc, model).items() if hasattr(v, 'swaps')}


def rigl_train_update(enc, model, opt, material, rng, compare):
    """TP.train_update with RigL connectivity updates every DT updates until END_FRACTION of training."""
    CURRENT['step'] += 1
    step, end = CURRENT['step'], END_FRACTION * 3 * D.STEPS
    layers = rigl_layers(enc, model) if step % DT == 0 and step <= end else {}
    for lin in layers.values():
        lin.collect, lin.probe = True, torch.zeros_like(lin.weight, requires_grad=True)
    out = TP.train_update(enc, model, opt, material, rng, compare)
    alpha = ALPHA * (1 + np.cos(np.pi * step / end)) / 2
    mode = RIGL_MODE.get(CURRENT['config'], 'zero')
    for name, lin in layers.items():
        lin.collect = False
        rng = (nc.stream('fanin_rewire', CURRENT['seed'], name, step) if mode in ('random', 'random-carry') else
               nc.stream('fanin_reset', CURRENT['seed'], name, step) if mode == 'inplace' else None)
        rigl_step(lin, rigl_groups(name, lin), opt, alpha, mode, rng, REWIRE_P.get(name, 0.) * RATE.get(CURRENT['config'], 1.))
        del lin.probe
    return out


def make_model(seed, config):
    enc, model = SP.make_model(seed)  # same global draws as the sequence pilot
    if CHANNELS_LAST:
        enc['image'].to(memory_format=torch.channels_last)
        enc['image'].forward = image_forward_channels_last.__get__(enc['image'])
    qe, qr, mode = CONFIGS[config]
    mkey = RIGL_BASE.get(config, config)  # a RigL arm starts from its random arm's masks
    if qe:
        for sense in ('image', 'sequence'):
            encoder_mask(enc[sense].out, qe, nc.stream('fanin_mask', seed, mkey, sense, 'out'))
            encoder_mask(model.proj[sense], qe, nc.stream('fanin_mask', seed, mkey, sense, 'proj'))
    if qr:
        key, base = START_OF.get(mode, (mkey, mode))
        for name, seq in (('g', model.g), ('P', model.P)):
            if base == 'tied':
                reader_mask(seq[0], qr, nc.stream('fanin_mask', seed, key, name))
            else:
                reader_mask(seq[0], qr, nc.stream('fanin_mask', seed, key, name, 'a'), nc.stream('fanin_mask', seed, key, name, 'b'))
            if mode in RANDOM_OF:
                seq[0] = SelectLinear(seq[0], tied=mode == 'learn-tied')
            if mode in FIXED_SHARE:
                rng = nc.stream('fanin_mask', seed, 'fixed' if mode == 'fixed-rigld' else mode, name)  # same order as 'fixed'
                fix_rows(seq[0], fixed_coordinate(seed, mode), FIXED_SHARE[mode], rng)
    if config in DENSE_READERS:
        for seq in (model.g, model.P):
            limit(seq[0], torch.ones_like(seq[0].weight), torch.arange(2 * CODE))
    if config in RIGL_BASE:
        for name, lin in limited_layers(enc, model).items():
            if hasattr(lin, 'mask') and name in RIGL_SITES.get(config, (name,)):
                lin.register_buffer('swaps', torch.zeros((), dtype=torch.long))
    if config == 'read-q16-untied-dead':
        for name, seq in (('g', model.g), ('P', model.P)):
            silence_units(seq[0], nc.stream('fanin_mask', seed, 'dead', name).permutation(seq[0].weight.shape[0])[:DEAD_UNITS[name]])
    return enc, model


@torch.no_grad()
def silence_units(lin, rows):
    """Units that never respond: every input masked (relation token included) and a negative bias. ReLU passes no
    gradient, so the bias stays fixed and the units stay silent, as RigL's dead units do."""
    rows = torch.as_tensor(np.sort(rows))
    lin.mask[rows] = 0.
    lin.bias[rows] = DEAD_BIAS


def gradual_share(step, end):
    """Share of units moved by update `step`: the integral of RigL's cosine drop fraction, normalized to 1 at `end`."""
    x = min(step / end, 1.)
    return x + np.sin(np.pi * x) / np.pi


@torch.no_grad()
def move_rows(lin, rows, d, opt):
    """Move each row's single code connection per operand onto dimension d, carrying its weight; Adam moments of every
    changed entry are reset."""
    W, M = lin.weight, lin.mask
    state = opt.state.get(W, {})
    for j in rows.tolist():
        for off in (0, CODE):
            old = off + int(M[j, off:off + CODE].argmax())
            new = off + d
            if old == new:
                continue
            W[j, new], W[j, old] = W[j, old].clone(), 0.
            M[j, old], M[j, new] = 0., 1.
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key][j, old] = state[key][j, new] = 0.


def gradual_train_update(enc, model, opt, material, rng, compare):
    """TP.train_update; at RigL's connectivity updates (every DT until END_FRACTION), move units onto the coordinate so
    that the moved share follows gradual_share."""
    CURRENT['step'] += 1
    step, end = CURRENT['step'], END_FRACTION * 3 * D.STEPS
    out = TP.train_update(enc, model, opt, material, rng, compare)
    if step % DT == 0 and step <= end:
        seed, mode = CURRENT['seed'], CONFIGS[CURRENT['config']][2]
        for name, seq in (('g', model.g), ('P', model.P)):
            m = seq[0].weight.shape[0]
            before = round(gradual_share(step - DT, end) * m)
            after = m if step + DT > end else round(gradual_share(step, end) * m)
            move_rows(seq[0], torch.as_tensor(gradual_order(seed, name, m)[before:after]), fixed_coordinate(seed, mode), opt)
    return out


def limited_layers(enc, model):
    return {'image/out': enc['image'].out, 'sequence/out': enc['sequence'].out, 'image/proj': model.proj['image'],
            'sequence/proj': model.proj['sequence'], 'g/first': model.g[0], 'P/first': model.P[0]}


CURRENT = {}
ORIG_RUN = D.run


def run(task):
    config, seed = task
    CURRENT['config'], CURRENT['seed'], CURRENT['step'] = config, seed, 0
    torch.set_num_threads(THREADS)
    DEV.install()
    mode = CONFIGS[config][2]
    D.train_update = rigl_train_update if config in RIGL_BASE else gradual_train_update if mode in GRADUAL else TP.train_update
    key, res, rt = ORIG_RUN(task)
    if DEV.ACTIVE:
        rt['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    return key, res, rt


def main():
    SB.install()
    X.make_model = lambda seed: DEV.build(lambda: make_model(seed, CURRENT['config']))
    TP.EXTRA_LOSS = SP.sequence_loss
    P.evaluate, D.evaluate_senses, D.train_update, D.batch_material = G.evaluate, SP.evaluate_senses, TP.train_update, SP.batch_material
    D.ARMS = {c: D.ARMS[BASE_ARM] for c in CONFIGS}
    if STAGE != 'stage1':
        D.SEEDS = CONFIRM_SEEDS
    if ONLY:
        picked = [t.split(':') for t in ONLY.split(',')]
        D.ARMS = {c: D.ARMS[c] for c in dict.fromkeys(c for c, _ in picked)}
        D.SEEDS = sorted({int(s) for _, s in picked})
    D.NAME = NAME
    D.run = DUR.durable(run, X.ROOT / 'runs' / NAME / 'done', '{0}-seed{1}')  # finished tasks survive a power cut
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, X.ROOT / 'runs' / NAME
    D.main()
    path = X.ROOT / 'reports' / f'{NAME}-runtime.json'
    rt = json.loads(path.read_text())
    rt.update({'threads_per_worker': THREADS, 'channels_last_image_encoder': CHANNELS_LAST, **DEV.info()})
    path.write_text(json.dumps(rt, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
