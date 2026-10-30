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

nc = X.nc
STAGE = sys.argv[2] if len(sys.argv) > 2 else 'stage1'
NAME = 'fan-in-pilot-v1-' + X.MODE + ('' if STAGE == 'stage1' else f'-{STAGE}') + ('-smoke' if X.SMOKE else '')
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
FIXED_SHARE = {'fixed': 1., 'fixed75': .75}  # fraction of reader units moved onto the fixed coordinate
RIGL_BASE = {'enc-q16-rigl': 'enc-q16', 'read-q4-untied-rigl': 'read-q4-untied',
             'read-q16-untied-rigl': 'read-q16-untied'}  # RigL arm -> random arm with the same start
CONFIGS = {'stage1': STAGE1, 'confirm': CONFIRM, 'rigl': RIGL, 'rigl1': RIGL1, 'fixed': FIXED}[STAGE]
CONFIRM_SEEDS = tuple(range(*{'confirm': (200, 210), 'rigl': (210, 220), 'rigl1': (220, 230), 'fixed': (230, 240)}.get(STAGE, (0, 0))))
DT, ALPHA, END_FRACTION = (10 if X.SMOKE else 100), .3, .75  # RigL: update interval, initial drop fraction, end of updates (fraction of training)
RANDOM_OF = {'learn-tied': ('read-q16', 'tied'), 'learn-untied': ('read-q16-untied', 'untied')}  # learned arm's starting selection
START_OF = {**RANDOM_OF, 'fixed': ('read-q16-untied', 'untied'), 'fixed75': ('read-q16-untied', 'untied')}  # masks an arm starts from
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


def fixed_coordinate(seed):
    return int(nc.stream('fanin_mask', seed, 'fixed-coordinate').integers(CODE))


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
def rigl_step(lin, groups, opt, alpha):
    """Per unit and group: drop the max(1, round(alpha k)) smallest-|w| active connections, grow as many inactive ones
    (dropped ones included) with the largest |dense gradient|. Grown connections start at w = 0 with zeroed Adam moments;
    a connection dropped and regrown at once is unchanged."""
    W, M, Gd = lin.weight, lin.mask, lin.probe.grad
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
        W[rows, full] = 0.
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state:
                state[key][rows, full] = 0.
        M[:, cols] = new
        lin.swaps += born.sum()


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
    for name, lin in layers.items():
        lin.collect = False
        rigl_step(lin, rigl_groups(name, lin), opt, alpha)
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
                fix_rows(seq[0], fixed_coordinate(seed), FIXED_SHARE[mode], nc.stream('fanin_mask', seed, mode, name))
    if config in RIGL_BASE:
        for lin in limited_layers(enc, model).values():
            if hasattr(lin, 'mask'):
                lin.register_buffer('swaps', torch.zeros((), dtype=torch.long))
    return enc, model


def limited_layers(enc, model):
    return {'image/out': enc['image'].out, 'sequence/out': enc['sequence'].out, 'image/proj': model.proj['image'],
            'sequence/proj': model.proj['sequence'], 'g/first': model.g[0], 'P/first': model.P[0]}


CURRENT = {}
ORIG_RUN = D.run


def run(task):
    config, seed = task
    CURRENT['config'], CURRENT['step'] = config, 0
    torch.set_num_threads(THREADS)
    D.train_update = rigl_train_update if config in RIGL_BASE else TP.train_update
    return ORIG_RUN(task)


def main():
    SB.install()
    X.make_model = lambda seed: make_model(seed, CURRENT['config'])
    TP.EXTRA_LOSS = SP.sequence_loss
    P.evaluate, D.evaluate_senses, D.train_update, D.batch_material = G.evaluate, SP.evaluate_senses, TP.train_update, SP.batch_material
    D.ARMS = {c: D.ARMS[BASE_ARM] for c in CONFIGS}
    if STAGE != 'stage1':
        D.SEEDS = CONFIRM_SEEDS
    D.run = run
    D.NAME = NAME
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, X.ROOT / 'runs' / NAME
    D.main()
    path = X.ROOT / 'reports' / f'{NAME}-runtime.json'
    rt = json.loads(path.read_text())
    rt.update({'threads_per_worker': THREADS, 'channels_last_image_encoder': CHANNELS_LAST})
    path.write_text(json.dumps(rt, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
