"""Preflight checks for the limited-memory runner (docs/limited-memory-proposal.md), smoke scale on the CPU.

Run: NSY_SMOKE=1 NSY_WORKERS=4 .venv/bin/python -m experiments.limited_memory_checks pilot main
Each check prints PASS or raises.
"""
import numpy as np
import torch

from experiments import limited_memory as LM
from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import generic_relations as G
from experiments import fan_in_pilot as FI

nc = X.nc
SEED = 300
RESULT = {}


def build(arm):
    LM.CURRENT.update(arm=arm, seed=SEED, step=0, r=None)
    torch.manual_seed(SEED)
    return X.make_model(SEED)


def ok(name):
    print('PASS', name)


def check_starting_weights():
    _, dense = build('dense')
    _, sp64 = build('sparse64')
    _, mixed = build('mixed')
    _, sp48 = build('sparse48')
    for n in ('g', 'P'):
        d, s64, m, s48 = (getattr(x, n) for x in (dense, sp64, mixed, sp48))
        assert torch.equal(m[0].mem.weight, d[0].weight[LM.STRUCT:]) and torch.equal(m[0].mem.bias, d[0].bias[LM.STRUCT:]), 'memory rows are the standard init'
        assert torch.equal(m[0].struct.weight, s64[0].weight[:LM.STRUCT]) and torch.equal(m[0].struct.mask, s64[0].mask[:LM.STRUCT]), 'structure rows are the first 48 of sparse64'
        assert torch.equal(s48[0].struct.weight, m[0].struct.weight) and s48[0].mem is None
        for i in (2, 4):
            assert torch.equal(m[i].weight, d[i].weight) and torch.equal(s64[i].weight, d[i].weight) and torch.equal(s48[i].weight, d[i].weight), 'later layers start equal'
    ok('arms start from the same weights; structure rows are sparse64\'s first 48')


def check_forward_and_silencing():
    _, mixed = build('mixed')
    _, sp48 = build('sparse48')
    x = torch.randn(7, 40)
    L = mixed.g[0]
    full = torch.cat([L.struct(x), L.mem(x)], 1)
    assert torch.equal(L(x), full)
    with LM.silenced(mixed, ('mem',)):
        assert torch.equal(mixed.g[0](x)[:, LM.STRUCT:], torch.zeros(7, LM.MEM)) and torch.equal(mixed.g[0](x)[:, :LM.STRUCT], full[:, :LM.STRUCT])
        mem_off = mixed.g(x)
    with LM.silenced(mixed, ('struct',)):
        assert torch.equal(mixed.g[0](x)[:, :LM.STRUCT], torch.zeros(7, LM.STRUCT))
    assert torch.equal(mixed.g[0].silence, set()) if False else mixed.g[0].silence == set()
    assert torch.equal(sp48.g(x), mem_off), 'sparse48 at the start is mixed with the memory part silenced'
    ok('SplitLinear forward, silencing, and sparse48 == mixed without memory at the start')


def check_decay():
    _, m = build('mixed')
    r = 2. ** (-1. / 500)
    L, out = m.g[0].mem, m.g[2]
    with torch.no_grad():
        L.weight.copy_(L.w0 + 1.); L.bias.copy_(L.b0 - 2.); out.weight[:, LM.STRUCT:] = 3.
        structure_before = m.g[0].struct.weight.clone(); other = out.weight[:, :LM.STRUCT].clone()
    LM.decay_step(m, r)
    assert torch.allclose(L.weight, L.w0 + r) and torch.allclose(L.bias, L.b0 - 2. * r) and torch.allclose(out.weight[:, LM.STRUCT:], torch.full_like(out.weight[:, LM.STRUCT:], 3. * r))
    assert torch.equal(m.g[0].struct.weight, structure_before) and torch.equal(out.weight[:, :LM.STRUCT], other), 'decay touches only the memory part'
    for _ in range(500):  # half-life: 500 steps halve the deviation
        LM.decay_step(m, r)
    assert abs(float((L.weight - L.w0).mean()) - r * 2 ** -1) < 1e-4
    ok('decay: w0 + (w - w0) r on memory incoming, r w on outgoing, nothing else; half-life H')


def check_training_invariants():
    """A short training run of each mixed variant: RigL keeps one connection per operand on structure rows only,
    the memory part stays dense, reset acts only on memory code weights, decay arms pull memory toward the start."""
    for arm in ('mixed', 'mixed-reset', 'mixed-h125'):
        enc, model, opt = LM.setup(arm, SEED)
        if arm == 'mixed-h125':
            LM.CURRENT['r'] = .5  # exaggerated decay: Adam moves a weight by about lr per step, so deviation stays within a few lr
        rng, crng = nc.stream('chk', 1), nc.stream('chk', 2)
        withheld = D.withheld_counts(SEED)
        for step in range(2 * FI.DT + 1):  # RigL/reset events at DT and 2 DT
            eps = P.episodes('english', withheld['english'], rng, D.BATCH_EP)
            LM.train_update(enc, model, opt, LM.SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
        for n, lin in LM.structure_layers(model).items():
            m = lin.mask
            assert m.shape[0] == LM.STRUCT and (m[:, :CODE].sum(1) == 1).all() and (m[:, CODE:2 * CODE].sum(1) == 1).all() and (m[:, 2 * CODE:] == 1).all(), 'RigL keeps k = 1 per operand; token columns stay'
            assert int(lin.swaps) > 0, 'RigL changed connections'
        for n, mem in LM.memory_layers(model).items():
            assert (mem.mask == 1).all(), 'memory part stays dense'
            if arm == 'mixed-reset':
                assert int(mem.swaps) > 0, 'reset happened'
            else:
                assert int(mem.swaps) == 0
        dev = max(float((mem.weight - mem.w0).abs().max()) for mem in LM.memory_layers(model).values())
        if arm == 'mixed-h125':
            assert dev <= 3 * P.LR, f'decayed memory weights stay within 3 lr of the start (max deviation {dev:.4f}, lr {P.LR})'
            continue
        assert dev > 3 * P.LR, 'without decay the memory weights drift beyond a few learning rates (the decay test below is sensitive)'
    ok('RigL on structure rows only (k = 1), memory dense, reset only in mixed-reset')


CODE = FI.CODE


def check_rehearsal_only_in_retraining():
    calls = []
    orig = LM.rehearsal_batch
    LM.rehearsal_batch = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        LM.CURRENT.update(arm='mixed')
        RESULT['mixed'] = LM.run_main(('mixed', SEED))[1]
    finally:
        LM.rehearsal_batch = orig
    assert len(calls) == LM.RELEARN, f'rehearsal drawn {len(calls)} times, expected {LM.RELEARN} (relearning only)'
    ok(f'rehearsal buffer sampled {len(calls)} times, all in relearning (blocks and delay: none)')


def check_matching_agrees_with_pilot():
    enc, model, _ = LM.setup('sparse64', SEED)
    withheld = D.withheld_counts(SEED)
    m = LM.matching(enc, model, withheld)
    iso = G.ORIG_EVALUATE(enc, model, withheld, langs=P.LANGS, full=False)['isolated']
    for k, v in m.items():
        assert abs(v['all'] - iso[k]) < 1e-9, (k, v['all'], iso[k])
        assert abs((v['even'] + v['odd']) / 2 - v['all']) < 1e-9
    ok('matching() equals the pilots\' isolated matching, and even/odd halves average to it')


def check_fork_parent_is_block3():
    res = RESULT['mixed']
    block3 = res['snapshots']['block3']['four_way']['full']['matching']
    ck = torch.load(LM.X.ROOT / 'runs' / LM.MAIN_NAME / 'parents' / f'mixed-seed{SEED}.pt', weights_only=False)
    LM.CURRENT.update(arm='mixed', seed=SEED, step=ck['step'], r=None)
    torch.manual_seed(SEED)
    enc, model = X.make_model(SEED)
    model.load_state_dict(ck['sri']); enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    again = LM.matching(enc, model, {k: tuple(v) for k, v in ck['withheld'].items()})
    assert again == block3, 'the saved parent reproduces the end-of-block-3 measures'
    ok('the fork parent reproduces the end-of-block-3 measures exactly')


def main():
    LM.install()
    X.prepare(); P.prepare(); D.prepare()
    check_starting_weights()
    check_forward_and_silencing()
    check_decay()
    check_training_invariants()
    check_matching_agrees_with_pilot()
    check_rehearsal_only_in_retraining()
    check_fork_parent_is_block3()


if __name__ == '__main__':
    main()
