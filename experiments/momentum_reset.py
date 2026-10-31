"""Momentum reset v1 (docs/momentum-reset-v1-contract.md): does zeroing the inherited Adam first moments reduce the early recall drop in real
context training, and at what cost? Branches start from the end of block 4 (Option A), seeds 340-349, the four arms of the forgetting study:
  ctx / ctx-m0 / ctx-fresh   context episodes with the inherited Adam state, with `exp_avg` zeroed, with the whole state cleared
  hold / hold-m0             context + A pairs every update, with and without the zeroed first moments
  zero-grad / zero-grad-m0   explicit zero gradients (no decay), with and without the zeroed first moments (the latter must move nothing)

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.momentum_reset pilot pairs
(NSY_SMOKE=1 for a short CPU run; NAP_ARMS and NAP_SEEDS override the defaults below.)
"""
import os

os.environ.setdefault('NAP_SEEDS', ','.join(map(str, range(340, 350))))
if not os.environ.get('NSY_SMOKE'):  # the contract's configuration, not the arbitrary-pairings defaults
    for k, v in (('NAP_OPTION', 'A'), ('NAP_N', '45'), ('NAP_L', '2000'), ('NAP_ARMS', 'sparse48,dense,mixed,mixed-h125')):
        os.environ.setdefault(k, v)

from experiments import arbitrary_pairings as AP
from experiments import forgetting_mechanism as FM
from experiments import number_systems as X

FM.SPECS = {
    'zero-grad': dict(zero_grad=True, decay=False),
    'zero-grad-m0': dict(zero_grad=True, decay=False, adam='m0'),
    'ctx': dict(train=True, decay=True),
    'ctx-m0': dict(train=True, decay=True, adam='m0'),
    'ctx-fresh': dict(train=True, decay=True, adam='fresh'),
    'hold': dict(train=True, decay=True, pairs='hold'),
    'hold-m0': dict(train=True, decay=True, pairs='hold', adam='m0'),
}
FM.DECAY_BRANCHES = ()
FM.RELEARN_ON = False
NAME = 'momentum-reset-v1' + ('-smoke' if FM.SMOKE else '') + os.environ.get('NFI_TAG', '')
RUNS = X.ROOT / 'runs' / NAME

if __name__ == '__main__':
    AP.NAME, AP.RUNS, AP.run_task = NAME, RUNS, FM.run_task
    AP.main()
