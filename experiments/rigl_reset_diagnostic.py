"""Read-only final-checkpoint audit; does not infer unrecorded training dynamics."""
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import torch


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    root = Path(__file__).resolve().parents[1]
    rows, hashes = [], {}
    for seed in range(230, 240):
        p = root / 'runs/fan-in-pilot-v1-pilot-timing' / f'read-q16-untied-rigl-seed{seed}.pt'
        hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        state = torch.load(p, map_location='cpu', weights_only=False)['sri']
        row = {'seed': seed}
        for name in ('g', 'P'):
            W = state[name + '.0.weight'] * state[name + '.0.mask']
            bias = state[name + '.0.bias']
            zero_code = (W[:, :32] == 0).all(1)
            if name == 'P':
                max_pre = bias
            else:
                max_pre = (state['token.weight'] @ W[:, 32:].T + bias).max(0).values
            certified = zero_code & (max_pre <= 0)
            row[name] = {'units': len(bias), 'zero_code_input_rows': int(zero_code.sum()),
                         'certified_inactive_all_codes': int(certified.sum()),
                         'all_input_weights_zero': int((W == 0).all(1).sum()),
                         'nonpositive_bias_rows': int((bias <= 0).sum())}
        rows.append(row)
    summary = {name: {key: sum(r[name][key] for r in rows) / len(rows)
                      for key in ('zero_code_input_rows', 'certified_inactive_all_codes')}
               for name in ('g', 'P')}
    report = {'scope': 'Final state only; all real code inputs and the three learned g relation tokens',
              'seeds': rows, 'mean_counts_out_of_64': summary,
              'checkpoint_sha256': hashes, 'zero_test': 'Exact equality on masked weights; nonpositive maximum preactivation'}
    prefix = root / 'reports/rigl-reset-diagnostic'
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    paths = [Path(__file__), root / 'docs/rigl-after-timing-analysis.md', root / 'experiments/fan_in_pilot.py']
    runtime = {'command': '.venv/bin/python -m experiments.rigl_reset_diagnostic',
               'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()),
               'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(),
               'torch_build_config': torch.__config__.show(), 'threads': torch.get_num_threads(),
               'dtype': 'checkpoint float32, exact comparisons', 'device': 'cpu', 'workers': 1,
               'checkpoint_loads': 10, 'fits': 0, 'optimizer_steps': 0, 'tuning_candidates': 0,
               'seed_policy': 'All ten existing timing RigL seeds, no selection',
               'wall_seconds': time.perf_counter() - wall, 'process_cpu_seconds': time.process_time() - cpu,
               'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'scope': 'Whole main excluding import and final runtime write; disk read/hash/load included'}
    prefix.with_name(prefix.name + '-runtime.json').write_text(json.dumps(runtime, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'mean_counts_out_of_64': summary, 'seeds': rows, 'wall_seconds': runtime['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
