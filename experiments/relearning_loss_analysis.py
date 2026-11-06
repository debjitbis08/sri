"""Small deterministic checks for docs/relearning-loss-analysis-v1.md.

No training or checkpoint evaluation. Scientific results and runtime are separate.
Run: .venv/bin/python -m experiments.relearning_loss_analysis
"""
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    start, cpu = time.perf_counter(), time.process_time()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    gen = torch.Generator().manual_seed(42)
    c = torch.randn(9, 4, generator=gen, requires_grad=True)
    p = torch.randn(4, generator=gen, requires_grad=True)
    beta, y = .4, 7
    dist = (p - c).square().sum(1)
    loss = F.cross_entropy((-beta * dist)[None], torch.tensor([y]))
    gp, gc = torch.autograd.grad(loss, (p, c))
    prob = (-beta * dist).softmax(0).detach()
    target = F.one_hot(torch.tensor(y), 9)
    expected = 2 * beta * (target - prob)[:, None] * (c.detach() - p.detach())
    radial = (gc * c.detach()).sum() + (gp * p.detach()).sum()
    radial_formula = 2 * beta * (dist.detach()[y] - (prob * dist.detach()).sum())

    # A positive scalar line: individual L2 normalization destroys its order.
    line = torch.arange(1., 10.)[:, None]
    normalized = F.normalize(line, dim=1)

    # Local constraints on shared parameters, not independently movable codes.
    # Both modalities remain trainable, and form and collection features share W.
    x = torch.randn(9, 3, generator=gen)
    tap_noise = .05 * torch.randn(9, 2, generator=gen)
    wi = torch.randn(2, 3, generator=gen)
    theta = torch.cat([wi.flatten(), wi.flatten() + .1]).requires_grad_()

    def stats(t):
        image = x @ t[:6].reshape(2, 3).T
        taps = x @ t[6:].reshape(2, 3).T + tap_noise
        spread = (image - image.mean(0)).square().sum(1).mean()
        mismatch = (image - taps).square().sum(1).mean()
        return spread.log(), mismatch

    s, e = stats(theta)
    a, = torch.autograd.grad(s, theta, retain_graph=True)
    b, = torch.autograd.grad(e, theta)
    # Deliberately harmful proposed displacement: contracts and worsens alignment.
    n = b - a * (a @ b) / (a @ a)
    free = torch.randn(theta.shape, generator=gen)
    free = free - a * (a @ free) / (a @ a) - n * (n @ free) / (n @ n)
    proposal = -a + n * (1 + abs(float(a @ b)) / float(n @ n)) + free
    tangent = proposal - a * (a @ proposal) / (a @ a)
    corrected = tangent - n * torch.clamp(b @ tangent, min=0) / (n @ n)
    fd = {}
    for eps in (1e-2, 1e-3, 1e-4):
        sn, en = stats(theta.detach() + eps * corrected)
        fd[str(eps)] = {'log_spread_change': float(sn - s.detach()),
                        'alignment_error_change': float(en - e.detach())}

    # Predictor nonhomogeneity defeats lifting score-scale invariance to inputs.
    z = torch.tensor([1., 2.])
    predictor = lambda v: torch.relu(v + torch.tensor([2., -1.])) + 1
    homogeneity_error = float((predictor(.5 * z) - .5 * predictor(z)).norm())
    results = {
        'status': 'analytic toy checks; not evidence of full-system efficacy',
        'seed': 42, 'dtype': 'float64',
        'candidate_gradient_max_error': float((gc - expected).abs().max()),
        'joint_scale_derivative_error': float(abs(radial - radial_formula)),
        'individual_normalization_distinct_codes_on_positive_line': len(torch.unique(normalized, dim=0)),
        'nonhomogeneous_predictor_scale_error': homogeneity_error,
        'proposal_constraint_derivatives': [float(a @ proposal), float(b @ proposal)],
        'corrected_constraint_derivatives': [float(a @ corrected), float(b @ corrected)],
        'retained_update_norm': float(corrected.norm()),
        'finite_step_changes': fd,
    }
    assert results['candidate_gradient_max_error'] < 1e-12
    assert results['joint_scale_derivative_error'] < 1e-12
    assert max(abs(v) for v in results['corrected_constraint_derivatives']) < 1e-10
    root = Path(__file__).resolve().parents[1]
    path = root / 'reports/relearning-loss-analysis-v1.json'
    path.write_text(json.dumps(results, indent=2) + '\n')
    runtime = {'scope': 'whole-run toy derivative checks, excludes interpreter/import startup',
               'wall_seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu,
               'peak_rss_kib_linux_process': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(),
               'threads': 1, 'device': 'cpu', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'command': '.venv/bin/python -m experiments.relearning_loss_analysis',
               'training_fits': 0, 'checkpoint_reads': 0}
    path.with_name('relearning-loss-analysis-v1-runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
